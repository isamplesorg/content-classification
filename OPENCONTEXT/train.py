import argparse
import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer,BertForSequenceClassification,Trainer, TrainingArguments,BertConfig,EarlyStoppingCallback
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
import torch.nn as nn
from sklearn.utils import compute_class_weight
from sklearn.metrics import f1_score, classification_report
import pickle
from datasets import load_metric
import pickle

with open('data/material_additional_text.pkl', 'rb') as f:
  additional_material_text= pickle.load(f)
with open('data/sample_additional_text.pkl', 'rb') as f:
  additional_sample_text= pickle.load(f)

with open('data/bm_material_mapping.pkl', 'rb') as f:
  material_mapping= pickle.load(f)
with open('data/bm_sample_mapping.pkl', 'rb') as f:
  sample_mapping= pickle.load(f)

os.environ["WANDB_DISABLED"] = "true"

f1_metric =load_metric("f1")
# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
#create custom dataset 
class MulticlassDataset(Dataset):

    def __init__(self, encodings, labels):
      self.encodings = encodings
      self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
        
le = preprocessing.LabelEncoder()
def preprocess(dataframe, LABEL_COLUMN):
  #map the labels to higher level
  labels =[x.lower().strip() for x in dataframe[LABEL_COLUMN].values.tolist()]
  if LABEL_COLUMN == "Consists of_label":
    labels = [material_mapping[x] for x in labels]
  else:
    labels = [sample_mapping[x] if x in sample_mapping else "other" for x in labels  ]
  #convert labels into integers
  le.fit(labels)

  dataframe[LABEL_COLUMN] = le.transform(labels)
  # print(len(list(le.classes_)))
  #split data to training df, val df, test df
  train_df, dev_df, test_df =  np.split(dataframe.sample(frac=1, random_state=42),[int(.6*len(dataframe)), int(.8*len(dataframe))])
 
  return train_df, dev_df, test_df


def create_dataset(dataframe, tokenizer,LABEL_COLUMN):
  MAX_LENGTH = 256
  inputs = {
          "input_ids":[],
          "attention_mask":[]
        }
  COLUMNS = ['late bce/ce', 'early bce/ce', 'project label', 'Temporal Coverage_label',  'Has taxonomic identifier_label','Has anatomical identification_label', 'Consists of_label',  'Has type_label', 'item category', 'context label']
  features = dataframe.loc[:, dataframe.columns != LABEL_COLUMN]
  feature_columns = [x for x in list(features) if x in COLUMNS]
  def create_concatenated_text(dataframe):
    """combine the columns text to create a single sentence"""
    if LABEL_COLUMN=="Consists of_label":
      additional_text = additional_material_text
    else:
      additional_text = additional_sample_text

    additional_cnt = 0
    sents= [] #text that is a concatenation of all columns
    for _, row in dataframe.iterrows():
      combined = ""
      #check if additional text is available
      if row['citation uri'] in additional_text:
        additional_cnt+=1
        combined+= additional_text[row['citation uri']]
      #add dataset info
      for idx, col in enumerate(feature_columns):
        row_value = row[col]
        if row_value!="" and type(row_value)==str:
          if idx != len(feature_columns)-1:
            combined+= row_value +" , "
          else:
            combined+= row_value
      sents.append(combined)
    return sents
  sents = create_concatenated_text(dataframe)
  for sent in sents:
    tokenized_input = tokenizer(sent,max_length=MAX_LENGTH, padding='max_length', truncation=True)
    inputs["input_ids"].append(torch.tensor(tokenized_input["input_ids"]))
    inputs["attention_mask"].append(torch.tensor(tokenized_input["attention_mask"]))

  labels = torch.tensor(dataframe[LABEL_COLUMN].values.tolist())

  return MulticlassDataset(inputs,labels)

def get_class_weights(dataframe,LABEL_COLUMN):
  """computes the class weight and returns a list to account for class imbalance """
  labels = torch.tensor(dataframe[LABEL_COLUMN].values.tolist())
  class_weights=compute_class_weight( class_weight ='balanced',classes = np.unique(labels),y = labels.numpy())
  class_weight_dict = dict(zip(np.unique(labels), class_weights))
  total_class_weights =[]
  for i in range(len(le.classes_)):
    if i not in class_weight_dict:
      total_class_weights.append(1) #class_weight 1 for unseen labels
    else:
      total_class_weights.append(class_weight_dict[i])
  total_class_weights =torch.tensor(total_class_weights,dtype=torch.float).to(device)
  return total_class_weights

def create_custom_trainer(class_weights):
  """creates custom trainer that accounts for class imbalance"""
  class CustomTrainer(Trainer):
      def compute_loss(self, model, inputs, return_outputs=False):
          labels = inputs.get("labels")
          # forward pass
          outputs = model(**inputs)
          logits = outputs.get("logits")
          # compute custom loss 
          loss_fct = nn.CrossEntropyLoss(weight=class_weights)
          loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
          return (loss, outputs) if return_outputs else loss
  return CustomTrainer

def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  print(predictions.shape, labels.shape)
  return  f1_metric.compute(predictions=predictions, references=labels, average="macro")

def train(LABEL_COLUMN, dataframe, tokenizer, batch_size, learning_rate, epochs, train_mode,OUTPUT_DIR):

  train_df, dev_df, test_df = preprocess(dataframe,LABEL_COLUMN)
  train_dataset = create_dataset(train_df, tokenizer,LABEL_COLUMN)
  dev_dataset = create_dataset(dev_df,tokenizer,LABEL_COLUMN)
  test_dataset = create_dataset(test_df,tokenizer, LABEL_COLUMN)

  #load model
  model = BertForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels = len(le.classes_), )

  # Tell pytorch to run this model on the GPU.
  desc = model.cuda()

  training_args = TrainingArguments(
          output_dir= OUTPUT_DIR,     # output directory
          num_train_epochs=epochs,              # total number of training epochs
          per_device_train_batch_size=batch_size,  # batch size per device during training
          per_device_eval_batch_size=batch_size,   # batch size for evaluation
          learning_rate = learning_rate,
          warmup_steps=500,                # number of warmup steps for learning rate scheduler
          weight_decay=0.01, 
          metric_for_best_model = 'f1',
          load_best_model_at_end=True,            
          logging_dir=OUTPUT_DIR,            # directory for storing logs
          logging_steps=10,
          evaluation_strategy = "steps", 
          eval_steps = 50,
          do_train = True,
          do_eval = True,
  )
  #get class weight
  class_weights = get_class_weights(train_df,LABEL_COLUMN)
  CustomTrainer = create_custom_trainer(class_weights)

  if train_mode == "custom":
    trainer = CustomTrainer(model = model, args =training_args, train_dataset=train_dataset, eval_dataset=dev_dataset,compute_metrics=compute_metrics,callbacks = [EarlyStoppingCallback(early_stopping_patience=3)])
  else:
    trainer = Trainer(model=model, args =training_args, train_dataset=train_dataset, eval_dataset=dev_dataset,compute_metrics=compute_metrics,callbacks = [EarlyStoppingCallback(early_stopping_patience=3)])
  trainer.train()

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--nb_epochs",
                        default=None,
                        type=int,
                        required=True)

    parser.add_argument("--batch_size",
                        default=None,
                        type=int,
                        required=True)
    
    parser.add_argument("--lr_rate",
                        default=None,
                        type=float,
                        required=True)

    parser.add_argument("--label_type",
                        default=None,
                        type=str,
                        required=True)

    parser.add_argument("--train_mode",
                        default=False,
                        type=str,
                        help="Whether we account for class imbalance during training by using a custom trainer (custom) or not (none)",
                        required=False)

    parser.add_argument("--output_dir",
                    default=False,
                    type=str,
                    help="Output directory where the model checkpoint will be saved",
                    required=True)


    args = parser.parse_args()

    
    LABEL_COLUMN=""
    DATA_LOC = ""
    if args.label_type=="material":
      LABEL_COLUMN="Consists of_label"
      DATA_LOC = "data/processed_OPENCONTEXT_material_labeled.csv"
    else:
      LABEL_COLUMN="Has type_label"
      DATA_LOC = "data/processed_OPENCONTEXT_specimen_labeled.csv"
      
    #load tokenizer 
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True, use_fast=True)

    #load data in dataframe
    df = pd.read_csv(DATA_LOC)

    train(LABEL_COLUMN, df, tokenizer, args.batch_size,args.lr_rate, args.nb_epochs, args.train_mode, args.output_dir)



if __name__ == '__main__':
    main()


