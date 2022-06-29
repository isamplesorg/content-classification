import argparse
import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer,BertForSequenceClassification,Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
import torch.nn as nn
from sklearn.utils import compute_class_weight
from sklearn.metrics import f1_score, classification_report
import pickle

os.environ["WANDB_MODE"]="disabled"

"""## Model training"""
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
def preprocess(dataframe, selected_material_type=None):
  #convert the dataframe labels accordingly by the material type
  if selected_material_type!="None":
    new_df = dataframe.copy()
    for _, row in new_df.iterrows():
      if row['description_material'].split("_")[0] == selected_material_type:
        continue #leave the label
      else:
        row['description_material']="None" #set none as label
  else:
    new_df = dataframe.copy()   #flattened label version 
  new_df
  #convert labels into integers
  le.fit(new_df.description_material)

  new_df['description_material'] = le.transform(new_df.description_material)
  print(selected_material_type, " number of labels: ", len(le.classes_))
  #split data to training df, val df, test df
  train_df, dev_df, test_df =  np.split(new_df.sample(frac=1, random_state=42),[int(.6*len(new_df)), int(.8*len(new_df))])
 
  return train_df, dev_df, test_df


def create_dataset(dataframe, tokenizer):
  MAX_LENGTH = 256
  inputs = {
          "input_ids":[],
          "attention_mask":[]
        }
  features_columns =[x for x in dataframe.columns.values if x != 'description_material' and x.startswith("description")]
  def create_concatenated_text(dataframe):
    """combine the columns text to create a single sentence"""
    sents= [] #text that is a concatenation of all columns
    for _, row in dataframe.iterrows():
      combined = ""
      for col in features_columns:
        row_value = row[col]
        if row_value!="" and type(row_value)==str:
          combined+= row_value +" , "
      sents.append(combined)
    return sents
  #sents = create_concatenated_text(dataframe)
  sents = dataframe['concatenated_text'].values.tolist()
  for sent in sents:
    tokenized_input = tokenizer(sent,max_length=MAX_LENGTH, padding='max_length', truncation=True)
    inputs["input_ids"].append(torch.tensor(tokenized_input["input_ids"]))
    inputs["attention_mask"].append(torch.tensor(tokenized_input["attention_mask"]))

  labels = torch.tensor(dataframe['description_material'].values.tolist())

  return MulticlassDataset(inputs,labels)

def get_class_weights(dataframe):
  """computes the class weight and returns a list to account for class imbalance """
  labels = torch.tensor(dataframe['description_material'].values.tolist())
  class_weights=compute_class_weight( class_weight ='balanced',classes = np.unique(labels),y = labels.numpy())

  total_class_weights =torch.tensor(class_weights,dtype=torch.float).to(device)
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

def eval(selected_type, dataframe, tokenizer, batch_size, train_mode, MODEL_DIR, OUTPUT_DIR):

  train_df, dev_df, test_df = preprocess(dataframe,selected_type)
  test_dataset = create_dataset(test_df,tokenizer)

  #load model
  model = BertForSequenceClassification.from_pretrained(MODEL_DIR, num_labels = len(le.classes_), )

  # Tell pytorch to run this model on the GPU.
  desc = model.cuda()

  test_args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    do_train = False,
    do_predict = True,
    per_device_eval_batch_size = batch_size,   
  )
  #get class weight
  class_weights = get_class_weights(train_df)
  CustomTrainer = create_custom_trainer(class_weights)

  if train_mode == "custom":
    trainer = CustomTrainer(model = model, args =test_args)
  else:
    trainer = Trainer(model = model, args =test_args)

  #conduct evaluation 
  keys = []
  precision = []
  recall = []
  f1 = []
  
  logits = trainer.predict(test_dataset)[0] #get the logits 
  test_pred = np.argmax(logits,axis=-1)
  y_test= torch.tensor(test_df['description_material'].values.tolist())
  res = classification_report(y_test,test_pred,output_dict=True)
  for key, score in res.items():
    if key.isdigit():
      keys.append((le.inverse_transform([int(key)])[0]))
      precision.append(round(score['precision'],2))
      recall.append(round(score['recall'],2))
      f1.append(round(score['f1-score'],2))
      print("%s \t\t\t %0.2f \t %0.2f \t %0.2f"% (le.inverse_transform([int(key)])[0],score['precision'], score['recall'], score['f1-score']))
  #write the results to excel and save
  result_df = pd.DataFrame(data=zip(keys,precision,recall,f1), columns=['label','precision','recall','f1'])
  result_output_dir =OUTPUT_DIR+"/SESAR_evaluation_result.xlsx"
  result_df.to_excel(result_output_dir)
  print("Macro average: ",f1_score(y_test,test_pred,average='macro'))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters

    parser.add_argument("--batch_size",
                        default=None,
                        type=int,
                        required=True)

    parser.add_argument("--train_mode",
                        default=False,
                        type=str,
                        help="Whether we account for class imbalance during training by using a custom trainer (custom) or not (none)",
                        required=False)

    parser.add_argument("--model_dir",
                    default=False,
                    type=str,
                    help="Directory where the finetuned model is saved",
                    required=True)

    parser.add_argument("--output_dir",
                    default=False,
                    type=str,
                    help="Output directory where the model checkpoint will be saved",
                    required=True)


    args = parser.parse_args()

    df = pd.read_csv("data/SESAR_dataset.csv")
    df = df.fillna("")
    #remove rows that do not have a material type
    df = df[df["description_material"]!=""]
  
    #load tokenizer 
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True, use_fast=True)

    eval(args.material_type, df, tokenizer, args.batch_size,args.lr_rate, args.nb_epochs, args.train_mode, args.model_dir, args.output_dir)




if __name__ == '__main__':
    main()

