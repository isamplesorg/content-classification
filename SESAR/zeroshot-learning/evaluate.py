from argparse import ArgumentParser
import pandas as pd
from transformers import (AutoTokenizer, pipeline,AutoModelForSequenceClassification, Trainer, TrainingArguments)
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
from sklearn.metrics import classification_report
import logging
import torch
import os
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
import numpy as np 
from sklearn.preprocessing import MultiLabelBinarizer
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
os.environ["WANDB_MODE"]="disabled"


THRESHOLD = 0.5
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
    
def get_multilabel_predictions(predictions, THRESHOLD):
    """Get all predictions by conducting multilabel classification"""
    predicted_labels = [] 
    for pred in predictions:
        # get predictions that have probability larger than THRESHOLD
        indices = [i for i, val in enumerate(pred['scores']) if val >= THRESHOLD]
        prediction = [pred['labels'][i] for i in indices]
        predicted_labels.append(prediction)
    return predicted_labels

def get_predictions(output_dir, test_df, template_type, label_names, batch_size, max_length):
    """ Get the zero shot predictions by applying the model to the full label space 
    
    Args:
        multilabel : whether to solve the problem as multilabel or not. 
        output_dir : directory(label-partially-unseen) that stores the finetuned model or the pretrained model name(label-fully-unseen) we want to use for prediction
        test_df : test dataset used for evaluation 
        template_type : hypothesis template type (A/B/C)
        label_names : list of labels that the model will use as label space
        batch_size : batch size that is used in prediction
        max_length : max length of tokens that are used during tokenization of input text 
    """
    device = 0 if torch.cuda.is_available() else -1
    # load saved tokenizer and classifier
    tokenizer = AutoTokenizer.from_pretrained('roberta-large-mnli',use_fast=True, model_max_length=max_length) 
    model = (
        AutoModelForSequenceClassification.from_pretrained(output_dir, num_labels = 3)
    )
    test_args = TrainingArguments(
        output_dir = "./results",
        do_train = False,
        do_predict = True,
        per_device_eval_batch_size = 64,
    )
    trainer = Trainer(model = model, args =test_args)
    # load test dataset
    test_col = 'concatenated_text_' + template_type
    if template_type=='C':
        test_col = 'concatenated_text_B' # no column for C exists for the test set 
    # select hypothesis template
    if template_type == "A":
      hypothesis_template = "Material:{}."
    elif template_type == "B":
      hypothesis_template = "The material of this physical sample is {}."
    else:
      hypothesis_template = "The kind of material that constitutes this physical sample is {}."
    # load test dataset text 
    test_text = test_df[test_col].values.tolist()
    final_predictions = []
    idx = 0
    recall_k = [] # store for getting average of recall@k
    for text in test_text:
        test_batch = []
        # check entailment for all possible labels
        for label in gold_label_names:
            # select hypothesis template
            if template_type == "A":
                hypothesis_template = "Material:" + label + ".</s>"
            elif template_type == "B":
                hypothesis_template = "The material of this physical sample is " + label + ".</s>"
            else:
                hypothesis_template = "The kind of material that constitutes this physical sample is " + label + ".</s>"
            to_test = text + "<s>" + hypothesis_template
            test_batch.append(to_test)
        # prediction for this instance
        test_encodings = tokenizer(test_batch, truncation=True, padding=True)
        test_dataset = Dataset.from_dict(test_encodings)
        predictions = trainer.predict(test_dataset)
        logits = predictions.predictions
        # apply sigmoid + threshold
        probabilities = torch.softmax(torch.tensor(logits), dim=1)
        #turn predicted id's into actual label names
        entailment =np.argmax(probabilities, axis=1).tolist()
        pos_labels = [i for i, x in enumerate(entailment) if x == 2] # entailment
        predicted_labels = [gold_label_names[i] for i in pos_labels]
        final_predictions.append(predicted_labels)
        #gold_labels = test_gold_labels[idx]
        
        ######### EXTRACT TOP K RECALL ######## 
        # extract entailment probabilities 
        # probabilities_matrix = []
        # for i, label in enumerate(gold_label_names):
        #     # Extract the probabilities of entailment for the current label
        #     entailment_probabilities = probabilities[i][2].item()
        #     probabilities_matrix.append(entailment_probabilities)
        # probabilities_matrix = np.array(probabilities_matrix)
        # top5_probabilities, top5_indices = torch.topk(torch.from_numpy(probabilities_matrix), k=5)
        # top5_probabilities = top5_probabilities.tolist()
        # top5_indices = top5_indices.tolist()
        # top5_predictions = [gold_label_names[i] for i in top5_indices]
        # # calculate recall @ 5 of this instance
        # correct_at_5 = [x for x in top5_predictions if x in gold_labels]
        # recall_at_5 = len(correct_at_5) / len(gold_labels)
        # #print(f"Recall at 5 : {recall_at_5}")
        # recall_k.append(recall_at_5)
        idx += 1
        logging.info(f"{predicted_labels}")
        logging.info(f"{idx}-th prediction is done")
    #logging.info(f"Average recall@k : {sum(recall_k) / len(recall_k)}", )
    return final_predictions

def evaluate_classification_performance(predicted_labels, gold_labels, gold_label_names):
    mlb = MultiLabelBinarizer()
    # Fit the MultiLabelBinarizer on your labels and transform them into one-hot vectors
    mlb.fit([gold_label_names]) 
    gold_labels = mlb.transform(gold_labels)
    predicted_labels = mlb.transform(predicted_labels)
    target_names = mlb.classes_
    accuracy = accuracy_score(gold_labels, predicted_labels)
    report = classification_report(gold_labels, predicted_labels, target_names=target_names, output_dict=True)
    logging.info(classification_report(gold_labels, predicted_labels, target_names = target_names))
    for key, score in report.items():
        if type(score)==dict:
            logging.info(f"{key:<30}  {score['precision']:.3f} {score['recall']:.3f} {score['f1-score']:.3f}")
    logging.info(f"Accuracy : {accuracy}")
  
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--hypothesis_template_type", type=str, default='C')
    parser.add_argument("--label_file", type=str)
    parser.add_argument("--test_dataset_dir", type=str)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--depth_level", type=int,default=1)
    parser.add_argument("--output_dir", type=str, default='roberta-large-mnli')

    args = parser.parse_args()
    
    # load dataset
    test_df = pd.read_csv(args.test_dataset_dir)
    test_df = test_df.fillna("")
    test_df = test_df[(test_df['description_material'] != '')]
    # get subset of data
    #test_df = test_df.groupby('description_material').sample(n=500, random_state=42, replace=True) 
    logging.info("Test data size : ", test_df.shape)
   
    leaf_label_file = "leaf_labels_replaced.txt"
    leaf_label_names = open(leaf_label_file).read().splitlines()
    label_col_name ="label_list"
    # use the stored label space
    gold_label_names = open(args.label_file).read().splitlines()
    logging.info(f"Total {len(gold_label_names)} candidate labels to predict: {gold_label_names}")
    # select leaf label or non-leaf label
    test_gold_labels = [x.split("/") for x in test_df[label_col_name].values.tolist()]
    final = []
    for label_lst in test_gold_labels:
        temp = []
        for label in label_lst:
            if label not in leaf_label_names: # only include leaf labels
                temp.append(label)
        final.append(temp)
    test_gold_labels = final
    #print("Test gold labels", test_gold_labels)
    predicted_labels = get_predictions(args.output_dir, test_df, template_type=args.hypothesis_template_type, label_names=gold_label_names, batch_size=args.eval_batch_size, max_length=args.max_length)
    evaluate_classification_performance(predicted_labels, test_gold_labels, gold_label_names)
