from argparse import ArgumentParser
import pandas as pd
from transformers import (AutoTokenizer, pipeline)
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
from sklearn.metrics import classification_report
import logging
import torch
import os
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
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

def get_zero_shot_predictions(multilabel,output_dir, test_df, template_type, label_names, batch_size, max_length):
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
    tokenizer = AutoTokenizer.from_pretrained(output_dir, use_fast=True, model_max_length=max_length)
    classifier = pipeline("zero-shot-classification", model=output_dir, tokenizer=tokenizer, device=device)
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
    prefix = "<s>"
    suffix = "</s>"
    test_text = test_df[test_col].values.tolist()
    # strip prefix
    test_text = [text[len(prefix):][:-len(suffix)] for text in test_text]
    test_ds = Dataset.from_dict({'text': test_text })
    # get zero-shot predictions 
    preds_list = []
    for text, output in tqdm(zip(test_text, classifier(KeyDataset(test_ds, 'text'), batch_size=batch_size, hypothesis_template = hypothesis_template, candidate_labels=label_names, multi_label=multilabel)),
                             total=len(test_ds), desc="SESAR Zero Shot"):
        preds_list.append(output)
    if not multilabel:
        # get a single predicted label
        return [x['labels'][0] for x in preds_list]
    else:
        return get_multilabel_predictions(preds_list, THRESHOLD)

def evaluate_classification_performance(multilabel, predicted_labels, gold_labels, gold_label_names):
    target_names = None
    if multilabel:
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
    parser.add_argument("--hypothesis_template_type", type=str, default='A')
    parser.add_argument("--label_file", type=str)
    parser.add_argument("--test_dataset_dir", type=str)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--multilabel", type=bool,default=True)
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
   
    prefix = "mat:"
    if args.multilabel:
        label_col_name ="label_list"
        # use the stored label space
        gold_label_names = open(args.label_file).read().splitlines()
    else:
        # using specified depth level to restrict the label space 
        if args.depth_level == 1:
            label_col_name = "description_material_depth_1"
        elif args.depth_level == 2:
            label_col_name = "description_material_depth_2"
        else:
            label_col_name = "description_material_depth_3"
        gold_label_names = [x for x in list(set(test_df[label_col_name].values.tolist()))] # all possible gold labels
    logging.info(f"Total {len(gold_label_names)} candidate labels to predict: {gold_label_names}")
    # Evaluate performance 
    predicted_labels = get_zero_shot_predictions(args.multilabel, args.output_dir, test_df, template_type=args.hypothesis_template_type, label_names=gold_label_names, batch_size=args.eval_batch_size, max_length=args.max_length)
    if args.multilabel:
        test_gold_labels = [x.split("/") for x in test_df[label_col_name].values.tolist()]
    else:
        test_gold_labels = [x for x in test_df[label_col_name].values.tolist()]
    evaluate_classification_performance(args.multilabel,predicted_labels, test_gold_labels, gold_label_names)
