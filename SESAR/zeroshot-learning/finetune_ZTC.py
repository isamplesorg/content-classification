from argparse import ArgumentParser
import pandas as pd
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments)
from datasets import Dataset, DatasetDict
import logging
import torch
import os
import numpy as np
from datasets import load_metric
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
os.environ["WANDB_MODE"]="disabled"

"""## Model training"""
if torch.cuda.is_available():    
    # GPU is available
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
def convert_dataframe_format(dataframe, template_type , config):
    """ Convert dataframe into correct format for textual entailment task.
    Each value in the negative_sample will be converted as a new row as result. 
    The labels will be also converted to a format that corrsponds to the pretrained model that is used.
    Negative samples should be converted to NEUTRAL label.
    
    Args:
        dataframe : dataframe to convert format
        template_type : hypothesis template type (A/B/C)
        config : model config that is used to determine the labels
    """
    # positive sample
    pos_column = 'concatenated_text_' + template_type
    pos_text = dataframe[pos_column].values.tolist()
    text = pos_text
    labels = [config.label2id['ENTAILMENT']] * len(pos_text)
    
    # add negative sample as each row
    neg_column = 'negative_sample_' + template_type
    neg_text = dataframe[neg_column].values.tolist()
    text += neg_text
    labels += [config.label2id['NEUTRAL']] * len(neg_text)

    # generate new dataframe
    data = {'text': text, 'label': labels}
    df = pd.DataFrame(data)
    # shuffle data
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def create_datasets(tokenizer, train_df, dev_df, max_length):
    """ Generate dataset dict that is going to be used by the trainer during finetuning """
    def tokenize(batch):
        return tokenizer(batch['text'], truncation='only_first', padding='max_length', max_length= max_length)
    ds = DatasetDict()
    ds['train'] = Dataset.from_pandas(train_df)
    ds['validation'] = Dataset.from_pandas(dev_df)
    
    train_ds = ds['train'].map(
        tokenize,
        batched=True,
    )
    dev_ds = ds['validation'].map(
        tokenize,
        batched=True,
    )
    return train_ds, dev_ds

def compute_metrics(eval_pred):
    metric = load_metric('glue', 'rte') # textual entailment task
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

    
def finetune_ZTC_model(train_df, dev_df, model_name, template_type, num_epochs, lr_rate, weight_decay, train_batch_size, eval_batch_size, max_length):
    # load pretrained model and tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model is finetuned on our domain specific NLI data
    num_labels = 3 # entailment or neutral 
    model = (
        AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = num_labels)
    )
    config = model.config
    logging.info(f"Loaded pretrained model {model_name}")
    # preprocess dataset for textual entailment finetuning task
    train_df = convert_dataframe_format(train_df, template_type, config)
    dev_df = convert_dataframe_format(dev_df, template_type, config)
    
    train_ds, dev_ds = create_datasets(tokenizer, train_df, dev_df, max_length)
    logging.info(f"Dataset size : train - {len(train_ds)}, dev - {len(dev_ds)}")
    #### CONDUCT TRAINING ####
    output_dir =  template_type + "_" + model_name + "_" + str(num_epochs) + "_" + str(lr_rate) + "_" + str(weight_decay) + "_" + str(train_batch_size)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    metric_name = "accuracy"
    training_args = TrainingArguments(
        output_dir=output_dir,
        log_level='error',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        # warmup_steps=500, 
        gradient_accumulation_steps=1, # batch size * accumulation_steps = total batch size
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    # store for future 
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return output_dir 

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--hypothesis_template_type", type=str, default='A')
    parser.add_argument("--model_name", type=str, default='roberta-large-mnli')
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--dev_data_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()
    
    # load dataset
    dev_df = pd.read_csv(args.dev_data_dir)
    train_df = pd.read_csv(args.train_data_dir)
    # finetune the textual entailment model on the dataset 
    output_dir = finetune_ZTC_model(train_df, dev_df, model_name=args.model_name, template_type=args.hypothesis_template_type,num_epochs=args.num_epochs, lr_rate=args.lr_rate, train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,weight_decay=args.weight_decay,max_length=args.max_length)
    logging.info(f"Saved finetuned model in {output_dir}")
