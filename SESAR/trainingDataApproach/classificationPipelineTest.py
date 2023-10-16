import datasets
from transformers import pipeline
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from transformers.pipelines.pt_utils import KeyDataset
from torch.utils.data import Dataset
import pandas as pd
#from sklearn import preprocessing
import time

classdict = {"LABEL_0":"mat:anthropogenicmetal",
"LABEL_1":"mat:biogenicnonorganicmaterial",
"LABEL_2":"mat:gas",
"LABEL_3":"mat:liquidwater",
"LABEL_4":"mat:material",
"LABEL_5":"mat:mineral",
"LABEL_6":"mat:mixedsoilsedimentrock",
"LABEL_7":"mat:organicmaterial",
"LABEL_8":"mat:otheranthropogenicmaterial",
"LABEL_9":"mat:particulate",
"LABEL_10":"mat:rock",
"LABEL_11":"mat:rockorsediment",
"LABEL_12":"mat:sediment",
"LABEL_13":"mat:soil",
"LABEL_14":"xxx"
}

if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

#inputfilename = "SESARTraining-iSamMaterial.csv"
#inputfilename = "trainingdata-part2.csv"
inputfilename =  "SESARTrainingiSamKeywords.csv"


df = pd.read_csv(inputfilename, usecols=['igsn','traintext','iSampleMaterial'])
df = pd.DataFrame(df)
# load Dataset from Pandas DataFrame
dataset = datasets.Dataset.from_pandas(df)

for col in df.columns:
    print(col)

print("Dataset ready. ", inputfilename)
#dataset = datasets.load_dataset("csv", data_files=inputfilename)

result_output_dir = str('./output/savedmodels/model-iSampleMaterial-43-10-20-0.0001')
model = BertForSequenceClassification.from_pretrained(result_output_dir)
model = model.to(device)

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

result_df = pd.DataFrame(columns=["igsn","classificationtext","iSampleMaterial","class","score"])

idx = 0
print("time:",time.time())
for thisone in pipe(KeyDataset(dataset, "traintext")):
    thisone["label"] = classdict[thisone["label"]]
    igsn = df.at[idx,'igsn']
    isammat = df.at[idx,'iSampleMaterial']
    thistext = KeyDataset(dataset, "traintext")[idx]
    result_df.loc[len(result_df.index)] = [igsn,thistext, isammat, thisone["label"], thisone["score"]]
 #   print("Original text: ", KeyDataset(dataset, "trainingtext")[idx])
  #  print("class:", thisone["label"], "score:", thisone["score"])
    idx = idx + 1
    # write results each 10000 in case of crash...
    if (idx % 10000) == 500:
        print(idx, "  time:",time.time())
        result_df.to_csv('MaterialsClassification-' + inputfilename)

# write final result
result_df.to_csv('MaterialsClassification-'+ inputfilename)

print("yahoo! Done. Time:",time.time())