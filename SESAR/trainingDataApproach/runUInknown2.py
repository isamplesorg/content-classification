import os
import pickle
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing

#global variables
classcol = "iSampleMaterial"  # classcol is the target class that should be inferred fromtext in traintextcol
#classcol = "extMaterialType"
#classcol = "iSamMaterialSampleType"
rand_state = int(43) # was 19  mak 78; np.split uses this for reproducible subsetting of a dataframe
samplesize = int(0)
traintextcol = "traintext"  # name of the field in the data that is used to classify
#traintextcol = "concatenated_text"
MAX_LENGTH = 100

#*************************************
#  functions
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


def create_dataset(dataframe, tokenizer):
    MAX_LENGTH = 100  # raise from 60 to 80 to 100
    inputs = {
        "input_ids": [],
        "attention_mask": []
    }

    sents = []
    for _, row in dataframe.iterrows():
        row_value = row[traintextcol]
        sents.append(str(row_value))

    for sent in sents:
        tokenized_input = tokenizer(sent, max_length=MAX_LENGTH, padding='max_length', truncation=True)
        inputs["input_ids"].append(torch.tensor(tokenized_input["input_ids"]))
        inputs["attention_mask"].append(torch.tensor(tokenized_input["attention_mask"]))

    print("torch tensor dataframe columns:", dataframe.columns.values)
    labels = torch.tensor(dataframe[classcol].values.tolist())

    return MulticlassDataset(inputs, labels)


# ***************************************************
#  start processing ********************************************************

if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Set PYTORCH_CUDA_ALLOC_CONF environment variable
# GPU is running out of memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "heuristic"

result_output_dir = str('./output/savedmodels/model-iSampleMaterial-43-10-20-0.0001')

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

# Load model
model = BertForSequenceClassification.from_pretrained(result_output_dir)
model = model.to(device)

#set up the input data
inputfilename = "SESARTraining-iSamMaterial.csv"
df = pd.read_csv(inputfilename, usecols=['igsn', classcol, traintextcol],dtype={'igsn':str, classcol:str, traintextcol:str})

# classify_df = preprocess(df)
# classify_df = df.copy()
# le = preprocessing.LabelEncoder()
# le.fit(classify_df[classcol])
# print(" number of labels: ", len(le.classes_))
# print("label encoder classes:", le.classes_)
# classify_df[classcol] = le.transform(classify_df[classcol])

sents = []
for _, row in df.iterrows():
    row_value = row[traintextcol]
    sents.append(str(row_value))

tokenized_input = tokenizer(sents, max_length=MAX_LENGTH, padding='max_length', truncation=True, return_tensors='pt', batch_size=32)
# Ensure the encoded input tensors are CUDA tensors, if GPU available:
if torch.cuda.is_available():
    tokenized_input = {k: v.to("cuda") for k, v in tokenized_input.items()}
print("have tokenized input")
# Tokenize texts (handled by create_dataset)
# encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

# Make predictions
result_df = pd.DataFrame()

#output = model(**encoded_input)
for batch in tokenized_input:
    output = model(**batch)
# this line is equivalent to:
# output = model(input_ids=encoded_input['input_ids'],
#                attention_mask=encoded_input['attention_mask'],
#                ...)

    Print("calculate logits")
#  The output will also be on GPU, so move it back to CPU for processing:
    logits = output.logits.to("cpu")
    predictions = output.logits.argmax(-1)

    Print("Get prediction probabilities")
    proba = torch.nn.functional.softmax(output.logits, dim=-1)

# Output results -- original from Claude
# results = []
# for i, text in enumerate(texts):
#     probas = dict(zip(model.config.id2label, proba[i].tolist()))
#     results.append({
#         "text": text,
#         "predicted_class": model.config.id2label[predictions[i].item()],
#         "probabilities": probas
#     })

    input_id_seq = batch['input_ids'][0]
    sents = tokenizer.decode(input_id_seq)

    for i, text in enumerate(sents):
        probas = dict(zip(model.config.id2label, proba[i].tolist()))
        result_df.append({
            "text": text,
            "predicted_class": model.config.id2label[predictions[i].item()],
            "probabilities": probas
        })


print("SAVE predictions")
filenamestr = "output/predict-" + inputfilename
result_df.to_csv(filenamestr)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
print("done")