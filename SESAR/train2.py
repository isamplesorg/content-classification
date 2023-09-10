import os
import time

import numpy as np
# import argparse
import pandas as pd
import torch
import torch.nn as nn
import openpyxl
from sklearn import preprocessing
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# import pickle
# code by Sarah Song and Stephen Richard
# 2023-09

os.environ["WANDB_MODE"] = "disabled"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
#!export CUDA_VISIBLE_DEVICES=""

#global variables
#classcol = "iSampleMaterial"  # classcol is the target class that should be inferred fromtext in traintextcol
#classcol = "extMaterialType"
classcol = "iSamMaterialSampleType"
rand_state = int(43) # was 19  mak 78; np.split uses this for reproducible subsetting of a dataframe
samplesize = int(0)
classname = ""
traintextcol = "traintext"  # name of the field in the data that is used to classify
#traintextcol = "concatenated_text"

#training parameters:
nb_epochs = int(10)  #was 10,
batch_size = int(20) # 30.
lr_rate = float(0.0001) #was.01

le = preprocessing.LabelEncoder()
# use this to map categories to integers.

  #load tokenizer
tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True, use_fast=True)

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


def create_dataset(dataframe, tokenizer, thecol):
    MAX_LENGTH = 100  # raise from 60 to 80 to 100
    inputs = {
        "input_ids": [],
        "attention_mask": []
    }

    # use this if concatenting text from all fields in the data (except the class field)
    # features_columns = [x for x in dataframe.columns.values if x != thecol]

    # def create_concatenated_text(dataframe):
    #     """combine the columns text to create a single sentence"""
    #     ttext = []  # text that is a concatenation of all columns
    #     for _, row in dataframe.iterrows():
    #         row_value = row["text"]
    #         if row_value != "" and type(row_value) == str:
    #             combined += row_value  # +" , "
    #         ttext.append(combined)
    #     return ttext

    # sents = create_concatenated_text(dataframe)

    def getTrainText(dataframe, ttcol):  # smr version-- pre concatenate training text in one column
        ttext = []
        for _, row in dataframe.iterrows():
            row_value = row[ttcol]
            ttext.append(str(row_value))
        return ttext

    sents = getTrainText(dataframe, traintextcol)

    for sent in sents:
        tokenized_input = tokenizer(sent, max_length=MAX_LENGTH, padding='max_length', truncation=True)
        inputs["input_ids"].append(torch.tensor(tokenized_input["input_ids"]))
        inputs["attention_mask"].append(torch.tensor(tokenized_input["attention_mask"]))

    print("torch tensor dataframe columns:", dataframe.columns.values)
    # print("dataframe['iSampleMaterial']: ",dataframe['iSampleMaterial'].values )
    labels = torch.tensor(dataframe[classcol].values.tolist())

    return MulticlassDataset(inputs, labels)


def get_class_weights(dataframe):
    """computes the class weight and returns a list to account for class imbalance """

    # dataframe['iSampleMaterial'] = le.transform(dataframe.iSampleMaterial)
    labels = torch.tensor(dataframe[classcol].values.tolist())
    # label_le = le.classes_
    print("np unique labels for weights:", np.unique(labels))
    # print ("le class labels: ",label_le)
    print("labels.numpy:", labels.numpy())

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels.numpy())
    # class_weights=compute_class_weight( class_weight ='balanced',classes = labels,y = labellist.numpy())

    print(class_weights)

    total_class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
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


def preprocess(dataframe, selected_material_type=None):
    # convert the dataframe labels accordingly by the material type
    # original preprocess from Sarah Song

    new_df = dataframe.copy()

    # convert labels into integers
    le.fit(new_df.iSampleMaterial)
    print(" number of labels: ", len(le.classes_))
    new_df[classcol] = le.transform(new_df.iSampleMaterial)

    # split data to training df, dev df, test df
    sample_size = 10000
    # fraction=sample_size/len(new_df)  # get about 500 samples
    # sel_len = sample_size
    train_df, dev_df, test_df = np.split(new_df.sample(n=sample_size, random_state=42),
                                         [int(.6 * sample_size), int(.8 * sample_size)])

    train_df.to_csv('output/train_df.csv')
    dev_df.to_csv('output/dev_df.csv')
    test_df.to_csv('output/test_df.csv')

    return train_df, dev_df, test_df


# use dictionary of classes
# SMR 2023-08-21

def preprocess_3(dataframe):
    # classname : samplesize
    classdictiSamMaterial = {
        "mat:rock": 2000,
        "mat:mineral": 2000,
        "mat:organicmaterial": 2000,
        "mat:sediment": 1000,
        "mat:soil": 1000,
        "mat:liquidwater": 1000,
        "mat:material": 500,
        "mat:rockorsediment": 500,
        "mat:mixedsoilsedimentrock": 500,
        "mat:biogenicnonorganicmaterial": 500,
        "mat:otheranthropogenicmaterial": 300,
        "mat:particulate": 300,
        "xxx": 300,
        "mat:gas": 100,
        "mat:anthropogenicmetal": 50
    }

    classdictextmat = {
        "ISI": 43000,
        "ming:silicategermanatemineral": 8000,
        "ming:oxidemineral": 4000,
        "rksd:Basalt": 4000,
        "rksd:Metamorphic_Rock": 3000,
        "ming:phosphatearsenatevanadatemineral": 3000,
        "ming:sulfidesulfosaltmineral": 3000,
        "rksd:Metasomatic_Rock": 3000,
        "ming:carbonatenitratemineral": 2000,
        "rksd:Generic_Sandstone": 2000,
        "rksd:Granitoid": 2000,
        "rksd:Fine_Grained_Igneous_Rock": 2000,
        "ming:sulfateselenatetelluratemineral": 1000,
        "rksd:Pyroclastic_Rock": 1000,
        "rksd:Generic_Mudstone": 1000,
        "ming:nativeelementmineral": 1000,
        "rksd:Sedimentary_Rock": 900,
        "rksd:Carbonate_Sedimentary_Rock": 900,
        "rksd:Peridotite": 800,
        "ming:halidemineral": 700,
        "rksd:Andesite": 700,
        "rksd:Gabbroid": 600,
        "rksd:Tephra": 500,
        "rksd:Doleritic_Rock": 500,
        "rksd:Glass_Rich_Igneous_Rock": 400,
        "rksd:Chemical_Sedimentary_Material": 400,
        "rksd:Dioritoid": 400,
        "rksd:Rhyolitoid": 400,
        "rksd:Ultramafic_Igneous_Rock": 300,
        "rksd:Igneous_Rock": 300,
        "rksd:Pyroxenite": 300,
        "rksd:Granodiorite": 300,
        "ming:boratemineral": 300,
        "rksd:Exotic_Composition_Igneous_Rock": 300,
        "rksd:Syenitoid": 300,
        "rksd:Trachytoid": 300,
        "rksd:Porphyry": 300,
        "rksd:Breccia": 200,
        "rksd:Dacite": 200,
        "rksd:Generic_Conglomerate": 200,
        "rksd:Carbonate_Sediment": 200,
        "rksd:Clastic_Sedimentary_Rock": 200,
        "rksd:Non_Clastic_Siliceous_Sedimentary_Rock": 200,
        "rksd:Mud_Size_Sediment": 200,
        "rksd:Clastic_Sediment": 200,
        "rksd:Impact_Generated_Material": 200,
        "rksd:Diamictite": 200,
        "rksd:Iron_Rich_Sedimentary_Rock": 200,
        "rksd:Phonolitoid": 200,
        "rksd:Anorthositic_Rock": 200,
        "rksd:Coal": 200,
        "rksd:Tephritoid": 200,
        "rksd:Fragmental_Igneous_Rock": 200,
        "rksd:Tonalite": 200,
        "rksd:residual_material": 200,
        "rksd:Sand_Size_Sediment": 200,
        "rksd:Tuffite": 200,
        "rksd:Biogenic_Sediment": 200,
        "rksd:Mylonitic_Rock": 200,
        "ming:organicmineral": 200,
        "rksd:Foid_Syenitoid": 200,
        "rksd:Phaneritic_Igneous_Rock": 200,
        "rksd:High_Magnesium_Fine_Grained_Igneous_Rock": 200,
        "rksd:Basic_Igneous_Rock": 200,
        "rksd:Foiditoid": 200,
        "insuf": 200,
        "rksd:Diamicton": 200,
        "rksd:Gravel_Size_Sediment": 200,
        "rksd:Massive_Sulphide": 150,
        "rksd:Hornblendite": 150,
        "rksd:Organic_Rich_Sedimentary_Rock": 150,
        "rksd:Acidic_Igneous_Rock": 150,
        "rksd:Foidolite": 150,
        "ocmat:glass": 100,
        "rksd:Foid_Gabbroid": 100,
        "rksd:Fault_Related_Material": 100
    }

    classdict = {
        "spec:othersolidobject": 63000,
        "spec:genericaggregation": 12000,
        "spec:wholeorganism": 5000,
        "spec:slurrybiomeaggregation": 5000,
        "spec:bundlebiomeaggregation": 4000,
        "spec:organismpart": 3000,
        "spec:fluidincontainer": 3000,
        "spec:organismproduct": 2500,
        "spec:fossil": 1500,
        "spec:analyticalpreparation": 1400,
        "spec:biologicalspecimen": 700,
        "spec:physicalspecimen": 700,
        "xxx": 500,
        "spec:artifact": 250,
        "spec:experimentalproduct": 150,
        "spec:researchproduct": 60,
        "spec:anthropogenicaggregation": 44
    }


    samplesize = int(0)
    classname = ""

    # empty data frames to accumulate results
    work_df = pd.DataFrame()
    train_df = pd.DataFrame()
    dev_df = pd.DataFrame()
    test_df = pd.DataFrame()

    le.fit(dataframe[classcol])
    print(" number of labels: ", len(le.classes_))
    print("label encoder classes:", le.classes_)
    print("transform classes:", le.fit_transform(le.classes_))

    for classname, samplesize in classdict.items():
        # split data to training df, val df, test df
        # print("class:", classname, "  samplesize:",samplesize)
        work_df = dataframe[dataframe[classcol] == classname].copy()  # flattened label version
        # build the data frames
        # print(classname, " rowcount: ", len(work_df.index))
        train_df_work, dev_df_work, test_df_work = np.split(work_df.sample(n=samplesize, random_state=rand_state),
                                                            [int(.6 * samplesize), int(.8 * samplesize)])
        print("finished ", classname, " dataframe. samplesize:", samplesize, " split at:", int(.6 * samplesize),
              int(.8 * samplesize))

        # merge into the output dataframes
        train_df = pd.concat([train_df_work, train_df])
        test_df = pd.concat([test_df_work, test_df])
        dev_df = pd.concat([dev_df_work, dev_df])

        # sort by igsn, convert labels to integers
    # train_df.sort_values(by='igsn', inplace=True )
    train_df[classcol] = le.transform(train_df[classcol])
    # test_df.sort_values(by='igsn', inplace=True )
    test_df[classcol] = le.transform(test_df[classcol])
    # dev_df.sort_values(by='igsn', inplace=True )
    dev_df[classcol] = le.transform(dev_df[classcol])

    # write dataframes for reference
    train_df.to_csv('output/train_df.csv')
    dev_df.to_csv('output/dev_df.csv')
    test_df.to_csv('output/test_df.csv')

    return train_df, dev_df, test_df

#df = pd.read_csv("iSamplesMaterialTrainingSmall.csv")
#df = pd.read_csv("SESARTrainingiSamKeywords.csv", usecols=['igsn', 'traintext'],dtype={'igsn':str,'traintext':str})
#df = pd.read_csv("MaterialTypeData2023-08-07.csv") # only has rock, sediment, rocksed, soil, mineral

df = pd.read_csv("SESARTrainingiSamKeywords.csv", usecols=['igsn', classcol, traintextcol],dtype={'igsn':str, classcol:str, traintextcol:str})

df = df.fillna("")
#remove rows that do not have a class name or training text
df = df[df[classcol]!=""]
df = df[df[traintextcol]!=""]

#load tokenizer
# train(material_type, df, tokenizer, batch_size,lr_rate, nb_epochs, train_mode, output_dir)

# insert train function in line here for debugging...
#train_df, dev_df, test_df = preprocess(df)  #original function from Sarah Song
#train_df, dev_df, test_df = preprocess_2(df)  #steves update, only rock, mineral, rocksed, soil, sediment
train_df, dev_df, test_df = preprocess_3(df)  #dictionary to set sample size for each class


#print("train_df columns:", train_df.columns.values)
#print("train_df:", train_df.describe)
#train_df['iSampleMaterial'].values

train_dataset = create_dataset(train_df, tokenizer, classcol)
dev_dataset = create_dataset(dev_df,tokenizer, classcol)
test_dataset = create_dataset(test_df,tokenizer,classcol)

print ("le classes: ", len(le.classes_))
# load model
model = BertForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels = len(le.classes_), problem_type = "single_label_classification")


#train_mode = str('FALSE')
train_mode = str('custom')
 #  Whether we account for class imbalance during training by using a custom trainer
    # (custom) or not (none)
output_dir =str('output')
 #Output directory where the model checkpoint will be saved

desc = model.to(device)
training_args = TrainingArguments(
          no_cuda = False,
          output_dir= output_dir,     # output directory
          num_train_epochs=nb_epochs,              # total number of training epochs
          per_device_train_batch_size=batch_size,  # batch size per device during training
          per_device_eval_batch_size=batch_size,   # batch size for evaluation
          learning_rate = lr_rate,
          warmup_steps=500,                # number of warmup steps for learning rate scheduler
          weight_decay=0.01,
          load_best_model_at_end=True,
          logging_dir=output_dir,            # directory for storing logs
          logging_steps=10,
          evaluation_strategy = "epoch", #To calculate metrics per epoch
          save_strategy = "epoch"
  )

class_weights = get_class_weights(train_df)
CustomTrainer = create_custom_trainer(class_weights)
if train_mode == "custom":
    trainer = CustomTrainer(model = model, args =training_args, train_dataset=train_dataset, eval_dataset=dev_dataset)
else:
    trainer = Trainer(model = model, args =training_args, train_dataset=train_dataset, eval_dataset=dev_dataset)


st = time.time()
trainer.train()
et = time.time()
elapsed_time = et - st
print ('trainer execution time:', elapsed_time, ' seconds')
# conduct evaluation
keys = []
precision = []
recall = []
f1 = []

logits = trainer.predict(test_dataset)[0] #get the logits

test_pred = np.argmax(logits,axis=-1)
y_test= torch.tensor(test_df[classcol].values.tolist())

print (y_test)

res = classification_report(y_test,test_pred,output_dict=True)

print(logits[0])
logits.shape

print ("N epochs =", nb_epochs, " Batch: ", batch_size, " learn rate: ", lr_rate)
print ("random seed: ", rand_state)

for key, score in res.items():
  if key.isdigit():
    keys.append((le.inverse_transform([int(key)])[0]))
    precision.append(round(score['precision'],2))
    recall.append(round(score['recall'],2))
    f1.append(round(score['f1-score'],2))
    print("%s \t\t\t %0.2f \t %0.2f \t %0.2f"% (le.inverse_transform([int(key)])[0],score['precision'], score['recall'], score['f1-score']))

#write the results to excel and save

filenamestr = "-" + classcol + "-" + str(rand_state) + "-" + str(nb_epochs) + "-" + str(batch_size) + "-" + str(lr_rate)

result_df = pd.DataFrame(data=zip(keys,precision,recall,f1), columns=['label','precision','recall','f1'])
result_output_dir ="output/sesar_result" + filenamestr + ".xlsx"
result_df.to_excel(result_output_dir)
print("Macro average: ",f1_score(y_test,test_pred,average='macro'))

print("all done")