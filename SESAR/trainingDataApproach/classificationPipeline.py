#from sklearn import preprocessing
# from sklearn import preprocessing
import time

import datasets
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

classdict_mat = {"LABEL_0":"mat:anthropogenicmetal",
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

classdict_matSamType = {
"LABEL_0":"spec:analyticalpreparation",
"LABEL_1":"spec:anthropogenicaggregation",
"LABEL_2":"spec:artifact",
"LABEL_3":"spec:biologicalspecimen",
"LABEL_4":"spec:bundlebiomeaggregation",
"LABEL_5":"spec:experimentalproduct",
"LABEL_6":"spec:fluidincontainer",
"LABEL_7":"spec:fossil",
"LABEL_8":"spec:genericaggregation",
"LABEL_9":"spec:organismpart",
"LABEL_10":"spec:organismproduct",
"LABEL_11":"spec:othersolidobject",
"LABEL_12":"spec:physicalspecimen",
"LABEL_13":"spec:researchproduct",
"LABEL_14":"spec:slurrybiomeaggregation",
"LABEL_15":"spec:wholeorganism",
"LABEL_16":"xxx"
}

classdict = {
"LABEL_0":"ISI",
"LABEL_1":"ming:boratemineral",
"LABEL_2":"ming:carbonatenitratemineral",
"LABEL_3":"ming:halidemineral",
"LABEL_4":"ming:nativeelementmineral",
"LABEL_5":"ming:organicmineral",
"LABEL_6":"ming:oxidemineral",
"LABEL_7":"ming:phosphatearsenatevanadatemineral",
"LABEL_8":"ming:silicategermanatemineral",
"LABEL_9":"ming:sulfateselenatetelluratemineral",
"LABEL_10":"ming:sulfidesulfosaltmineral",
"LABEL_11":"ocmat:glass",
"LABEL_12":"rksd:Acidic_Igneous_Rock",
"LABEL_13":"rksd:Andesite",
"LABEL_14":"rksd:Anorthositic_Rock",
"LABEL_15":"rksd:Basalt",
"LABEL_16":"rksd:Basic_Igneous_Rock",
"LABEL_17":"rksd:Biogenic_Sediment",
"LABEL_18":"rksd:Breccia",
"LABEL_19":"rksd:Carbonate_Sediment",
"LABEL_20":"rksd:Carbonate_Sedimentary_Rock",
"LABEL_21":"rksd:Chemical_Sedimentary_Material",
"LABEL_22":"rksd:Clastic_Sediment",
"LABEL_23":"rksd:Clastic_Sedimentary_Rock",
"LABEL_24":"rksd:Coal",
"LABEL_25":"rksd:Dacite",
"LABEL_26":"rksd:Diamictite",
"LABEL_27":"rksd:Diamicton",
"LABEL_28":"rksd:Dioritoid",
"LABEL_29":"rksd:Doleritic_Rock",
"LABEL_30":"rksd:Exotic_Composition_Igneous_Rock",
"LABEL_31":"rksd:Fault_Related_Material",
"LABEL_32":"rksd:Fine_Grained_Igneous_Rock",
"LABEL_33":"rksd:Foid_Gabbroid",
"LABEL_34":"rksd:Foid_Syenitoid",
"LABEL_35":"rksd:Foiditoid",
"LABEL_36":"rksd:Foidolite",
"LABEL_37":"rksd:Fragmental_Igneous_Rock",
"LABEL_38":"rksd:Gabbroid",
"LABEL_39":"rksd:Generic_Conglomerate",
"LABEL_40":"rksd:Generic_Mudstone",
"LABEL_41":"rksd:Generic_Sandstone",
"LABEL_42":"rksd:Glass_Rich_Igneous_Rock",
"LABEL_43":"rksd:Granitoid",
"LABEL_44":"rksd:Granodiorite",
"LABEL_45":"rksd:Gravel_Size_Sediment",
"LABEL_46":"rksd:High_Magnesium_Fine_Grained_Igneous_Rock",
"LABEL_47":"rksd:Hornblendite",
"LABEL_48":"rksd:Igneous_Rock",
"LABEL_49":"rksd:Impact_Generated_Material",
"LABEL_50":"rksd:Iron_Rich_Sedimentary_Rock",
"LABEL_51":"rksd:Massive_Sulphide",
"LABEL_52":"rksd:Metamorphic_Rock",
"LABEL_53":"rksd:Metasomatic_Rock",
"LABEL_54":"rksd:Mud_Size_Sediment",
"LABEL_55":"rksd:Mylonitic_Rock",
"LABEL_56":"rksd:Non_Clastic_Siliceous_Sedimentary_Rock",
"LABEL_57":"rksd:Organic_Rich_Sedimentary_Rock",
"LABEL_58":"rksd:Peridotite",
"LABEL_59":"rksd:Phaneritic_Igneous_Rock",
"LABEL_60":"rksd:Phonolitoid",
"LABEL_61":"rksd:Porphyry",
"LABEL_62":"rksd:Pyroclastic_Rock",
"LABEL_63":"rksd:Pyroxenite",
"LABEL_64":"rksd:Rhyolitoid",
"LABEL_65":"rksd:Sand_Size_Sediment",
"LABEL_66":"rksd:Sedimentary_Rock",
"LABEL_67":"rksd:Syenitoid",
"LABEL_68":"rksd:Tephra",
"LABEL_69":"rksd:Tephritoid",
"LABEL_70":"rksd:Tonalite",
"LABEL_71":"rksd:Trachytoid",
"LABEL_72":"rksd:Tuffite",
"LABEL_73":"rksd:Ultramafic_Igneous_Rock",
"LABEL_74":"rksd:residual_material"
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
#classcol = "iSampleMaterial"
#classcol = "iSamMaterialSampleType"
classcol = "extMaterialType"

#df = pd.read_csv("trainingData/"+inputfilename, usecols=['igsn','traintext',classcol])
df = pd.read_csv("trainingData/"+inputfilename, usecols=['igsn','traintext',classcol])
df = pd.DataFrame(df)
# load Dataset from Pandas DataFrame
dataset = datasets.Dataset.from_pandas(df)

for col in df.columns:
    print(col)

print("Dataset ready. ", inputfilename)
#dataset = datasets.load_dataset("csv", data_files=inputfilename)

#result_output_dir = str('./output/savedmodels/model-iSampleMaterial-43-10-20-0.0001')
#result_output_dir = str('./output/savedmodels/model-iSamMaterialSampleType-43-10-20-0.0001')
result_output_dir = str('./output/savedmodels/model-extMaterialType-43-10-20-0.0001')
model = BertForSequenceClassification.from_pretrained(result_output_dir)
model = model.to(device)

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

result_df = pd.DataFrame(columns=["igsn","classificationtext",classcol,"class","score"])

idx = 0
start = time.time()
print("time:",time.time())
for thisone in pipe(KeyDataset(dataset, "traintext")):
    thisone["label"] = classdict[thisone["label"]]
    igsn = df.at[idx,'igsn']
    isammat = df.at[idx,classcol]
    thistext = KeyDataset(dataset, "traintext")[idx]
    result_df.loc[len(result_df.index)] = [igsn,thistext, isammat, thisone["label"], thisone["score"]]
 #   print("Original text: ", KeyDataset(dataset, "trainingtext")[idx])
  #  print("class:", thisone["label"], "score:", thisone["score"])
    idx = idx + 1
    # write results each 10000 in case of crash...
    if (idx % 10000) == 500:
        print(idx, "  time:",time.time())
        result_df.to_csv('output/' + classcol + 'Classification-' + inputfilename)

# write final result
result_df.to_csv('output/' + classcol + 'Classification-' + inputfilename)

print("yahoo! Done. Time(sec): ", time.time() - start)
