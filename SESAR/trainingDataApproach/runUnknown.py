import torch
import pandas as pd
from transformers import BertTokenizer,  BertForSequenceClassification


# Load model checkpoint

datadir = str('C:/Users/smrTu/OneDrive/Documents/GithubC/iSamples/content-classification/SESAR/')


classesMat = ['mat:anthropogenicmetal','mat:biogenicnonorganicmaterial','mat:gas','mat:liquidwater',\
           'mat:material','mat:mineral','mat:mixedsoilsedimentrock','mat:organicmaterial',\
           'mat:otheranthropogenicmaterial','mat:rock','mat:rockorsediment','mat:sediment','mat:soil','xxx'
]

classes = [   'spec:othersolidobject',  'spec:genericaggregation',   'spec:wholeorganism',\
        'spec:slurrybiomeaggregation',  'spec:bundlebiomeaggregation',  'spec:organismpart',\
        'spec:fluidincontainer', 'spec:organismproduct', 'spec:fossil',  'spec:analyticalpreparation',\
        'spec:biologicalspecimen',  'spec:physicalspecimen', 'xxx', 'spec:artifact',\
        'spec:experimentalproduct',  'spec:researchproduct', 'spec:anthropogenicaggregation' ]


# model = BertForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels = len(classes), \
#               problem_type = "single_label_classification")
tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True, use_fast=True)

model = BertForSequenceClassification.from_pretrained('./output/savedmodels/model-iSamMaterialSampleType-43-10-20-0.0001')

# model.load_state_dict(torch.load(datadir + 'output/checkpoint-6170'))
model.eval()

# Load CSV data to classify
df = pd.read_csv('C:/Users/smrTu/OneDrive/Documents/Workspace/iSamples/training/trainingdata-part1.csv')

text_col = 'trainingtext'
id_col = 'igsn'

# Run model inference on text column
inputs = df[text_col].tolist()
predictions = model(inputs)

# Get predicted class names
# classes = ['mat:anthropogenicmetal','mat:biogenicnonorganicmaterial','mat:gas','mat:liquidwater',\
#            'mat:material','mat:mineral','mat:mixedsoilsedimentrock','mat:organicmaterial',\
#            'mat:otheranthropogenicmaterial','mat:rock','mat:rockorsediment','mat:sediment','mat:soil','xxx'
# ]
preds = [classes[p] for p in predictions.argmax(dim=1)]

# Create output DataFrame
output = pd.DataFrame({id_col: df[id_col],
                       text_col: df[text_col],
                       'pred_class': preds})

# Save output csv
output.to_csv('predictions.csv', index=False)