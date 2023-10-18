# Discussion of results

To generate the model run on the full unknow raw data, generate a dataset using the preprocess_3 routine to set up the samples from the big SESARTrainingiSamKeywords.csv file, and the following sample sizes for each material class: 

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

12050  samples. rand_state=43 for np sampling. This dataset is saved in trainingdata/SESARTraining-iSamMaterial.csv. This was input to train2, using the preprocess routine to generate the model saved as output/savedmodels/model-iSampleMaterial-43-10-20-0.0001.
- rand_state = 43 for np sampling in the train, eval/dev, and test subsets
- 10 epochs
- batches of 20
- learning rate 0.0001.

Model training takes 4-5 hours (see hardware information below)

Evaluation results

| label | precision | recall | f1 |
|:---|:---:|:---:|:---:|
| mat:anthropogenicmetal | 0.75 | 0.75 | 0.75 |
| mat:biogenicnonorganicmaterial | 0.96 | 1 | 0.98 |
| mat:gas | 0.83 | 0.95 | 0.88 |
| mat:liquidwater | 0.98 | 0.97 | 0.97 |
| mat:material | 0.95 | 0.97 | 0.96 |
| mat:mineral | 0.99 | 0.99 | 0.99 |
| mat:mixedsoilsedimentrock | 1 | 0.98 | 0.99 |
| mat:organicmaterial | 0.99 | 0.98 | 0.99 |
| mat:otheranthropogenicmaterial | 0.93 | 0.92 | 0.92 |
| mat:particulate | 1 | 0.98 | 0.99 |
| mat:rock | 0.96 | 0.98 | 0.97 |
| mat:rockorsediment | 0.99 | 0.99 | 0.99 |
| mat:sediment | 0.95 | 0.93 | 0.94 |
| mat:soil | 0.99 | 0.99 | 0.99 |
| xxx | 0.99 | 1 | 0.99 |


This model was run on the raw data trainingdata-part1.csv, trainingdata-part2.csv (see ESAR/rawData/ ...zip archive for each data file in repo), and on SESARTrainingiSamKeywords.csv (zip archive in SESAR\trainingDataApproach\trainingData). 

The part1 and part 2 csv files do not have annotated iSamples vocabulary terms. The text used for classification is in the 'trainingtext' column. Don't have 'gold' class assignments for samples, so evaluate based on the 'probabilities returned by the classifier. For part1, 1 percent of the result probabilies are less that 0.98. For part2, 1.1 percent of the result probabilies are less that 0.98.  Running the classificatoin pipeline on these datasets (with about 999,900 samples each) took 12-14 hours each, running on a laptop. Windows 10, python 3.11 running under pyCharm; System Model ROG Zephyrus G14, with AMD Ryzen 9 5900HS Processor 3300 Mhz, 8 Core(s), 16 Logical Processor(s), 40Gb RAM;  NVIDIA GeForce RTX 3050 GPU, with 4 Gb memory.

SESARTrainingiSamKeywords.csv is the fully annotated subset of SESAR samples, selected for records that have SESAR material type.   This dataset has 'gold' class assignments that can be compared with the classifier assignments. Results are in MaterialsClassification-SESARTrainingiSamKeywords.xlsx. For 1.9 percent of the classifications, the automated class assignment for iSamples MaterialType does not match the 'gold' assignment (from SMR). I observed that some of the discrepancies were actually better classification than the annotation, some are valid alternatives, and some are just plain wrong. 

This approach seems like a good start. Most of the work is in annotating the training data. Research needs to be done to forecast the size of training datasets required for different source data that needs to be classified. Possible approach is to start with a relatively small training dataset, running classification on a subset of data, then adding more training data for samples that are classified incorrectly. Build up the training dataset until accuracy of results is acceptable. Automatically assigned classification should be identified in the sample metadata records, with appropriate caveats about possible errors...