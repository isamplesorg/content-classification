# What's here

This repository contains files for categorizing physical samples using the iSamples controlled vocabulary using a machine-learning algorithm with annotated data used to train a model.

## Algorithm
The basic approach was developed by Sarah Hyunju Song, with data extracted from the SESAR database by Sara Ramdeen.  The ML algorthm is implemented in python using transformers (HuggingFace), torch (to use GPU on local laptop), sklearn, and some pandas and numpy functions. Input data is tabular text, with one column contining the text to classify, and one column with the target classification term. The classification terms are mapped to integers, and the input text is tokenized and the tokens mapped to integers. A tensor of these values (with added attention vectors) is the dataset input to the trainer. The trainer uses a training dataset and a dev datasets to refine the base language model used-- allenai/scibert_scivocab_uncased from huggingFace. After the trainer runs, the test dataset is run to compare the predicted classes with the annotated classes, and generated recall and precision numbers for the model.  The model is saved in the output/savedmodels folder.

The trainer has load_best_model_at_end=True set in its arguments, so the checkpoint with the best results on the evaluation (dev) dataset is the final model that is saved. Checkpoints are saved the end of each epoch in the output directory.   The trainer can be started from any of these checkpoints, but I haven't tried to get that working and there's no code for it in this repo.  I generally keep the best checkpoint after the run is complete. YOu can see which this is by looking at trainer_state.json in the final checkpoint folder. There's a key for 'best_model_checkpoint' near the top.

## Parameters
There are [lots of parameters to fiddle with](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.TrainingArguments) in the model, these are the ones I tried changing... 
- Initial configuration:
    - MAX_LENGTH = 64  input text used for classification are truncated to 64 tokens (not entirely clear if this is tokens or characters...). set in create_dataset function.
    - nb_epochs =  2 
    - batch_size = 10 
    - lr_rate = .01
    - sample_size = 10000
	 
- Started getting good results with preprocess_3 (see train2.py), 
    - MAX_LENGTH = 100  input text used for classification are truncated to 64 tokens (not entirely clear if this is tokens or characters...). set in create_dataset function.
    - nb_epochs =  10  (not clear if this could be lower, the best model seems to have been on epoch 3 or 4) 
    - batch_size = 20 (can't make larger, GPU on my laptop can't handle)_
    - lr_rate = .0001
    - sample_size = 12000 to 20000
	 
## Training data
The original training data was apparently selected for records that have values for the SESAR materialtype property  I'm not sure who did the original annotation of the data for iSamples material, but the results were reviewed by S.M. Richard. The original training data set was enhanced by SM Richard to add columns for iSample MaterialSampleType (specimen type), iSample sampledFeature, for the RockSediment and Mineral material type extension vocabulary, and the MaterialSampleType extension vocabulary.  Some iSampleMaterial annotations were also updated. 
This dataset contains 988426 records, all categorized for  the iSample vocabularies, and where applicable, extension vocab terms. Some samples fit multiple iSampleMaterial types, and might have multiple values delimited by a pipe ('|') character.  There are two extension material columns because some samples fit in more than one class; there is also a extProtolithMaterialType column to specify a protolith material for rock or sediment derived from pre-existing rock or sediment. For the actual training data, a single value was selected to represent each sample. The original work was done in an Excel workbook (SESARTrainingiSamKeywords.xlsx) The csv exported from Excel has been used for subsequent analysis and processing, and updates have been made that were not synchronized with the Excel workbook, so it has been superseded.


## root directory 
contains python code as follows:

- GetTrainingData.py  code to extract a random subset of records from the SESAR raw database dump files.
- train2.py  This is the training code.  Lots of inline comments. There are various options for selecting records to use for training from a large input annotation datasets. these are configured in the preprocess or preprocess_3 functions. Output files include the evaluation summary (xlsx), csv versions of the data used for training (..._df.csv), evaluation, and testing; final models are saved in subdirectories in output/savedmodels.
- classificationPipeline.py  This code imports a saved model from train2.py and processes an input tabular text file to generate inferred classes for input text. Result igsn, inputtext, inferred class and probability are saved to a tabular text file in the output directory.
	
## trainingData folder
- SESARTrainingiSamKeywords.zip zip archive containing the csv tabular data file with igsn, input text,  and annotated controlled vocabulary classes. Input to train2.py, with various options for selecting the subset of records to use for training. 
- SESARTraining-iSamMaterial.csv a training dataset that includes 12000 records extracted from SESARTrainingiSamKeywords.csv by preprocess_3, with the adition of 7998 records randomly selected and annotated from the raw source data. The goal is a training dataset more representative of all SESAR data.
- SESARNewTrainingData.xlsx  This is a new set of training data extracted from classification pipeline results for which the probability is < 0.98.  The idea is to annotate more records that cause problems to see if results can be improved.

## output folder
- checkpoint directories generated by trainer
- savedmodels folder:  saved models that can be run to classify input text
- files: 
    - ...Classification...xlsx  Ouput from the pipeline with classificatoin results. These are generated as csv, but thos are large, saving as Excel makes them small enough to get to Github
    - Result...xlsx  evaluation results from trainer runs. File names encode (cryptically, sorry!) key run parameters. result is table of precision and recall for each class. 
    - ..._df.csv  the train, dev, and test datasets from the most recent trainer run.
		

../SESAR/rawData
	zip archives containing the raw data dump from SESAR (April 2023), separated into 5 files of about 1,000,000 records each to make it more manageable.  Unzip one of these to run the classification pipeline.