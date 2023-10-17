This directory contains the ongoing experiments on using Zeroshot Text Classification to predict the material type iSamples vocabulary of the SESAR dataset.

## Datasets
Different datasets were created from the original SESAR dump and the annotated data `SESARTrainingiSamKeywords.csv` to find the best method to solve our problem.
The datasets that are used during this process will be uploaded [here](https://drive.google.com/drive/folders/1o9vZ4CzTDi0N93KKPCGgvcvqNTrPx4jI?usp=sharing).

- `SESAR_ZTC_train_multi.csv`, `SESAR_ZTC_dev_multi.csv`, `SESAR_ZTC_test_multi.csv` :  Used for multilabel finetuning (non zero-shot)
- `SESAR_ZTC_test_multiclass_label_fully_unseen.csv` : Used for multiclass label-fully-unseen tasks. 
- `SESAR_ZTC_test_multilabel_label_fully_unseen.csv` : Used for multilabel label-fully-unseen tasks. 
- `SESAR_ZTC_partial_label_unseen_train.csv`, `SESAR_ZTC_partial_label_unseen_dev.csv`,`SESAR_ZTC_partial_label_unseen_test.csv` : Used for multiclass partially-label-unseen tasks.
- `SESAR_labeled_original.csv` : Original dump of SESAR description_material labeled records.

## Code

-`hyperparam_search_ZTC.py` : Code to find the optimal hyperparameters for finetuning.

-`finetune_ZTC.py` : Implementation of fine-tuning a textual entailment model on the SESAR dataset. Converts the dataset into a format that is applicable for textual entailment finetuning task and uses the given arguments to execute finetuning. The finetuned model will be stored in the output directory.

-`evaluate.py` : Implementation of evaluating the model on SESAR dataset. The model that can be used could be either a finetuned model from finetune_ZTC.py or an out-of-box textual entailment model(completely zeroshot). Result of evaluation will be logged. Supports solving the task as a multilabel or multiclass. For multiclass approach, also contains implementation of using specified depth level of the entire hierarchical label space of iSamples vocabulary.

## Results of Experiments
Results of the ongoing experiments will be updated [here](https://docs.google.com/spreadsheets/d/19Q95HsjRS7JGyHoY8o8hxirBO6NiJ1ufHYB_xg0X4Ks/edit?usp=sharing).
Approaches experimented so far: 
1) Partially-label-unseen : Multiclass. Finetune the model with a partial label space (partial training data) and see how it evaluates on the entire label space. This approaches uses the iSamplesMaterialType that was contained in SESARTrainingiSamKeywords.csv directly as labels. 
2) Fully-label-unseen : Multiclass. Use the pretrained model directly and apply it on the test dataset. This approaches uses the iSamplesMaterialType that was contained in SESARTrainingiSamKeywords.csv directly as labels. 
3) Multilabel-fully-label-unseen : Multilabel. Use the pretrained model directly and apply it on the test dataset. This approach uses the label including the extension vocabulary and expects the model to predict the label and all of the parent labels of it.
4) Depth-fully-label-unseen: Multiclass. Use the iSamplesMaterialType field and the extMaterialType	field in SESARTrainingiSamKeywords.csv to convert the label into specified depth level of the iSamples MaterialType hierarchy. Use this converted label as expected prediction space. 
5) Multilabel-Finetune : Multilabel. Use the pretrained model and finetune on the entire SESAR dataset. (code TBC)
