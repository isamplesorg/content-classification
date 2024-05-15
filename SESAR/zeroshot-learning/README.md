This directory contains the ongoing experiments on using Zeroshot Text Classification to predict the material type iSamples vocabulary of the SESAR dataset.

## Datasets
Different datasets were created from the original SESAR dump and the annotated data `SESARTrainingiSamKeywords.csv` to find the best method to solve our problem.
The datasets that are used during this process will be uploaded [here](https://drive.google.com/drive/folders/1o9vZ4CzTDi0N93KKPCGgvcvqNTrPx4jI?usp=sharing).

- `SESAR_ZTC_train_multi_nonleaves_replaced.csv`, `SESAR_ZTC_dev_multi_nonleaves_replaced.csv`, `SESAR_ZTC_test_multi_nonleaves_replaced.csv`  : Used for multilabel finetuning on entire label space(with underperforming labels that are substituted by definitions)
- `SESAR_ZTC_train_multi_entire.csv`, `SESAR_ZTC_dev_multi_entire.csv`, `SESAR_ZTC_test_multi_entire.csv`:  Used for multilabel finetuning (non zero-shot) on the entire label space
- `SESAR_ZTC_train_multi_wo_leaf.csv`, `SESAR_ZTC_dev_multi_wo_leaf.csv`, `SESAR_ZTC_test_multi_wo_leaf.csv`: Used for multilabel finetuning on the label space excluding leaf labels
- `SESAR_ZTC_test_multiclass_label_fully_unseen.csv` : Used for multiclass label-fully-unseen tasks. 
- `SESAR_ZTC_test_multilabel_label_fully_unseen.csv` : Used for multilabel label-fully-unseen tasks. 
- `SESAR_ZTC_partial_label_unseen_train.csv`, `SESAR_ZTC_partial_label_unseen_dev.csv`,`SESAR_ZTC_partial_label_unseen_test.csv` : Used for multiclass partially-label-unseen tasks.
- `SESAR_labeled_original.csv` : Original dump of SESAR description_material labeled records.

## Code

-`hyperparam_search_ZTC.py` : Code to find the optimal hyperparameters for finetuning.

-`finetune_ZTC.py` : Implementation of fine-tuning a textual entailment model on the SESAR dataset. Converts the dataset into a format that is applicable for textual entailment finetuning task and uses the given arguments to execute finetuning. The finetuned model will be stored in the output directory.
    ``python finetune_ZTC.py --hypothesis_template_type B --lr_rate 5e-05 --train_data_dir ./datasets/SESAR_ZTC_train_multi_wo_leaf.csv --dev_data_dir ./datasets/SESAR_ZTC_dev_multi_entire.csv --model_name roberta-large-mnli --num_epochs 3 --train_batch_size 4 --eval_batch_size 4`` 

-`evaluate.py` : Implementation of evaluating the model on SESAR dataset. The model that can be used could be either a finetuned model from finetune_ZTC.py. Result of evaluation will be logged.
    ``python evaluate.py --label_file total_unique_nonleaf_multi_labels.txt --test_dataset_dir ./datasets/SESAR_ZTC_test_multi_nonleaves.csv --output_dir ./C_roberta-large-mnli_3_5e-05_0.01_8_2023_11_17_17_30_35/checkpoint-1400``

## Results of Experiments
Results of the ongoing experiments will be updated [here](https://docs.google.com/spreadsheets/d/19Q95HsjRS7JGyHoY8o8hxirBO6NiJ1ufHYB_xg0X4Ks/edit?usp=sharing).
Approaches experimented so far: 
1) Partially-label-unseen : Multiclass. Finetune the model with a partial label space (partial training data) and see how it evaluates on the entire label space. This approaches uses the iSamplesMaterialType that was contained in SESARTrainingiSamKeywords.csv directly as labels. 
2) Fully-label-unseen : Multiclass. Use the pretrained model directly and apply it on the test dataset. This approaches uses the iSamplesMaterialType that was contained in SESARTrainingiSamKeywords.csv directly as labels. 
3) Multilabel-fully-label-unseen : Multilabel. Use the pretrained model directly and apply it on the test dataset. This approach uses the label including the extension vocabulary and expects the model to predict the label and all of the parent labels of it.
4) Depth-fully-label-unseen: Multiclass. Use the iSamplesMaterialType field and the extMaterialType	field in SESARTrainingiSamKeywords.csv to convert the label into specified depth level of the iSamples MaterialType hierarchy. Use this converted label as expected prediction space. 
5) Multilabel-Finetune-Leaves/Nonleaves: Multilabel. Use the pretrained model and finetune it on the SESAR datasets with each record's label up to the non-leaf labels (nonleaves) OR each record's label up to the leaf labels (leaves). Evaluation is done on test dataset records that have only non-leaf labels(`SESAR_ZTC_test_multi_nonleaves.csv`) and test dataset records that have leaf labels and using only those leaf labels as output space (`SESAR_ZTC_test_multi_leaves_only.csv`). After underperforming non-leaf labels are identified, experiment with replacing those labels with layman term definitions (using this definition[here](https://docs.google.com/spreadsheets/d/1JD_F37bLxuqeuGVIPCuuWvKxzA6tGmMzYrogX_mJC_M/edit?usp=sharing)). 
