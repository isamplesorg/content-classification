# content-classification

Training, evaluation code of classifiers that were created for the purpose of metadata enhancement. 

# How to use

## Setup
Download the dataset, and the checkpoint of the finetuned model based on the schema. 
*URL: to be updated 
<br/>
| Dataset                |  Finetuned Model |  Dataset  |
| -------------------- | --------------------------------------------------------------------------------: | ------------------------------------------------------------------------: | 
| `SESAR`  |     [Model]() |     [Dataset]() |
| `OPENCONTEXT: Material`  |     [Model]() |     [Dataset]() |
| `OPENCONTEXT: Sample `  |     [Model]() |     [Dataset]() |


<br/>

Place the downloaded data in the `data` folder, and the model checkpoint in the `model` folder.

<br/>
<br/>

## Finetuning the model
1. If needed, conduct hyperparameter search by running hyperparameter_search.py file <br/>
```
python OPENCONTEXT/hyperparameter_search.py --nb_epochs 3 --batch_size 32 --lr_rate 3e-5 --label_type "material" --train_mode "custom" --pretrained_model PRETRAINED_MODEL --output_dir $OUTPUT_DIR 
```

2. Finetune the model using the train.py file 
Running this file expects the following arguments:
* `--nb_epochs` : number of epochs
* `--batch_size` : training batch size
* `--lr_rate` : value of learning rate
* `--train_mode`: whether or not to account for the class imbalance 
* `--output_dir` : the directory where the results, finetuned model checkpoint will exist
* `--label_type` ( if OPENCONTEXT dataset is used for training) : The type of label  (material / sample)
<br/>
For example, to train a model that classifies the material type of the OPENCONTEXT dataset 

```
python3 OPENCONTEXT/train.py --nb_epochs 3 --batch_size 32 --lr_rate 3e-5 --label_type "material" --train_mode "custom" --output_dir $OUTPUT_DIR 
```

<br/>

## Evaluate the finetuned model 

1. Load your model and use it for evaluation using eval.py file
* `--batch_size` : training batch size
* `--train_mode`: whether or not to account for the class imbalance 
* `--label_type` ( if OPENCONTEXT dataset is used for training) : The type of label  (material / sample)
* `--model_dir` : the directory that stores the checkpoint of the finetuned model. This would be the `model` folder where the downloaded model exists.
* `--output_dir` : the directory where the results, finetuned model checkpoint will exist

For example, to load a finetuned model that was trained on the OPENCONTEXT dataset and see its performance on the test dataset: 
```
python3 OPENCONTEXT/eval.py  --batch_size 32 --label_type "material" --train_mode "custom" --model_dir OPENCONTEXT/model/material_checkpoint --output_dir $OUTPUT_DIR 
```