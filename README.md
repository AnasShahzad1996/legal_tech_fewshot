# Legal Tech Fewshot
This is the fewshot learning repository for rhetorical role labelling using fewshot learning


This package programatically interacts with wandb. Hence you should register your account on Wandb

## Installation of Packages

In order to install the package for local development, create a new conda environment (or any other virtual environment) with:

```
conda create --name legal_env python=3.9
conda activate legal_env
```

Then, install the package in the environmet:

```
pip install -r requirements.txt
```

If the above program does not install all the packages and exits with an error try : 
- Add a `.` in front of the above command
- On your linux system execute `sudo apt-get install build-essential `
- On your linux system execute `sudo apt-get install Cmake`



## Datasets:

Please download the datasets folder from the google drive folder and place it in this directory. The link can be given to external collaborators upon request.

After downloading you should have five datasets:
1. Kalamkar Dataset
2. Paheli Dataset
3. Pubmed20k Dataset
4. Pubmed200k Dataset
5. Nickta Dataset


## Folder Structure:

The `baseline_train.py` and  `train_fewshot.py` are the only python files outside of the folders. The rest of the structure has the following order:

```
legal_tech_fewshot
│   README.md
│   baseline_train.py
│	train_fewshot.py
|
└───configs
│   │   ProtoBert_config.py
│   │   StrucShot.py
│   │	....
│   
└───models
|   │   ProtoBert_model.py
|   │   StrucShot.py
|   |	....
|
└───utils
|	|	task_fewshot.py
|	|	dataload_fewshot.py
|	|	....
|
└───visualization
	|	stats.py

```


## Training:

In order to train your model you have to specify
- A model
- A config
- An output directory

```
python baseline_train.py -c config_bert_2_way -m ProtoBert -o training_debug

```

The config is defined separately for each model type in the `configs folder` : 

```json
config_bert_2_way = {
	"project_name"		:	"ProtoBert_2_way_5_shot",
	"bert_model"        :   "bert-base-uncased",
	"model"             :   "ProtoBert",
	"dataloader"        :   "FewShotDataset_per_doc",
	"Labels"            :   ["ANALYSIS","FAC"],
	"num_class"         :   2,
	"max_sent"          :   5,
	"min_sent"          :   1,
	"max_epochs"        :   1,
	"lr"                :   3e-05,
	"lr_epoch_decay"    :   0.9,
	"batch_size"        :   32,
	"max_seq_length"    :   128,
	"max_epochs"        :   20,
	"early_stopping"    :   5,
	"dot"               :   false,
	"dropout"           :   true,
	"train_dataset"     :   "datasets/kalamkar/inter/train_format.txt",
	"val_dataset"       :   "datasets/kalamkar/inter/dev_format.txt",
	"batch_size"        :   1,
	"num_warmup_steps"	: 	30,
	"num_training_steps":	30000,
	"token_padding"		:	128
}
```
