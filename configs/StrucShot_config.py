import task_def


############################


############# Strucshot #####################

config_structshot_12_way = {
	"bert_model"		: 	"bert-base-uncased",
	"model"				: 	"StrucShot",
	"dataloader"		:	"FewShotDataset_per_doc",
	"Labels"			: 	task_def.KALAMKAR_LABELS,
	'num_class' 		:	5,
	'max_sent' 			:	5,
	'min_sent' 			:	1,
	'max_epochs'		:	1,
	"lr"				: 3e-05,
	"lr_epoch_decay"	: 	0.9,
	"batch_size"		:  	32,
	"max_seq_length"	: 	128,
	"max_epochs"		: 	20,
	"early_stopping"	: 	5,
	"dot"				: 	False,
	"dropout"			: True
}
############################