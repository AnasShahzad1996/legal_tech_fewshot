import numpy as np
import os
import task_fewshot
import sampler_fewshot
import dataload_fewshot

from datetime import datetime


def log(str):
	print (str,file=sys.stderr)
	sys.stderr.flush()


config = {
	"bert_model": "bert-base-uncased",
	"bert_trainable": False,
	"model": "BertHSLN.__name__",
	"cacheable_tasks": [],

	"dropout": 0.5,
	"word_lstm_hs": 758,
	"att_pooling_dim_ctx": 200,
	"att_pooling_num_ctx": 15,

	"lr": 3e-05,
	"lr_epoch_decay": 0.9,
	"batch_size":  32,
	"max_seq_length": 128,
	"max_epochs": 20,
	"early_stopping": 5,
	"MAX_DOCS" : -1,
}




############################
fewshot_config = {
	'invariant_class' 	: 	True,
	'num_class' 		:	5,
	'max_exam' 			:	5,
	'min_exam' 			:	1,
	'epochs'			:	1,
}
############################


def main():
	print ("Creating the task........")
	curr_task = task_fewshot.kalamkar_task(train_batch_size=config["batch_size"], max_docs=config["MAX_DOCS"],data_to_load="inter/",percenti=None) 
	selected_classes = np.random.choice(curr_task.labels,fewshot_config['num_class'],replace=False)


	total_dataset,all_sent = sampler_fewshot.gather_all(curr_task,fewshot_config)
	data_train,data_train_loader= dataload_fewshot.ret_dataloader_per_doc(total_dataset,all_sent,task_fewshot.KALAMKAR_LABELS,fewshot_config)

	# define model and optimizer here

	for epoch in range(0,fewshot_config['epochs']):
		for support_batch,query_batch in data_train_loader:
			pass

if __name__ == '__main__':
	main()