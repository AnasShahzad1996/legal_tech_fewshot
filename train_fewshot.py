import numpy as np
import os
import getopt
import sys
# If it causes an error you can change it
sys.path.insert(1,'configs')
sys.path.insert(0,'models')

#import wandb



def validate(config, model,validation_dataloader):
	pass


def train(config, model,optimizer_func,scheduler,train_dataloader,validation_dataloader):
	metrics = {}
	wandb.init(project=config["project_name"])
	


	curr_iter = 0
	curr_train_loss = 0.0
	best_f1 = 0.0


	for curr_epoch in range(0,config["max_epochs"]):
		print ("Epoch number : ",(curr_epoch+1))

		for support_batch,query_batch,every_class_present in train_dataloader:		
			if every_class_present:
				# training dataloader
				curr_iter += 1

				model.train()
				logits,pred = model.forward(support_batch,query_batch)
				loss = model.custom_loss(logits,query["label_ids"])
				loss.backward()

				



				return model,metrics



	print ("Finished training model.....")
	return model,metrics

