import sys
import getopt
import os
import torch
import numpy as np
import argparse

sys.path.insert(0,'utils')
sys.path.insert(0,'configs')
sys.path.insert(0,'models')
import dataload_fewshot
import train_fewshot

import ProtoBert_config
import StrucShot_config
import Hier_config

import Hier_model
import ProtoBert_model
import StrucShot_model


from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup


def myfunc(argv):
	arg_config = ""
	arg_model = ""
	arg_output_dir = ""
	arg_help = "{0} -c <config> -m <model> -o output_dir".format(argv[0])
	
	try:
		opts, args = getopt.getopt(argv[1:], "hc:m:o:", ["help", "config=","model=","output_dir="])
	except:
		print(arg_help)
		sys.exit(2)
	
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print(arg_help)  # print the help message
			sys.exit(2)
		elif opt in ("-c", "--config"):
			arg_config = arg
		elif opt in ("-m", "--model"):
			arg_model = arg
		elif opt in ("-o", "--output_dir"):
			arg_output_dir = arg



############# Argument parsing done ###############



	if len(arg_output_dir)==0 or len(arg_model)==0 or len(arg_config)==0:
		print ("Invalid Arguments...")
		print ("Usage : ",arg_help)
	else:
		print ("Starting Training........")
		if arg_model == "ProtoBert":
			
			config = ""
			try:
				config  = getattr(ProtoBert_config, arg_config)
			except:
				print("The config does not exist in ProtoBert_config")
			
			model 					= ProtoBert_model.ProtoBert(config)
			optimizer_func  		= AdamW(model.parameters(),lr=config["lr"])
			train_dataloader 		= DataLoader(dataload_fewshot.FewShotDataset_per_doc(config,config["train_dataset"]), batch_size=config['batch_size'])
			validation_dataloader 	= DataLoader(dataload_fewshot.FewShotDataset_per_doc(config,config["val_dataset"]), batch_size=config['batch_size'])
			scheduler 				= get_linear_schedule_with_warmup(optimizer_func,num_warmup_steps=config["num_warmup_steps"],num_training_steps=config["num_training_steps"])
			
			model, metric  = train_fewshot.train(config, model,optimizer_func,scheduler,train_dataloader,validation_dataloader)

		elif arg_model == "StrucShot":
			pass
		elif arg_model == "Hier_model":
			pass
		else:
			print ("Invalid Model.....")
			print ("Model options : ProtoBert, StrucShot and Hier_model")            


if __name__ == "__main__":
	myfunc(sys.argv)
