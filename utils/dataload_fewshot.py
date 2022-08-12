import os
import numpy as np
import sys
import json
import torch
import torchvision
import task_fewshot

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from allennlp.common.util import pad_sequence_to_length
from sklearn.model_selection import KFold, train_test_split


import task_fewshot



class FewShotDataset_per_doc(Dataset):
	def __init__(self,config,dataset_str):
		super(Dataset, self).__init__()
		self.config = config
		

		# train generator 
		train_generator = task_fewshot.DocumentsDataset(dataset_str,max_docs=-1)
		self.train_examples = []
		for i in train_generator:
			self.train_examples.append(i)


	def __len__(self):
		return int(len(self.train_examples)/2)

	def __getitem__(self, index):

		# first half is the support set while the second half is the query set
		support_index = index
		query_index = index + int(len(self.train_examples)/2)


		self.support_sentences 	= list(self.train_examples[support_index].sentences)
		self.support_labels 	= list(self.train_examples[support_index].labels)
		support_batch 			= {"sentence_mask":1,"attention_mask":[],"input_ids":[],"label_ids":[]}


		max_sent_per_label		= self.config["max_sent"]
		max_sent_pad			= self.config["token_padding"]
		support_track_class 	= {}
		for curr_task in self.config["Labels"]:
			support_track_class[curr_task] = 0



		for curr_sent,curr_label in zip(self.support_sentences,self.support_labels):
			if curr_label in self.config["Labels"]:
				if support_track_class[curr_label] == 0:
					curr_ids = [int(t) for t in curr_sent.split()]
					support_batch["input_ids"].append( torch.tensor(curr_ids + ([0]*(max_sent_pad - len(curr_ids))) ))
					support_batch["label_ids"].append(curr_label)
					support_batch["attention_mask"].append(torch.tensor( ([1] * len(curr_ids)) + ([0]*(max_sent_pad - len(curr_ids))) ))
					support_track_class[curr_label] = support_track_class[curr_label] + 1


				elif support_track_class[curr_label] >= max_sent_per_label:
					pass
				else:
					random_ch = np.random.randint(10)
					if random_ch % 2 == 0:
						curr_ids = [int(t) for t in curr_sent.split()]
						support_batch["input_ids"].append( torch.tensor(curr_ids + ([0]*(max_sent_pad - len(curr_ids))) ))
						support_batch["label_ids"].append(curr_label)
						support_batch["attention_mask"].append(torch.tensor( ([1] * len(curr_ids)) + ([0]*(max_sent_pad - len(curr_ids))) ))
						support_track_class[curr_label] = support_track_class[curr_label] + 1


		support_batch["sentence_mask"] = torch.tensor(pad_sequence_to_length([1]*len(support_batch["input_ids"]),desired_length=len(support_batch["input_ids"])))



		# constructing the query batch
		query_batch 			= {}
		self.query_sentences 	= list(self.train_examples[query_index].sentences)
		self.query_labels 		= list(self.train_examples[query_index].labels)
		query_batch 			= {"sentence_mask":1,"attention_mask":[],"input_ids":[],"label_ids":[]}


		query_track_class 	= {}
		for curr_task in self.config["Labels"]:
			query_track_class[curr_task] = 0


		for curr_sent,curr_label in zip(self.query_sentences,self.query_labels):
			if curr_label in self.config["Labels"]:
				if query_track_class[curr_label] == 0 and curr_label in self.config["Labels"]:
					curr_ids = [int(t) for t in curr_sent.split()]
					query_batch["input_ids"].append( torch.tensor(curr_ids + ([0]*(max_sent_pad - len(curr_ids))) ))
					query_batch["label_ids"].append(curr_label)
					query_batch["attention_mask"].append(torch.tensor( ([1] * len(curr_ids)) + ([0]*(max_sent_pad - len(curr_ids))) ))
					query_track_class[curr_label] = query_track_class[curr_label] + 1


				elif query_track_class[curr_label] >= max_sent_per_label:
					pass
				else:
					random_ch = np.random.randint(10)
					if random_ch % 2 == 0:
						curr_ids = [int(t) for t in curr_sent.split()]
						query_batch["input_ids"].append( torch.tensor(curr_ids + ([0]*(max_sent_pad - len(curr_ids))) ))
						query_batch["label_ids"].append(curr_label)
						query_batch["attention_mask"].append(torch.tensor( ([1] * len(curr_ids)) + ([0]*(max_sent_pad - len(curr_ids))) ))
						query_track_class[curr_label] = query_track_class[curr_label] + 1




		query_batch["sentence_mask"] = torch.tensor(pad_sequence_to_length([1]*len(query_batch["input_ids"]),desired_length=len(query_batch["input_ids"])))


		every_class_present = True
		for  i in support_track_class.keys():
			if query_track_class[i] == 0 :
				every_class_present = False
			if support_track_class[i] == 0:
				every_class_present = False

		return support_batch,query_batch,every_class_present


