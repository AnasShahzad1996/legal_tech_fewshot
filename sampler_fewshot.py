import os
import numpy as np
import sys
import json
import torch

from task_fewshot import KALAMKAR_LABELS
from allennlp.common.util import pad_sequence_to_length
from sklearn.model_selection import KFold, train_test_split




def convert_bat_2_n2k(curr_doc,num_class,min_exam,max_exam,doc_num):
	sentPerLab = {}
	document,sents,tokens = curr_doc['input_ids'].shape

	for role in KALAMKAR_LABELS:
		sentPerLab[role] = []



	for i in range(0,document):
		for j in range(0,sents):
			curr_label = KALAMKAR_LABELS[curr_doc['label_ids'][0,j] ]
			sentPerLab[curr_label].append({"doc_num:":doc_num,"sent_num":j})

	return sentPerLab


def gather_all(curr_task,fewshot_config):
	print ("gather all the documents....")
	curr_task.get_folds()


	all_sent = {}
	for role in KALAMKAR_LABELS:
		all_sent[role] = []



	total_dataset = []
	for fold_num, fold in enumerate(curr_task.get_folds()):
		train_batches, dev_batches, test_batches = fold.train, fold.dev, fold.test
		print ("train_batches shape :",len(train_batches))
		print ("dev_batches shape :",len(dev_batches))
		print ("train batches shape:", len(train_batches))


		for doc_num,doc_batch in enumerate(train_batches):
			sentPerLab = convert_bat_2_n2k(doc_batch,fewshot_config['num_class'],fewshot_config['min_exam'],fewshot_config['max_exam'],doc_num)
			total_dataset.append({"real_doc":doc_batch,"sentPerLab":sentPerLab})

			for keyi in all_sent.keys():
				all_sent[keyi] = all_sent[keyi] + sentPerLab[keyi] 


	return total_dataset,all_sent