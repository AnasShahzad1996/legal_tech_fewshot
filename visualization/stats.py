import numpy as np
import os
import sys
import tabulate
import json

# change this directory to prevent any errors
sys.path.insert(1,'configs')
sys.path.insert(0,'utils')


import task_def
import task_fewshot





class Doc_stat:
    def __init__(self, doc_name,sentences, labels, doc_mean,sent_per_lab):
        self.doc_name = doc_name
        self.sentences = sentences
        self.labels = labels
        self.doc_mean = doc_mean

        self.sent_per_lab = sent_per_lab



def sent_per_doc(sentences,labels,predef_labs):
	#print (predef_labs)
	sent_per_lab = np.zeros(len(predef_labs))
	max_token_len = 0
	average_token = 0
	for curr_sent,curr_label in zip(list(sentences),list(labels)):
		if len([int(t) for t in curr_sent.split()]) > max_token_len:
			max_token_len = len([int(t) for t in curr_sent.split()])
		average_token += len([int(t) for t in curr_sent.split()])

		curr_index = predef_labs.index(curr_label)
		sent_per_lab[curr_index] += 1

	average_token = average_token / len(sentences)
	return sent_per_lab,max_token_len,average_token


def document_stats(train_generator,Dataset_name,predef_labs):

	mean_sent = 0
	max_sent = 0
	max_sent_name = ""
	max_token = 0
	max_token_name = ""
	total_document = 0
	doc_list = []

	total_sent_per_label = np.zeros(len(predef_labs))


	for curr_doc in train_generator:
		if max_sent < len(curr_doc.sentences):
			max_sent = len(curr_doc.sentences)
			max_sent_name = curr_doc.doc_name
		mean_sent += len(curr_doc.sentences)
		total_document += 1


		# Per document sentence outputs
		sent_per_lab,max_token_len,average_token = sent_per_doc(curr_doc.sentences,curr_doc.labels,predef_labs)
		total_sent_per_label += sent_per_lab
		if max_token < max_token_len:
			max_token = max_token_len
			max_sent_name = curr_doc.doc_name
		doc_class = Doc_stat(curr_doc.doc_name,curr_doc.sentences,curr_doc.labels,average_token,sent_per_lab)
		doc_list.append(doc_class)

	mean_sent = mean_sent /  total_document
	print ("The",Dataset_name," dataset has",total_document,"sentences")
	print ("Document",max_sent_name,"has the highest number of sentences at :",max_sent)
	print ("Document",max_token_name,"has the sentence with highest number of tokens :",max_token)
	print ("Mean number of sentences in the",Dataset_name,"are",mean_sent)
	print ("########################################")
	table_final = tabulate.tabulate(zip(predef_labs,total_sent_per_label.tolist()),headers=["Labels","No of sentences"])
	print (table_final)
	print ("########################################")
	return doc_list

def main():
	print ("choose which dataset you want to visualize: ")
	print ("1. Kalamkar dataset")
	print ("2. Paheli dataset")
	print ("3. Pubmed20k")
	print ("4. Pubmed200k")


	curr_option = input("Option : ")


	if curr_option == "1" :
		print ("########################################")
		print ("######### Kalamkar dataset  ############")
		print ("########################################")

		train_generator = task_fewshot.DocumentsDataset('datasets/kalamkar/inter/train_format.txt',max_docs=-1)
		doc_list=document_stats(train_generator,"Kalamkar dataset",task_def.KALAMKAR_LABELS)
		

		read_dec=input("Do you want to read a particular document from the Kalamkar dataset(Y/n)?")
		if read_dec == "n" or read_dec=="N":
			print ("Exiting visualization")
			return 0
		Document_name = input("Enter document name : ")
		f = open('datasets/kalamkar/inter/train.json')
		data = json.load(f)
		document_found = False
		for curr_doc in data:
			if curr_doc['id'] == int(Document_name):
				document_found = True
				for curr_sent in curr_doc['annotations'][0]["result"]:
					print (curr_sent["value"]["text"])
					print (" Label : ",curr_sent["value"]["labels"])
					input("----- More ----")
					print ("\033[A                             \033[A")
					print ("\033[A                             \033[A")


		for curr_doc in doc_list:
			if curr_doc.doc_name == Document_name:
				print ("Document stats : ")
				table_final = tabulate.tabulate(zip(task_def.KALAMKAR_LABELS,curr_doc.sent_per_lab.tolist()),headers=["Labels","No of sentences"])
				print (table_final)

		if not document_found:
			print ("Document not found")




	elif curr_option =='2':
		pass #to be implemented
	elif curr_option == '3':
		pass# to be implemented


if __name__ == '__main__':
	main()