import os
import torch


from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import BertModel,AutoTokenizer, AutoModel





class ProtoBert(nn.Module):
	def __init__(self,config):
		super(ProtoBert, self).__init__()

		self.cost = nn.CrossEntropyLoss()

		self.drop = nn.Dropout()
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.config = config

	def onehot_encoder(self,labels):
		y_label = torch.zeros(len(label),self.config["num_class"])
		for i,curr_lab in enumerate(labels):
			y_label[i][self.config["Labels"].index(curr_label)] = 1

		return y_label

	def custom_loss(self,logits,label):
		return self.cost(logits,label)


	def forward(self, support,query):

		support_emb_average = torch.zeros(len(support["input_ids"]),768)
		query_emb_average = torch.zeros(len(query["input_ids"]),768)

		for i in range(0,len(support["input_ids"])):
			output_bert = self.bert(support["input_ids"][i].reshape(1,self.config["max_seq_length"]),support["attention_mask"][i].reshape(1,self.config["max_seq_length"]))
			support_emb_average[i] = output_bert
			
			print (output_bert)
			print (1/0)

		#support_emb = self.bert(support["input_ids"],support["attention_mask"])
		#query_emb = self.bert(query["input_ids"],query["attention_mask"])

		#print ("this is the support embedding shape : ",support_emb.shape)
		return 1,1

