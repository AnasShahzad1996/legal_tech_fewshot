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
		self.cos_dist = torch.nn.CosineSimilarity(dim=0,eps=1e-6)

		self.drop = nn.Dropout()
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.config = config

	def onehot_encoder(self,labels,simple_way = False):
		if simple_way:
			y_label = torch.zeros(len(labels),self.config["num_class"])
			for i,curr_lab in enumerate(labels):
				y_label[i][self.config["Labels"].index(curr_lab[0])] = 1

			return y_label
		else:
			y_label = torch.zeros(len(labels))
			for i,curr_lab in enumerate(labels):
				y_label[i] = self.config["Labels"].index(curr_lab[0])
			return y_label


	def custom_loss(self,logits,label):
		return self.cost(logits,label)

	def custom_dist(self,support_emb,curr_emb):
		if self.config["dot"] :
			return (support_emb * curr_emb).sum()
		else:
			return self.cos_dist(support_emb,curr_emb)
			#return -(torch.pow((support_emb - curr_emb),2)).sum()


	def forward(self, support,query):

		support_emb_average = {}
		for curr_lab in self.config["Labels"]:
			support_emb_average[curr_lab] = {"count":0,"average_emb":torch.zeros(1,self.config["token_padding"],768)}
		

		for i in range(0,len(support["input_ids"])):
			output_bert = self.bert(support["input_ids"][i].reshape(1,self.config["max_seq_length"]),support["attention_mask"][i].reshape(1,self.config["max_seq_length"]))
			output_bert.requires_grad = True
			curr_lab = support["label_ids"][i]
			support_emb_average[curr_lab[0]]["average_emb"]  +=	output_bert.last_hidden_state
			support_emb_average[curr_lab[0]]["count"] +=  	1		
			

		for curr_lab in support_emb_average.keys():
			support_emb_average[curr_lab]["average_emb"] = support_emb_average[curr_lab]["average_emb"]/support_emb_average[curr_lab]["count"]
		

		logits = []
		for i in range(0,len(query["input_ids"])):
			output_bert = self.bert(query["input_ids"][i].reshape(1,self.config["max_seq_length"]),query["attention_mask"][i].reshape(1,self.config["max_seq_length"]))
			output_bert.requires_grad = True
			curr_dist = []
			for curr_lab in support_emb_average.keys():
				output_dist = self.custom_dist(support_emb_average[curr_lab]["average_emb"].reshape(self.config["token_padding"]*768),output_bert.last_hidden_state.reshape(self.config["token_padding"]*768))	
				curr_dist.append(output_dist)
			logits.append(curr_dist)

		logits = torch.tensor(logits,requires_grad = True)
		_,pred = torch.max(logits,1)
		return pred,logits

