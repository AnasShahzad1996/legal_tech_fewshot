import os
import torch


from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import BertModel,AutoTokenizer, AutoModel







class StrucShot(nn.Module):
	def __init__(self,dot =False):
		super(StrucShot, self).__init__()
		self.drop = nn.Dropout()
		self.bert = BertModel.from_pretrained('bert-base-uncased')

	def forward(self, support,query):
		support_emb = self.bert(support["input_ids"],support["attention_mask"])
		query_emb = self.bert(query["input_ids"],query["attention_mask"])

		print ("this is the support embedding shape : ",support_emb.shape)
		return 1