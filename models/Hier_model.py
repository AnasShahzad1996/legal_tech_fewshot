import os
import torch


from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import BertModel,AutoTokenizer, AutoModel





class HierFewshot(nn.Module):
	def __init__(self,dot =False):
		super(HierFewshot, self).__init__()
		self.drop = nn.Dropout()
		self.bert = BertModel.from_pretrained('bert-base-uncased')

	def forward(self, batch,label):
		return 1