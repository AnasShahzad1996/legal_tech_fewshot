import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import BertModel





class ProtoBert(nn.Module):
    def __init__(self):
        super(ProtoBert, self).__init__()
        self.drop = nn.Dropout()
        self.bert = BertModel.from_pretrained("bert_models/bert_uncased.bin")

    def forward(self, support,query):
        return 1