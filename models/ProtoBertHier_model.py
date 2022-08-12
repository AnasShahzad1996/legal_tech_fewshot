import torch
import math
import copy

from allennlp.common.util               import pad_sequence_to_length
from allennlp.modules.seq2seq_encoders  import PytorchSeq2SeqWrapper
from allennlp.nn.util                   import masked_mean, masked_softmax
from transformers                       import BertModel
from allennlp.modules                   import ConditionalRandomField




class BertTokenEmbedder(torch.nn.Module):
    def __init__(self, config):
        super(BertTokenEmbedder, self).__init__()



        self.bert = BertModel.from_pretrained(config["bert_model"])
        self.bert_trainable = config["bert_trainable"]
        self.bert_hidden_size = self.bert.config.hidden_size
        self.cacheable_tasks = config["cacheable_tasks"]


        for param in self.bert.parameters():
            param.requires_grad = self.bert_trainable

    def forward(self, batch):
        documents, sentences, tokens = batch["input_ids"].shape

        if "bert_embeddings" in batch:
            return batch["bert_embeddings"]

        attention_mask = batch["attention_mask"].view(-1, tokens)
        input_ids = batch["input_ids"].view(-1, tokens)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # shape (documents*sentences, tokens, 768)
        bert_embeddings = outputs[0]



        return bert_embeddings




class ProtoBertHier(torch.nn.Module):
    '''
    Model for Baseline ProtoBert
    '''
    def __init__(self, config):
        super(ProtoBertHier, self).__init__()

        self.bert = BertTokenEmbedder(config)
        self.dropout = torch.nn.Dropout(config["dropout"])
        self.config = config

    def all_classes_present(self,train_batch):
        print (train_batch)
        unique_vals = set(train_batch)
        if len(unique_vals) == self.config["Labels"]:
            return True
        return False


    def forward(self, batch, labels=None, output_all_tasks=False):


        if self.all_classes_present(labels) :
            documents, sentences, tokens = batch["input_ids"].shape

            print ("These are the : ",documents,sentences,tokens,batch.keys())
            # shape (documents*sentences, tokens, 768)
            #bert_embeddings = self.bert(batch)


            # in Jin et al. only here dropout
            #bert_embeddings = self.dropout(bert_embeddings)
            #print ("Bert Embeddins output shape: ",bert_embeddings.keys())


            bert_embeddings = torch.zeros(documents*sentences,tokens,768)
        else:
            return None

        return None