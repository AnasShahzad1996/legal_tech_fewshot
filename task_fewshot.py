import os
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import sys
import json
from allennlp.common.util import pad_sequence_to_length
import bucketing as bucketing
import torch


class BatchCreator:
    def __init__(self, dataset, tokenizer, labels, batch_sentence_size, max_seq_length):
        #'''dataset: Iterable over documents
        #   tokenizer: WordPiece tokenizer of BERT. If None, then it is assumed that sentences are already tokenized.
        #   labels: possible lables of the sentences
        #   max_sequence_length: Max number of tokens for each sentences. Only required if tokenizer is provided.
        #'''
        self.dataset = dataset
        self.labels = labels
        self.batch_sentence_size = batch_sentence_size
        self.max_sequence_length = max_seq_length
        self.tokenizer = tokenizer
        self.batches = None

    def get_batches(self, task_name=None):
        self.build_batches()
        result = []
        for b in self.batches:
            batch = self.batch_to_tensor(b)
            if task_name is not None:
                batch["task"] = task_name
            result.append(batch)
        return result

    def build_batches(self):
        if self.batches is None:
            def wrap_document(doc):
                return bucketing.Record(doc.get_sentence_count(), doc)

            mapped_ds = map(wrap_document, self.dataset)
            self.batches = bucketing.bucket_records(mapped_ds, self.batch_sentence_size)

        return len(self.batches)


    def get_batches_count(self):
        batches_count = self.build_batches()
        return batches_count


    def batch_to_tensor(self, b):
        # dictionary of arrays
        tensors_dict_arrays = b.to_tensor(self.document_to_sequence_example, merge_records)
        # convert to dictionary of tensors and pad the tensors
        result = {}
        for k, v in tensors_dict_arrays.items():

            if k in ["input_ids", "attention_mask"]:
                # determine the max sentence len in the batch
                max_sentence_len = -1
                for doc in v:
                    for sentence in doc:
                        max_sentence_len = max(len(sentence), max_sentence_len)
                # pad the sentences to max sentence len
                for doc in v:
                    for i, sentence in enumerate(doc):
                        doc[i] = pad_sequence_to_length(sentence, desired_length=max_sentence_len)
            if k!='doc_name':
                result[k] = torch.tensor(v)
            else:
                result[k] = v
        return result

    def document_to_sequence_example(self, document, sentence_padding_len):

        sentences = list(document.data.sentences)
        labels = list(document.data.labels)

        #print ('sentence padding : ',sentence_padding_len)
        # pad number of sentences
        for _ in range(len(document.data.labels), sentence_padding_len):
            sentences.append("")
            labels.append("mask")

        token_ids = []
        attention_masks = []
        label_ids = []
        for sentence, label in zip(sentences, labels):
            if self.tokenizer is None:
                # sentence already tokenized
                if isinstance(sentence, list):
                    tok_ids = sentence
                else:
                    tok_ids = [int(t) for t in sentence.split()]
            else:
                tok_ids = self.tokenizer.encode(sentence, add_special_tokens=True, max_length=128)

            attention_mask = [1] * len(tok_ids)

            # map label id
            label_id = self.labels.index(label)


            token_ids.append(tok_ids)
            attention_masks.append(attention_mask)
            label_ids.append(label_id)

        #print ('sentence mask : ',len(pad_sequence_to_length([1] * document.length, desired_length=sentence_padding_len)))
        #print ('label_ids : ',len(label_ids))
        #print ('attention_mask mask : ',len(attention_mask))


        return {
            "sentence_mask": pad_sequence_to_length([1] * document.length, desired_length=sentence_padding_len),
            "input_ids": token_ids,
            "attention_mask": attention_masks,
            "label_ids": label_ids,
            "doc_name": document.data.doc_name
        }


def merge_records (merged, r):
    if merged is None:
        merged = dict()
        for k in r:
            merged[k] = [] 
    
    for k in r:
        merged[k].append(r[k])
    
    return merged
    
def one_hot(num, v):
    r = np.zeros(num, dtype=int)
    r[v] = 1
    return r

class InputDocument:
    """Represents a document that consists of sentences and a label for each sentence"""
    
    def __init__(self, sentences, labels, doc_name):
        """sentences: array of sentences labels: array of labels for each sentence """
        self.sentences = sentences
        self.labels = labels
        self.doc_name = doc_name

    def get_sentence_count(self):
        return len(self.sentences)



class DocumentsDataset:
    def __init__(self, path, max_docs=-1):
        self.path = path    
        self.length = None
        self.max_docs = max_docs
    
    #Adapter functions for Iterator 
    def __iter__(self):
        return self.readfile()
    
    def __len__(self):
        return self.calculate_len()    
    
    
    def calculate_len(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length
    
    def readfile(self):
        """Yields InputDocuments """
        read_docs = 0
        with open(self.path, encoding="utf-8") as f:
            sentences, tags = [], []
            doc_name=''
            for line in f:
                if self.max_docs >= 0 and read_docs >= self.max_docs:
                    return
                line = line.strip()
                if not line:
                    if len(sentences) != 0:
                        read_docs += 1
                        yield InputDocument(sentences, tags,doc_name)
                        sentences, tags = [], []
                        doc_name = ''
                elif not line.startswith("###"):
                    ls = line.split("\t")
                    if len(ls) < 2:
                        continue
                    else:
                        tag, sentence = ls[0], ls[1]
                    sentences += [sentence]
                    tags += [tag]

                elif line.startswith("###"):
                    doc_name = line.replace("###","").strip()

####################################################################
                # helper classes above
####################################################################

def log(str):
	print (str,file=sys.stderr)
	sys.stderr.flush()


#####################################################################
				# Kalamakar labels
#####################################################################

#"DEFAULT", 'mask', 
KALAMKAR_LABELS = ["NONE", "PREAMBLE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER", "ANALYSIS",
                 "PRE_RELIED", "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO"]
KALAMKAR_LABELS_PRES = ["NONE", "PREAMBLE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER",
                      "ANALYSIS", "PRE_RELIED", "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO"]
KALAMKAR_TASK = "kalamkar"


def kalamkar_task(train_batch_size, max_docs, data_folder="datasets/",data_to_load="inter",percenti=None):
    return Task(KALAMKAR_TASK, KALAMKAR_LABELS,
                train_batch_size, 1, max_docs, short_name="PMD", labels_pres=KALAMKAR_LABELS_PRES,
                data_folder=data_folder,data_to_load=data_to_load,percenti=percenti)



class Fold:
    def __init__(self, train, dev, test):
        self.train = train
        self.dev = dev
        self.test = test

class Task:
    def __init__(self, task_name, labels, train_batch_size, num_fols, max_docs=-1,
                 dev_metric="weighted-f1", portion_training_data=1.0,
                 task_folder_name=None, short_name=None,
                 labels_pres=None,
                 data_folder="datasets/",data_to_load="file_name",percenti=None):
        self.labels_pres = labels_pres
        self.short_name = short_name
        self.task_name = task_name
        self.labels = labels
        self.data_dir = os.path.join(data_folder, task_name if task_folder_name is None else task_folder_name)
        self.train_batch_size = train_batch_size
        self.num_folds = num_fols
        self.max_docs = max_docs
        self.folds = None
        self.folds_examples = None
        self.dev_metric = dev_metric
        self.portion_training_data = portion_training_data
        


        # my additions 
        self.percenti = percenti
        self.data_to_load = data_to_load

    def get_labels_pres_titled(self):
        '''Labels ordered in presentation-order titled. '''
        return [l.title() for l in self.labels_pres]

    def get_labels_titled(self):
        '''Labels titled. '''
        return [l.title() for l in self.labels]

    def _get_batches(self, examples):
        ds_builder = BatchCreator(
            examples,
            tokenizer=None,
            labels=self.labels,
            batch_sentence_size=self.train_batch_size,
            max_seq_length=None
        )
        batches = ds_builder.get_batches(task_name=self.task_name)
        return batches

    def _load_full_set(self, file_suffix='scibert'):
        '''Returns one Fold. '''
        log("Loading tokenized data...")
        full_examples = DocumentsDataset(os.path.join(self.data_dir, f"full_{file_suffix}.txt"), max_docs=self.max_docs)
        log("Loading tokenized data finished.")
        return list(full_examples)

    def _load_train_dev_test_examples(self, file_suffix='scibert') -> Fold:

        log("Loading tokenized data...")
        directory_string = self.data_dir + "/"+self.data_to_load
        if self.percenti is None:
            directory_string = directory_string + "/train_format.txt"
        else:
            directory_string = directory_string + "/train_format_" + self.percenti + ".txt"

        train_examples = DocumentsDataset(directory_string,
                                          max_docs=self.max_docs)

        dev_string = self.data_dir + "/"+self.data_to_load + "/dev_format.txt"
        dev_examples = DocumentsDataset(dev_string, max_docs=self.max_docs)
        
        # change this parameter 
        test_file_not_available = True
        if test_file_not_available :
            test_examples = DocumentsDataset(dev_string,max_docs=self.max_docs)
        else:
            test_examples = DocumentsDataset(test_file_str = os.path.join(self.data_dir, f"test__format_{file_suffix}.txt"), max_docs=self.max_docs)


        train_examples = self.truncate_train_examples(train_examples)

        log("Loading tokenized data finished.")
        return [(train_examples, dev_examples, test_examples)]

    def truncate_train_examples(self, train_examples):
        if self.portion_training_data < 1.0:
            train_examples = list(train_examples)
            new_len = int(len(train_examples) * self.portion_training_data)
            log(f"Truncating training examples with factor {self.portion_training_data} from {len(train_examples)} to {new_len}")
            train_examples = train_examples[0: new_len]
        return train_examples

    def get_all_examples(self, file_suffix='scibert'):
        if self.num_folds == 1:
            train, dev, test = self._load_train_dev_test_examples(file_suffix)[0]
            all_examples = []
            all_examples += list(train)
            print(len(all_examples))
            all_examples += list(dev)
            print(len(all_examples))
            all_examples += list(test)
            print(len(all_examples))
            return all_examples
        else:
            return self._load_full_set(file_suffix)

    def get_folds_examples(self, file_suffix='scibert'):
        if self.folds_examples is not None:
            return self.folds_examples
        self.folds_examples = []
        log(f"Loading data with {self.num_folds} folds...")
        if self.num_folds == 1:
            self.folds_examples = self._load_train_dev_test_examples(file_suffix=file_suffix)
        else:
            full_examples = np.array(self._load_full_set(file_suffix=file_suffix))
            self.folds = []
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=268)
            for train_index, test_index in kf.split(full_examples):
                train_and_dev = full_examples[train_index]
                test = full_examples[test_index]
                train, dev = train_test_split(train_and_dev, test_size=1.0 / self.num_folds, shuffle=False)

                train = self.truncate_train_examples(train)

                self.folds_examples.append((train, dev, test))

        return self.folds_examples

    def get_folds(self):
        if self.folds is not None:
            return self.folds

        folds_examples = self.get_folds_examples()
        self.folds = []
        log(f"Creating batches for {self.num_folds} folds...")
        for train, dev, test in folds_examples:
            train_batches = self._get_batches(train)
            dev_batches = self._get_batches(dev)
            test_batches = self._get_batches(test)

            self.folds.append(Fold(train_batches, dev_batches, test_batches))
        log(f"Creating batches finished.")
        return self.folds

    def get_stats_counts(self):
        counts = dict()
        all_examples = self.get_all_examples()
        counts["docs"] = len(all_examples)
        counts["sentences"] = 0
        for d in all_examples:
            for l in d.labels:
                counts["sentences"] += 1
                if l in counts:
                    counts[l] += 1
                else:
                    counts[l] = 1

        return counts

    def get_test_label_counts(self, fold_num):
        fold_num = fold_num % self.num_folds
        _, _, test = self.get_folds_examples()[fold_num]

        counts = [0] * len(self.labels)
        for d in test:
            for l in d.labels:
                label_id = self.labels.index(l)
                counts[label_id] += 1
        return counts