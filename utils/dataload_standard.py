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



from task_fewshot import BatchCreator
from task_fewshot import DocumentsDataset

sys.path.insert(0,'configs')
import task_def



def log(str):
    print (str,file=sys.stderr)
    sys.stderr.flush()



def kalamkar_task(train_batch_size, max_docs, data_folder="datasets/",file_names=None):
    return Task(task_def.KALAMKAR_TASK, task_def.KALAMKAR_LABELS,
                train_batch_size, 1, max_docs, short_name="PMD", labels_pres=task_def.KALAMKAR_LABELS_PRES,
                data_folder=data_folder,file_names=file_names)

###########################################
######### Helper functions above ##########
###########################################

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
                 data_folder="datasets/",file_names=None):
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
        

        self.data_folder = data_folder
        # my additions 
        self. file_names = file_names

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
        full_examples = DocumentsDataset(os.path.join(self.data_dir, f"full_{file_suffix}.txt"), max_docs=self.max_docs)
        return list(full_examples)

    def _load_train_dev_test_examples(self, file_suffix='scibert') -> Fold:

        train_examples = DocumentsDataset(self.data_folder+"/"+self.file_names[0],max_docs=self.max_docs)

        dev_examples = DocumentsDataset(self.data_folder+"/"+self.file_names[1],max_docs=self.max_docs)

        test_examples = []
        if len(self.file_names[2]) == 0:
            test_examples = DocumentsDataset(self.data_folder+"/"+self.file_names[1],max_docs=self.max_docs)

        return [(train_examples, dev_examples, test_examples)]

    def truncate_train_examples(self, train_examples):
        if self.portion_training_data < 1.0:
            train_examples = list(train_examples)
            new_len = int(len(train_examples) * self.portion_training_data)
            #log(f"Truncating training examples with factor {self.portion_training_data} from {len(train_examples)} to {new_len}")
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
        #log(f"Loading data with {self.num_folds} folds...")
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
        #log(f"Creating batches for {self.num_folds} folds...")
        for train, dev, test in folds_examples:
            train_batches = self._get_batches(train)
            dev_batches = self._get_batches(dev)
            test_batches = self._get_batches(test)

            self.folds.append(Fold(train_batches, dev_batches, test_batches))
        #log(f"Creating batches finished.")
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