		#print ('mein dhoondney ko zamanay mein',self.train_examples[0].sentences[0])

		self.class2sent = {}
		for i in task_fewshot.KALAMKAR_LABELS:
			self.class2sent[i] = []

		for doc_index,curr_doc in enumerate(self.train_examples):
			for indexi,curr_sent in enumerate(self.train_examples[doc_index].sentences):
				mapped_class = self.train_examples[doc_index].labels[indexi]
				self.class2sent[mapped_class].append({'document':doc_index,'sentence':indexi})
				if len(self.train_examples[doc_index].sentences[indexi]) > self.max_sent:
					self.max_sent = len(self.train_examples[doc_index].sentences[indexi])
				self.max_len += 1

		self.max_len = int(self.max_len/config['max_exam'])



_______________________________________________________________________


		for keyi in self.class2sent.keys():
			sent2choose = np.random.randint(1,high=self.config['max_exam'],size=1)
			length_sent = range(0,len(self.class2sent[keyi]))

			random_ind = np.random.choice(length_sent,sent2choose)
			sent2choosemap = np.array(self.class2sent[keyi])[random_ind]
			for curr_sent in sent2choosemap:
				self.support_sentences.append(self.train_examples[curr_sent['document']].sentences[curr_sent['sentence']])
				self.support_labels.append(self.train_examples[curr_sent['document']].labels[curr_sent['sentence']])		# ignore the index
		

		token_ids = []
		attention_masks = []
		label_ids = []
		for sentence, label in zip(self.support_sentences, self.support_labels):
			# sentence already tokenized
			if isinstance(sentence, list):
				tok_ids = sentence
			else:
				tok_ids = [int(t) for t in sentence.split()]
			attention_mask = [1] * len(tok_ids)

			# map label id
			label_id = task_fewshot.KALAMKAR_LABELS.index(label)


			token_ids.append(tok_ids)
			attention_masks.append(attention_mask)
			label_ids.append(label_id)



		support_batch = {
		    "sentence_mask": pad_sequence_to_length([1] * len(token_ids), desired_length=self.max_sent),
            "input_ids": token_ids,
            "attention_mask": attention_masks,
            "label_ids": label_ids
        }
		query_batch = {}



		class FewShotDataset(Dataset):
	def __init__(self, total_dataset, all_sent,classes,config):
		super(Dataset, self).__init__()
		print ("Initializing the dataset")


		self.total_dataset = total_dataset
		self.all_sent = all_sent
		self.max_len = 0


		self.classes = classes
		self.config = config


		self.max_sent = 0

		# train generator 
		train_generator = task_fewshot.DocumentsDataset('datasets/kalamkar/inter/train_format.txt',max_docs=-1)
		self.train_examples = []
		for i in train_generator:
			self.train_examples.append(i)


		# dev generator
		dev_generator = task_fewshot.DocumentsDataset('datasets/kalamkar/inter/dev_format.txt',max_docs=-1)
		self.dev_examples = []
		for i in dev_generator:
			self.dev_examples.append(i)

		# test generator
		test_generator = task_fewshot.DocumentsDataset('datasets/kalamkar/inter/dev_format.txt',max_docs=-1)
		self.test_examples = []
		for i in test_generator:
			self.test_examples.append(i)

		#print ('mein dhoondney ko zamanay mein',self.train_examples[0].sentences[0])

		self.class2sent = {}
		for i in task_fewshot.KALAMKAR_LABELS:
			self.class2sent[i] = []

		for doc_index,curr_doc in enumerate(self.train_examples):
			for indexi,curr_sent in enumerate(self.train_examples[doc_index].sentences):
				mapped_class = self.train_examples[doc_index].labels[indexi]
				self.class2sent[mapped_class].append({'document':doc_index,'sentence':indexi})
				if len(self.train_examples[doc_index].sentences[indexi]) > self.max_sent:
					self.max_sent = len(self.train_examples[doc_index].sentences[indexi])
				self.max_len += 1

		self.max_len = int(self.max_len/config['max_exam'])



	def __len__(self):
		return 1
		return self.max_len

	def __getitem__(self, index):
		## move this to either get_item or another function likee get+bash
		self.support_sentences = []
		self.support_labels = []

		for keyi in self.class2sent.keys():
			sent2choose = np.random.randint(1,high=self.config['max_exam'],size=1)
			length_sent = range(0,len(self.class2sent[keyi]))

			random_ind = np.random.choice(length_sent,sent2choose)
			sent2choosemap = np.array(self.class2sent[keyi])[random_ind]
			for curr_sent in sent2choosemap:
				self.support_sentences.append(self.train_examples[curr_sent['document']].sentences[curr_sent['sentence']])
				self.support_labels.append(self.train_examples[curr_sent['document']].labels[curr_sent['sentence']])		# ignore the index
		

		token_ids = []
		attention_masks = []
		label_ids = []
		for sentence, label in zip(self.support_sentences, self.support_labels):
			# sentence already tokenized
			if isinstance(sentence, list):
				tok_ids = sentence
			else:
				tok_ids = [int(t) for t in sentence.split()]
			attention_mask = [1] * len(tok_ids)

			# map label id
			label_id = task_fewshot.KALAMKAR_LABELS.index(label)


			token_ids.append(tok_ids)
			attention_masks.append(attention_mask)
			label_ids.append(label_id)



		support_batch = {
		    "sentence_mask": pad_sequence_to_length([1] * len(token_ids), desired_length=self.max_sent),
            "input_ids": token_ids,
            "attention_mask": attention_masks,
            "label_ids": label_ids
        }
		query_batch = {}


		return support_batch,query_batch



def ret_dataloader(total_dataset,all_sent,classes,config):
	data_train = FewShotDataset(total_dataset,all_sent, classes,config)
	data_train_loader = DataLoader(data_train, batch_size=config_dataloader['batch_size'])
	return data_train,data_train_loader