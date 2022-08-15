#####################################################################
				# Kalamakar labels
#####################################################################

# These are the labels for kalamkar dataset
KALAMKAR_LABELS = ["NONE", "PREAMBLE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER", "ANALYSIS",
                 "PRE_RELIED", "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO"]
KALAMKAR_LABELS_PRES = ["NONE", "PREAMBLE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER",
                      "ANALYSIS", "PRE_RELIED", "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO"]
KALAMKAR_TASK = "kalamkar"

############################################################################
				# Paheli labels
############################################################################

PAHELI_LABELS = ["Argument", "Precedent", "Statute", "Facts", "Ratio of the decision", "Ruling by Lower Court", "Ruling by Present Court"]

PAHELI_LABELS_PRES = ["Argument", "Precedent", "Statute", "Facts", "Ratio of the decision", "Ruling by Lower Court", "Ruling by Present Court"]

PAHELI_TASK = "paheli/correct_format"

############################################################################
				# Pubmed labels
############################################################################

PUBMED_LABELS = ["Argument", "Precedent", "Statute", "Facts", "Ratio of the decision", "Ruling by Lower Court", "Ruling by Present Court"]

PUBMED_LABELS_PRES = ["Argument", "Precedent", "Statute", "Facts", "Ratio of the decision", "Ruling by Lower Court", "Ruling by Present Court"]

PUBMED_TASK = "pubmed"




#### config for kalamkar ######

config = {
	"bert_model": "bert-base-uncased",
	"bert_trainable": False,
	"model": "BertHSLN.__name__",
	"cacheable_tasks": [],

	"dropout": 0.5,
	"word_lstm_hs": 758,
	"att_pooling_dim_ctx": 200,
	"att_pooling_num_ctx": 15,

	"lr": 3e-05,
	"lr_epoch_decay": 0.9,
	"batch_size":  32,
	"max_seq_length": 128,
	"max_epochs": 20,
	"early_stopping": 5,
	"MAX_DOCS" : -1,
}


