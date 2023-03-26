import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from tqdm import tqdm
from datetime import datetime
from BertClassifier import BertClassifier
from GPTClassifier import GPTClassifier, SimpleGPT2SequenceClassifier
from ModelTrainer import ModelTrainer
from GPTTrainer import GPTTrainer
from ModelEvaluator import ModelEvaluator
from transformers import BertTokenizer, GPT2Tokenizer
import time
from liardataset import loadLiarDataFrame, getLiarLabels

datapath = 'liar-test.csv'
df = pd.read_csv(datapath)


# BERT_MODEL = 'neuralmind/bert-base-portuguese-cased'  #'n
GPT_MODEL = 'gpt2'# 'pierreguillou/gpt2-small-portuguese' #

def getGPTTokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained(GPT_MODEL)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# tokenizer = BertTokenizer.from_pretrained(BERT_MODEL) 
tokenizer = getGPTTokenizer()


# labels = {'fake':0,
#           'true':1,
#           }

labels = getLiarLabels(binary=True)

NUM_EXPERIMENTS = 1
EPOCHS = 5
LR = 1e-6
for i in range(0, NUM_EXPERIMENTS):
    print('Running experiment: #' + str( i + 1))

    np.random.seed(int(time.time()))
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=int(time.time())), 
                                        [int(.8*len(df)), int(.9*len(df))])
    # df_train, df_val, df_test = loadLiarDataFrame()

    # model = BertClassifier(BERT_MODEL)
    model = SimpleGPT2SequenceClassifier(num_classes=2, gpt_model_name=GPT_MODEL)
    trainer = ModelTrainer(tokenizer, labels)  
    trainer.train(model, df_train, df_val, LR, EPOCHS, batch_size=2)

    torch.save(model.state_dict(), 'models/'+GPT_MODEL+'-'+datetime.now().strftime("%Y-%m-%dT%H:%M")+'.model')

    evaluator = ModelEvaluator(tokenizer, labels)
    evaluator.evaluate('GPT', model, df_test)
