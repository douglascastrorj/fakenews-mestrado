import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from tqdm import tqdm
from datetime import datetime
from BertClassifier import BertClassifier
from ModelTrainer import ModelTrainer
from ModelEvaluator import ModelEvaluator
from transformers import BertTokenizer, GPT2Tokenizer
import time

datapath = 'dataset-half.csv'
df = pd.read_csv(datapath)


BERT_MODEL = 'neuralmind/bert-base-portuguese-cased'  #'n
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL) 

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

labels = {'fake':0,
          'true':1,
          }

NUM_EXPERIMENTS = 1
EPOCHS = 1
LR = 1e-6
for i in range(0, NUM_EXPERIMENTS):
    print('Running experiment: #' + str( i + 1))

    np.random.seed(int(time.time()))
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=int(time.time())), 
                                        [int(.8*len(df)), int(.9*len(df))])

    model = BertClassifier(BERT_MODEL)
    trainer = ModelTrainer(tokenizer, labels)  
    trainer.train(model, df_train, df_val, LR, EPOCHS)

    evaluator = ModelEvaluator(tokenizer, labels)
    evaluator.evaluate(model, df_test)
