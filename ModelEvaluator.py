import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import time
from datetime import datetime
from Dataset import Dataset



class ModelEvaluator:
    def __init__(self, tokenizer, labels):
        self.tokenizer = tokenizer
        self.labels = labels


    def evaluate(self, modelName, model, test_data):
        print('Evaluating model')
        test = Dataset(test_data, self.tokenizer, self.labels)

        test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:
            print('\n\n-Utilizando cuda\n')
            model = model.cuda()
        else:
            print('\n\nUtilizando CPU\n\n')


        resultFileName = 'result-' + modelName +'-' +datetime.now().strftime("%Y-%m-%dT%H:%M") + '.csv'
        resultFile = open(resultFileName, 'w')
        resultFile.write('target,predicted\n')
        total_acc_test = 0
        with torch.no_grad():

            for test_input, test_label in test_dataloader:

                test_label = test_label.to(device)
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                resultFile.write(str(int(test_label[0]))+','+str(int(output.argmax(dim=1)[0]))) #printando label e predicted
                resultFile.write('\n')
                resultFile.write(str(int(test_label[1]))+','+str(int(output.argmax(dim=1)[1]))) #printando label e predicted
                resultFile.write('\n')

                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_test += acc
        
        print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
        resultFile.close()
