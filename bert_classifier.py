import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import time
from datetime import datetime


BERT_MODEL = 'neuralmind/bert-base-portuguese-cased' #'neuralmind/bert-base-portuguese-cased'  #'neuralmind/bert-large-portuguese-cased'  #'bert-base-cased' 

# datapath = 'Fake.br-Corpus-master/preprocessed/pre-processed.csv'
datapath = 'dataset-full.csv'
df = pd.read_csv(datapath)
df.head()

# df.groupby(['label']).size().plot.bar()


tokenizer = BertTokenizer.from_pretrained(BERT_MODEL) 
labels = {'fake':0,
          'true':1,
          }

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [labels[label] for label in df['label']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['preprocessed_news']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer



## funcoes para treinar modelo

def train(model, train_data, val_data, learning_rate, epochs):
    print('Training model')

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
        print('\n\n-Utilizando cuda\n')
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        print('\n\nUtilizando CPU\n\n')

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_acc_val / len(val_data): .3f}')
                

def evaluate(model, test_data):
    print('Evaluating model')
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        print('\n\n-Utilizando cuda\n')
        model = model.cuda()
    else:
        print('\n\nUtilizando CPU\n\n')


    resultFileName = 'result-'+datetime.now().strftime("%Y-%m-%dT%H:%M") + '.csv'
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

#  ------- MAIN --------
NUM_EXPERIMENTS = 4
EPOCHS = 10
LR = 1e-6
for i in range(0, NUM_EXPERIMENTS):
    print('Running experiment: #' + str( i + 1))

    np.random.seed(int(time.time()))
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=int(time.time())), 
                                        [int(.8*len(df)), int(.9*len(df))])

    # print(len(df_train),len(df_val), len(df_test))

    model = BertClassifier()
          
    train(model, df_train, df_val, LR, EPOCHS)

    evaluate(model, df_test)


