from datasets import load_dataset
from Dataset import Dataset
from transformers import BertTokenizer, GPT2Tokenizer

#type = train / test / validation
def loadLiarDataset(tokenizer, type):

    dataset = load_dataset("liar")[type]
    # print(dataset)

    df = {'label': dataset['label'], 'text': dataset['statement']}  
   
    return df
