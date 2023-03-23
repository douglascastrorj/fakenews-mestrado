from torch import nn
from transformers import GPT2Model

class GptClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        
        super(GptClassifier, self).__init__()

        self.gpt = GPT2Model.from_pretrained('gpt2')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.gpt.config.n_embd, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask=None):
        
        output = self.gpt(input_ids=input_id)
        pooled_output = output.last_hidden_state.mean(dim=1) # média dos embeddings dos tokens
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer