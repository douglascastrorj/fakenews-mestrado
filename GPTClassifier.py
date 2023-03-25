from torch import nn
from transformers import GPT2Model

class GPTClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        
        super(GPTClassifier, self).__init__()

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
    
class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, num_classes:int, gpt_model_name:str):
        super(SimpleGPT2SequenceClassifier,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        self.fc1 = nn.Linear(393216, num_classes)
        
    def forward(self, input_id, mask):
        """
        Args:
                input_id: encoded inputs ids of sent.
        """
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size,-1))
        return linear_output