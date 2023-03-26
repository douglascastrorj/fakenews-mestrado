from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):

    def __init__(self, bertModelName, dropout=0.5, num_classes=2):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(bertModelName)
        self.dropout = nn.Dropout(dropout)
        self.nonlinear = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        final_layer = self.nonlinear(dropout_output)

        return final_layer
