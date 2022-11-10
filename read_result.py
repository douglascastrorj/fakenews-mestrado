import pandas as pd
import torch

from torchmetrics import F1Score, ConfusionMatrix, Accuracy
from torchmetrics.functional import precision_recall


#lendo arquivo
datapath = 'result.csv'
df = pd.read_csv(datapath)

test = []
predicted = []

for tupl in df.itertuples():
    test.append(tupl.target)
    predicted.append(tupl.predicted)

print(len(predicted), len(test))

preds = torch.tensor(predicted)
target = torch.tensor(test)


print("Accuracy: ", Accuracy()(preds, target))

#  CALCULANDO F1 SOCRE
f1 = F1Score(num_classes=3)
result = f1(preds, target)

print(result)


#CALCULANDO PRECISION E RECALL
pr = precision_recall(preds, target, average='macro', num_classes=2)

print(pr)

pr = precision_recall(preds, target, average='micro')

print(pr)


# CALCULANDO MATRIZ DE CONFUSAO
confmat = ConfusionMatrix(num_classes=2)
print(confmat(preds, target))