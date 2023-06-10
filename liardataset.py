from datasets import load_dataset
import pandas as pd

def getLiarLabels(binary=False):
    if binary:
        return { 'fake': 0, 'half-true': 0, 'mostly-true': 1, 'true': 1, 'barely-true': 0, 'pants-fire': 0}
    else:
        return { 'fake': 0, 'half-true': 1, 'mostly-true': 2, 'true': 3, 'barely-true': 4, 'pants-fire': 5}

def getLiarLabelsNames(labels):
    liarLabels = ['fake', 'half-true', 'mostly-true', 'true', 'barely-true', 'pants-fire']
    return [liarLabels[labels[i]] for i in range(0, len(labels))]

def loadLiarDataFrame():
    dataset = load_dataset("liar")

    df_train = pd.DataFrame({ 'text': dataset['train']['statement'], 'label': getLiarLabelsNames(dataset['train']['label'])})
    df_validation = pd.DataFrame({ 'text': dataset['validation']['statement'], 'label': getLiarLabelsNames(dataset['validation']['label'])})
    df_test = pd.DataFrame({ 'text': dataset['test']['statement'], 'label': getLiarLabelsNames(dataset['test']['label'])})

    return df_train, df_validation, df_test



def loadLiarDataFrameBinary():
    dataset = load_dataset("liar")

    df_train = pd.DataFrame({ 'text': dataset['train']['statement'], 'label': getLiarLabelsNames(dataset['train']['label'])})
    df_validation = pd.DataFrame({ 'text': dataset['validation']['statement'], 'label': getLiarLabelsNames(dataset['validation']['label'])})
    df_test = pd.DataFrame({ 'text': dataset['test']['statement'], 'label': getLiarLabelsNames(dataset['test']['label'])})

    return df_train, df_validation, df_test