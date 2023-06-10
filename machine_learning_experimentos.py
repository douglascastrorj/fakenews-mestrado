from sklearn import svm
from liardataset import loadLiarDataFrame, getLiarLabels
from extract_fatures import get_embeddings
from datetime import datetime


def embed_dataset(dataset, labels):
    print('Embedding...')
    X = []
    y = []
    for index, row in dataset.iterrows():
        text = row['text'] 
        label = row['label']

        sentence_embedding,  token_vecs = get_embeddings(text)
        # print(sentence_embedding)
        X.append(sentence_embedding.detach().cpu().numpy())
        y.append(labels[label])

    print('Embedding process completed.')
    return X, y

BERT_MODEL = 'bert-base-uncased'

labels = getLiarLabels(binary=True)

NUM_EXPERIMENTS = 1
for i in range(0, NUM_EXPERIMENTS):
    print('Running experiment: #' + str( i + 1))

    df_train, df_val, df_test = loadLiarDataFrame()

    # df1 = df_train.iloc[:2,:] # remover essa linha (linha apenas para tstar)

    X, y = embed_dataset(df_train, labels)
    # y = [0, 1] # remover essa linha (linha apenas para tstar)


    clf = svm.SVC()
    clf.fit(X, y)

    # df2 = df_test.iloc[:2,:] # remover essa linha (linha apenas para tstar)

    X_test, y_true = embed_dataset(df_test, labels)
    y_pred = clf.predict(X_test)

    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    accuracy =  accuracy_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)

    print('F1 Macro: ', f1_macro)

    print('F1 Micro: ', f1_micro)

    print('Accuracy: ', accuracy)

    print(confusion)

    file_name = 'results/mlexperiments' + datetime.now().strftime("%Y-%m-%dT%H%M")
    file = open(file_name, 'a')
    file.write('Experimento #'+ str(i + 1))
    file.write('\n')
    file.write('F1 Macro: ' + str(f1_macro))
    file.write('\n')
    file.write('F1 Micro: ' + str(f1_micro))
    file.write('\n')
    file.write('Accuracy: ' + str(accuracy))
    file.write('\n')
    file.write('Confusion Matrix\n')
    for r in range(0, len(confusion)):
        for c in range(0, len(confusion[r])):
            file.write( str(confusion[r][c]) + ', ' )
        file.write('\n')
    file.write('\n\n')
    file.close()