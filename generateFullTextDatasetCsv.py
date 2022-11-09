import pandas as pd
from os import walk

def processString(str = ''):
    newStr = str.strip()
    newStr = newStr.replace('\n', '')
    # newStr = newStr.replace(',', '')
    newStr = newStr.replace(',', ';')
 
    return newStr

def generate_compiled_dataset_file(path = 'Fake.br-Corpus-master/full_texts'):
    filenamesTrue = next(walk(path+'/true'), (None, None, []))[2]  # [] if no file
    filenamesFake = next(walk(path+'/fake'), (None, None, []))[2]
    
    pathsFake = [ path + '/fake/' + fileName for fileName in filenamesFake ]
    pathsTrue =  [ path + '/true/' + fileName for fileName in filenamesTrue ]

    output = open('dataset-full.csv', 'w')
    output.write('label,preprocessed_news\n')
    
    for path in pathsFake:
        file = open(path, 'r')
        output.write('fake,'+processString(file.read())+'\n')

    for path in pathsTrue:
        file = open(path, 'r')
        output.write('true,'+processString(file.read())+'\n')

    output.close()

generate_compiled_dataset_file()


#test read dataset
datapath = 'dataset-full.csv'
df = pd.read_csv(datapath)
print(df.head())
