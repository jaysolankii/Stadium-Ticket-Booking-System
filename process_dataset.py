import os 
import numpy as np
import pandas as pd
import string 

test_dir = "train_test_mails/test-mails"
train_dir = "train_test_mails/train-mails"
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
def text_process(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords]
    return " ".join(text)

file_contents = []
labels = []

for filename in os.listdir(test_dir):
    if filename.startswith("spm"):
        label = 1
    else:
        label = 0
    with open(os.path.join(test_dir,filename),'r',encoding='utf-8') as file: 
        content = file.read() 
    file_contents.append(text_process(content))
    labels.append(label)
df = pd.DataFrame({'Content': file_contents, 'Label': labels})
df.to_csv('test.csv', index=False)

file_contents = []
labels = []
for filename in os.listdir(train_dir):
    if filename.startswith("spm"):
        label = 1
    else:
        label = 0
    with open(os.path.join(train_dir,filename),'r',encoding='utf-8') as file: 
        content = file.read() 
    file_contents.append(text_process(content))
    labels.append(label)
df = pd.DataFrame({'Content': file_contents, 'Label': labels})
df.to_csv('train.csv', index=False)
