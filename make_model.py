import pandas as pd 
import pickle
data = pd.read_csv('data.csv') 

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Content']) 
Y = data['Label']
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X,Y)
pickle.dump(clf, open('model.sav','wb'))
pickle.dump(vectorizer,open('vectorizer.sav','wb'))