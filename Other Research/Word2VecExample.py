# Importing the Dataset + Text Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Corona_NLP_train.csv', encoding='latin1')
import re 
texts=[]
for i in range(0,len(dataset)):
  text = re.sub('[^a-zA-Z]' , ' ', dataset['OriginalTweet'][i])
  text = text.lower()
  text = text.split()
  x = len(text) if text.count('https') ==0  else text.index('https') 
  text = text[: x ]
  text = [t for t in text if not t=='https']
  text = ' '.join(text)
  texts.append(text)

# Training the word2vec model
from gensim.models import Word2Vec
sentences = [line.split() for line in texts]

w2v =Word2Vec(sentences, size=100, window=5, workers=4, iter=10, min_count=5)

#Visualising word vectors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def display_pca_scatterplot(model, words=None, sample=0):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [ word for word in model.vocab ]
        
    word_vectors = np.array([model[w] for w in words])

    twodim = PCA().fit_transform(word_vectors)[:,:2]
    
    plt.figure(figsize=(6,6))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.05, y+0.05, word)

#Exporting pre trained embeddings + Saving/Loading our model
import gensim.downloader
print(list(gensim.downloader.info()['models'].keys()))
google_news = gensim.downloader.load('word2vec-google-news-300')
google_news.most_similar('twitter')
#save/load
w2v.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
model.train([["hello", "world"]], total_examples=1, epochs=1)
