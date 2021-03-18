import pandas as pd 
import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np
class MPSEDataset:
    def __init__(self, name):
        super().__init__() 
        self.path='../datasets/credit/'
        self.wv_from_bin = self.load_word2vec()
        
    def read_credit_card_data(self):
        files = [self.path+'discredit3_tsne_cluster_1000_1.csv',  self.path +
                  'discredit3_tsne_cluster_1000_2.csv', self.path+'discredit3_tsne_cluster_1000_3.csv']
        D = self.load_data(files)
        labels= pd.read_csv(self.path+"lbl_tsne.csv")
        return D, labels

    def load_data(self,files):
        D = [pd.read_csv(f, header=None).values for f in files]
        return D
    def get_distance_matrics(self, w1, w2, ntop=10):
        words=self.get_data_wordlist( w1, w2,ntop)
        w1w0=self.wv_from_bin.distance(w1, w2)
        allwords=list(set(words[0]| words[1]))
        m1=self.get_custom_pariwise_distances(w1w0,words[0],allwords)
        m2=self.get_custom_pariwise_distances(w1w0,words[1],allwords)
        return ([m1, m2], allwords)


    def get_custom_pariwise_distances(self, w0w1, words,allwords ):
        m=np.zeros((len(allwords),len(allwords)))
        for i in range( 0, len( m)):
            for j in range(i+1, len(m)):
                if allwords[i] in words and allwords[j] in words:
                    m[i,j]=self.wv_from_bin.distance(allwords[i], allwords[j])
                else:
                    m[i,j]=w0w1
                m[j,i]=m[i,j]
        return m
 

    def get_data_wordlist(self, w1, w2, topn):
        w=self.wv_from_bin.most_similar(w1, topn=topn)
        words1=set([w0[0] for w0 in w]) | set([w1])
        w=self.wv_from_bin.most_similar(w2, topn=topn) 
        words2=set([w0[0] for w0 in w]) | set([w2])
        return [words1,words2]


    def load_word2vec(self):
        """ Load Word2Vec Vectors
            Return:
                wv_from_bin: 2.5 million of 3 million embeddings, each lengh 300
        """
        wv_from_bin = KeyedVectors.load_word2vec_format(api.load("word2vec-google-news-300", return_path=True), limit=2500000, binary=True)
        print("loaded data", len(list(wv_from_bin.vocab.keys())))
        return wv_from_bin

if __name__=='__main__':
    d=MPSEDataset("")
    d.get_distance_matrics('love', 'hate')
    print(d)
