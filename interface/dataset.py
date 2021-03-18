import pandas as pd 

class MPSEDataset:
    def __init__(self, name):
        super().__init__() 
        self.path='../datasets/credit/'
        self.wv_from_bin = self.load_word2vec()
        print(len(self.wv_from_bin))
        
    def read_credit_card_data(self):
        files = [self.path+'discredit3_tsne_cluster_1000_1.csv',  self.path +
                  'discredit3_tsne_cluster_1000_2.csv', self.path+'discredit3_tsne_cluster_1000_3.csv']
        D = self.load_data(files)
        labels= pd.read_csv(self.path+"lbl_tsne.csv")
        return D, labels

    def load_data(self,files):
        D = [pd.read_csv(f, header=None).values for f in files]
        return D

    def load_word2vec(self):
        """ Load Word2Vec Vectors
            Return:
                wv_from_bin: 2.5 million of 3 million embeddings, each lengh 300
        """
        import gensim.downloader as api
        from gensim.models import KeyedVectors
        wv_from_bin = KeyedVectors.load_word2vec_format(api.load("word2vec-google-news-300", return_path=True), limit=2500000, binary=True)
        vocab = list(wv_from_bin.vocab.keys())
        print("Loaded vocab size %i" % len(vocab))
        return wv_from_bin

    