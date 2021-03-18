import pandas as pd 

class MPSEDataset:
    def __init__(self, name):
        super().__init__() 
        
    def read_credit_card_data(self):
        path = './datasets/credit/'
        files = [path+'discredit3_tsne_cluster_1000_1.csv',  path +
                  'discredit3_tsne_cluster_1000_2.csv', path+'discredit3_tsne_cluster_1000_3.csv']
        D = self.load_data(files)
        labels= pd.read_csv(path+"lbl_tsne.csv")
        return D, labels

    def load_data(self,files):
        D = [pd.read_csv(f, header=None).values for f in files]
        return D


    