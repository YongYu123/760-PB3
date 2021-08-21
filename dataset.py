import os
import torch
from torch_geometric.data import Dataset, Data
import glob
import pandas as pd
import numpy as np

class CrypoDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CrypoDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        crypto_path = glob.glob("./" + self.root + "/raw/*.csv")
        crypto_filenames = []
        for crypto_file in crypto_path:
            name = os.path.basename(crypto_file)
            crypto_filenames.append(name)
        return crypto_filenames


    @property
    def processed_file_names(self):
        crypto_path = glob.glob("./" + self.root + "/raw/*.csv")
        return [f'data_{i}.pt' for i in range(len(crypto_path))]
        
    def download(self): 
        pass

    def process(self):
        i = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            self.data = pd.read_csv(raw_path)
            node_feats = self._get_node_features(self.data, i)
            edge_index = torch.tensor([np.random.randint(5),np.random.randint(5)], dtype=torch.long)
            data = Data(x=node_feats,edge_index=edge_index,y=self._buy_or_not(self.data))
            

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

    def _get_node_features(self, crypo, node_no):
        all_node_feats = []
        all_node_feats.append(node_no)
        all_node_feats.extend(crypo["Open"])
        all_node_feats.extend(crypo["High"])
        all_node_feats.extend(crypo["Low"])
        all_node_feats.extend(crypo["Close"])
        all_node_feats.extend(crypo["Volume"])
        all_node_feats.extend(self._get_return_ratio(self.data))
        return torch.FloatTensor(all_node_feats).unsqueeze(1)
        
    
    def _get_return_ratio(self,crypo):
        return_ratio = []
        data = np.array(crypo["Close"])
        for i in range(len(data)):
            if i == 0:
             return_ratio.append(0)
            else:
                return_ratio.append((data[i]-data[i-1])/data[i-1])
        return return_ratio
    
    def _buy_or_not(self,crypo):
        butOrNot=0
        data = np.array(crypo["Close"])
        if(data[-1]>data[0]):
            butOrNot=1
        return torch.FloatTensor([butOrNot])
