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
            node_feats = self._get_node_features(self.data)
            data = Data(x=node_feats,y=self._buy_or_not(self._get_return_ratio(self.data)))

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

    def _get_node_features(self, crypo):
        all_node_feats = []
        all_node_feats.append(crypo["Open"])
        # feature of c_open / c_close / c_low
        all_node_feats.append(crypo["Open"])
        all_node_feats.append(crypo["High"])
        all_node_feats.append(crypo["Low"])
        all_node_feats.append(crypo["Close"])
        all_node_feats.append(crypo["Volume"])
        all_node_feats.append(self._get_return_ratio(self.data))
        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)
        
    
    def _get_return_ratio(self,crypo):
        return_ratio = []
        data = np.array(crypo["Close"])
        for i in range(len(data)):
            if i == 0:
             return_ratio.append(0)
            else:
                return_ratio.append((data[i]-data[i-1])/data[i-1])
        return return_ratio
    
    def _buy_or_not(self,returnRatio):
        butOrNot=[]
        for ratio in returnRatio:
            butOrNot.append((ratio>=0)*1)
        butOrNot = np.asarray(butOrNot)
        return torch.tensor(butOrNot, dtype=torch.int64)

data=CrypoDataset(root="Dataset/Crypto/")
