import os
import torch
from torch_geometric.data import Dataset, Data
import glob
import pandas as pd
import numpy as np

class CrypoDataset(Dataset):
    def __init__(self, root,transform=None, pre_transform=None):
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
        data_rows = pd.read_csv(self.raw_paths[0]).shape[0]
        return [f'data_{i}.pt' for i in range(data_rows)]
        
    def download(self): 
        pass

    def process(self):
        file_no = 0
        data_rows = pd.read_csv(self.raw_paths[0]).shape[0]
        for i in range(data_rows):
            date_time = 0
            all_node_feats=[]
            for raw_path in self.raw_paths:
            # Read data from `raw_path`.
                data_file = pd.read_csv(raw_path)
                if(date_time == 0):
                    date_time=data_file.iloc[i]["Date"]
                daily_data = data_file.loc[data_file['Date'] == date_time]
                if(daily_data.shape[0]>0):
                    node_feats = self._get_node_features(daily_data)
                    all_node_feats.append(node_feats)
            edge_index = self.generateEdge(len(all_node_feats))
            all_node_feats = torch.tensor(all_node_feats, dtype=torch.float)
            data = Data(x=all_node_feats,edge_index=edge_index,y=self._buy_or_not(data_file,date_time))
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(file_no)))
            file_no += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

    def _get_node_features(self, daily_data):
        all_node_feats = []
        all_node_feats.append(daily_data["Open"].item())
        all_node_feats.append(daily_data["High"].item())
        all_node_feats.append(daily_data["Low"].item())
        all_node_feats.append(daily_data["Close"].item())
        all_node_feats.append(daily_data["Volume"].item())
        return all_node_feats
    
    def _get_return_ratio(self,daily_data):
        return_ratio = []
        data = np.array(daily_data["Close"])
        for i in range(len(data)):
            if i == 0:
                return_ratio.append(0)
            else:
                return_ratio.append((data[i]-data[i-1])/data[i-1])
        return return_ratio
    
    def _buy_or_not(self,data_file,date_time):
        butOrNot=0
        data = np.array(data_file["Close"])
        data_row = data_file.loc[data_file['Date'] == date_time]
        if data_row.shape[0]>0:
            index = data_file.loc[data_file['Date'] == date_time].index[0]
            if(index < data_file.shape[0]-1 and data[index+1]>data[index]):
                butOrNot=1
        return torch.tensor(butOrNot, dtype=torch.float)
    def generateEdge(self,n):
        edges = []
        for i in range(n):
            for j in range(n):
                edges.append([i,j])
        return torch.tensor(edges, dtype=torch.long)
