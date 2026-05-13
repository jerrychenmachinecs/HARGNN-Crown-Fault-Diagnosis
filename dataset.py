import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

class HotRolledSteelDataset(Dataset):
    def __init__(self, df_path, window_size=5, num_racks=7):
        super(HotRolledSteelDataset, self).__init__()
        self.df = pd.read_csv(df_path)
        self.window_size = window_size
        self.num_racks = num_racks
        self.samples, self.labels = self.preprocess()

    def preprocess(self):
        grouped = self.df.groupby('coil_id')
        samples = []
        labels =[]
        for coil_id, group in grouped:
            if len(group) < self.window_size:
                continue
            feats = group.drop(columns=['coil_id', 'fault_label']).values
            label = group['fault_label'].values[-1]
            extracted_feats = self.extract_time_series_features(feats)
            samples.append(extracted_feats)
            labels.append(label)
        return samples, labels

    def extract_time_series_features(self, feats):
        seq_len, num_features = feats.shape
        params_per_rack = num_features // self.num_racks
        processed = np.zeros((self.num_racks, params_per_rack, 3))
        
        for r in range(self.num_racks):
            start_idx = r * params_per_rack
            end_idx = (r + 1) * params_per_rack
            rack_data = feats[-self.window_size:, start_idx:end_idx]
            mean_val = np.mean(rack_data, axis=0)
            var_val = np.var(rack_data, axis=0)
            roc_val = rack_data[-1] - rack_data[0]
            for p in range(params_per_rack):
                processed[r, p, 0] = mean_val[p]
                processed[r, p, 1] = var_val[p]
                processed[r, p, 2] = roc_val[p]
        return processed

    def build_complete_graph(self, num_nodes):
        edge_index =[]
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_matrix = self.samples[idx]
        label = self.labels[idx]
        graphs =[]
        for r in range(self.num_racks):
            x = torch.tensor(feat_matrix[r], dtype=torch.float32)
            edge_index = self.build_complete_graph(x.shape[0])
            g = Data(x=x, edge_index=edge_index)
            graphs.append(g)
        return graphs, torch.tensor(label, dtype=torch.long)

def collate_temporal_graphs(batch):
    num_racks = len(batch[0][0])
    batched_graphs = [[] for _ in range(num_racks)]
    labels =[]
    for graphs, label in batch:
        for r in range(num_racks):
            batched_graphs[r].append(graphs[r])
        labels.append(label)
    
    from torch_geometric.data import Batch
    final_batch =[]
    for r in range(num_racks):
        final_batch.append(Batch.from_data_list(batched_graphs[r]))
    return final_batch, torch.stack(labels)
