import numpy as np
import shap
import torch
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import kci
from models import HARGNNClassifier

class HARGNNGuidedCausalDiscovery:
    def __init__(self, model_path, data_loader, device):
        self.device = device
        self.model = HARGNNClassifier(node_input_dim=3, gnn_hidden_dim=128, gru_hidden_dim=256).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.data_loader = data_loader
        self.background_data = None
        self.test_data = None
        self.extract_data()

    def extract_data(self):
        bg_list =[]
        test_list =[]
        for i, (graphs, labels) in enumerate(self.data_loader):
            flat_graphs =[]
            for g in graphs:
                flat_graphs.append(g.x.view(g.x.size(0), -1))
            flat_tensor = torch.cat(flat_graphs, dim=1).detach().cpu().numpy()
            if i < 5:
                bg_list.append(flat_tensor)
            else:
                test_list.append(flat_tensor)
        self.background_data = np.concatenate(bg_list, axis=0)
        self.test_data = np.concatenate(test_list, axis=0)

    def model_wrapper(self, x_np):
        x_tensor = torch.tensor(x_np, dtype=torch.float32).to(self.device)
        graphs_list = self.reconstruct_graphs(x_tensor)
        with torch.no_grad():
            logits, _ = self.model(graphs_list)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def reconstruct_graphs(self, x_tensor):
        num_samples = x_tensor.shape[0]
        num_racks = 7
        feats_per_rack = x_tensor.shape[1] // num_racks
        graphs_list =[]
        from torch_geometric.data import Data, Batch
        for r in range(num_racks):
            rack_data = x_tensor[:, r*feats_per_rack:(r+1)*feats_per_rack]
            batch_g =[]
            for i in range(num_samples):
                node_feats = rack_data[i].view(-1, 3)
                edge_index = self.build_complete_graph(node_feats.shape[0]).to(self.device)
                batch_g.append(Data(x=node_feats, edge_index=edge_index))
            graphs_list.append(Batch.from_data_list(batch_g))
        return graphs_list

    def build_complete_graph(self, num_nodes):
        edge_index =[]
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def compute_shap_values(self):
        explainer = shap.KernelExplainer(self.model_wrapper, shap.kmeans(self.background_data, 50))
        shap_values = explainer.shap_values(self.test_data[:100])
        return shap_values

    def filter_candidates_by_shap(self, shap_values, target_class=0, top_k=20):
        class_shap = np.abs(shap_values[target_class]).mean(axis=0)
        top_indices = np.argsort(class_shap)[::-1][:top_k]
        filtered_data = self.test_data[:, top_indices]
        return filtered_data, top_indices

    def generate_causal_graph(self, target_class=0):
        shap_values = self.compute_shap_values()
        filtered_data, top_indices = self.filter_candidates_by_shap(shap_values, target_class)
        cg = pc(filtered_data, 0.01, kci)
        
        causal_matrix = cg.G.graph
        return causal_matrix, top_indices

    def analyze_paths(self, causal_matrix, top_indices):
        num_nodes = causal_matrix.shape[0]
        paths =[]
        for i in range(num_nodes):
            for j in range(num_nodes):
                if causal_matrix[i, j] == -1 and causal_matrix[j, i] == 1:
                    paths.append((top_indices[i], top_indices[j]))
        return paths

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from dataset import HotRolledSteelDataset, collate_temporal_graphs
    from torch.utils.data import DataLoader
    test_dataset = HotRolledSteelDataset('data/test.csv')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_temporal_graphs)
    
    causal_module = HARGNNGuidedCausalDiscovery('best_hargnn.pth', test_loader, device)
    
    matrix_under_crown, indices_0 = causal_module.generate_causal_graph(target_class=0)
    paths_0 = causal_module.analyze_paths(matrix_under_crown, indices_0)
    
    matrix_over_crown, indices_2 = causal_module.generate_causal_graph(target_class=2)
    paths_2 = causal_module.analyze_paths(matrix_over_crown, indices_2)
    
    np.save('causal_matrix_under_crown.npy', matrix_under_crown)
    np.save('causal_matrix_over_crown.npy', matrix_over_crown)

if __name__ == "__main__":
    main()
