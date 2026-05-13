import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool

class EnhancedGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=8, dropout=0.3):
        super(EnhancedGATLayer, self).__init__()
        self.gat = GATConv(in_dim, out_dim, heads=heads, dropout=dropout, concat=False)
        self.W_res = nn.Linear(in_dim, out_dim)
        self.W_gate = nn.Linear(out_dim * 2, out_dim)
        self.b_gate = nn.Parameter(torch.zeros(out_dim))
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index, edge_attr=None):
        res = self.W_res(x)
        agg = self.gat(x, edge_index, edge_attr=edge_attr)
        gate = torch.sigmoid(self.W_gate(torch.cat([res, agg], dim=-1)) + self.b_gate)
        out = (1 - gate) * res + gate * agg
        out = self.norm(out)
        out = F.gelu(out)
        return out

class StandGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, heads=8):
        super(StandGraphEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(EnhancedGATLayer(input_dim, hidden_dim, heads=heads))
        for _ in range(num_layers - 1):
            self.layers.append(EnhancedGATLayer(hidden_dim, hidden_dim, heads=heads))
        self.pool_att = nn.Linear(hidden_dim, 1)
        self.pool_transform = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
        att_weights = F.softmax(self.pool_att(x), dim=0)
        x_weighted = x * att_weights
        out = global_add_pool(x_weighted, batch)
        out = torch.tanh(self.pool_transform(out))
        return out

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states):
        q = self.W_q(hidden_states)
        k = self.W_k(hidden_states)
        score = self.V(torch.tanh(q + k)).squeeze(-1)
        alpha = F.softmax(score, dim=1)
        context = torch.sum(hidden_states * alpha.unsqueeze(-1), dim=1)
        return context

class TemporalGRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(TemporalGRUEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.temporal_att = TemporalAttention(hidden_dim * 2)

    def forward(self, x):
        out, _ = self.gru(x)
        return out

class RackLevelAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(RackLevelAttention, self).__init__()
        self.W_seq = nn.Linear(hidden_dim, hidden_dim)
        self.b_seq = nn.Parameter(torch.zeros(hidden_dim))
        self.v_seq = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h_seq):
        q_t = torch.tanh(self.W_seq(h_seq) + self.b_seq)
        alpha_t = F.softmax(self.v_seq(q_t), dim=1)
        c_final = torch.sum(alpha_t * h_seq, dim=1)
        return c_final, alpha_t

class HARGNNClassifier(nn.Module):
    def __init__(self, node_input_dim, gnn_hidden_dim, gru_hidden_dim, num_classes=3, num_racks=7):
        super(HARGNNClassifier, self).__init__()
        self.num_racks = num_racks
        self.stand_encoder = StandGraphEncoder(node_input_dim, gnn_hidden_dim)
        self.gru_encoder = TemporalGRUEncoder(gnn_hidden_dim, gru_hidden_dim)
        self.rack_attention = RackLevelAttention(gru_hidden_dim * 2)
        
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, gru_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(gru_hidden_dim, num_classes)
        )

    def forward(self, graphs_list):
        batch_size = graphs_list[0].num_graphs
        rack_reprs =[]
        for t in range(self.num_racks):
            g_t = graphs_list[t]
            h_t = self.stand_encoder(g_t.x, g_t.edge_index, g_t.batch)
            rack_reprs.append(h_t.unsqueeze(1))
        
        seq_input = torch.cat(rack_reprs, dim=1)
        gru_out = self.gru_encoder(seq_input)
        c_final, rack_weights = self.rack_attention(gru_out)
        logits = self.mlp(c_final)
        
        return logits, rack_weights
