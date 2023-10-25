"""
Edge to Node Project.
    GNN Model
"""
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class GCNBinaryNodeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.3):
        super(GCNBinaryNodeClassifier, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim // 2)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim // 2, num_classes)

        self.dropout = nn.Dropout(dropout_rate)
        self.initialize_weights()

    def initialize_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3]:
            for name, param in conv.named_parameters():
                if "weight" in name:
                    nn.init.kaiming_uniform_(param)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        return x
