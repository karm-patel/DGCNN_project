import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import time
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader 
from torch.utils.data import random_split

import torch_geometric.transforms as T

from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda")
cpu_device = torch.device("cpu")

class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, gcn_channels = [32, 32, 32, 1], num_classes=2):
        super(GNNStack, self).__init__()
        self.convs = nn.ModuleList()
        self.gcn_channels = [input_dim] + gcn_channels
        for i in range(len(self.gcn_channels)-1):
            self.convs.append(pyg_nn.GCNConv(self.gcn_channels[i], self.gcn_channels[i+1]))
        
        self.sort_pool = pyg_nn.SortAggregation(k=30)
        self.conv5 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=sum(gcn_channels), stride=sum(gcn_channels))
        self.pool = nn.MaxPool1d(2, 2)
        self.conv6 = nn.Conv1d(16, 32, 5, 1)
        self.classifier_1 = nn.Linear(352, 128)
        self.drop_out = nn.Dropout(0.5)
        self.classifier_2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)

        self.num_layers = len(gcn_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)
        
        ops = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.tanh(x)
            ops.append(x)
            
        x = torch.cat(ops, dim=-1) # (total nodes, 97)
        x = self.sort_pool(x, batch) # (B, 97*30)
        x = x.view(x.size(0), 1, x.size(-1)) # (B, 1, 97*30)
        x = self.relu(self.conv5(x)) # (B, 16, 30) -- applied kernel size of 97 in only vertical direction 
        x = self.pool(x) 
        x = self.relu(self.conv6(x))
        x = x.view(x.size(0), -1)
        out = self.relu(self.classifier_1(x))
        out = self.drop_out(out)
        log_probs = F.log_softmax(self.classifier_2(out), dim=-1)

        return out, log_probs

    def loss(self, pred, label):
        return F.nll_loss(pred, label)