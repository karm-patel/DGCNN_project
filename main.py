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
from models.dgcnn import GNNStack
from utils import train_epoch, val_epoch

device = torch.device("cuda")
cpu_device = torch.device("cpu")

cur_dataset = TUDataset(root='dataset', name = 'PROTEINS')
# cur_dataset = TUDataset(root="dataset/loaded/", 
#                                name="IMDB-BINARY")


gnn_model = GNNStack(input_dim=3, hidden_dim=128, output_dim=64).to(device)

data_size = len(cur_dataset) 
train_size = int(0.8*data_size)
train_dataset, val_dataset = random_split(cur_dataset, [train_size, data_size-train_size])
len(train_dataset), len(val_dataset)

B = 512
train_dataloader = DataLoader(train_dataset, batch_size=B, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=B, shuffle=True)

opt = optim.Adam(gnn_model.parameters())
train_losses, _, _ = train_epoch(opt, gnn_model, train_dataloader)

total_train_losses, total_val_losses = [], []
n_epochs = 500
for epoch in range(n_epochs):
    train_losses, _, _ = train_epoch(opt, gnn_model, train_dataloader)
    total_train_losses.extend(train_losses)
    
    val_losses, _, _ = val_epoch(opt, gnn_model, val_dataloader)
    total_val_losses.extend(val_losses)

