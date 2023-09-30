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

def train_epoch(opt, gnn_model, train_dataloader):
    train_losses = []
    tqdm_obj = tqdm(train_dataloader)
    gnn_model.train()
    for each in tqdm_obj:
        embs, log_probs = gnn_model(each.to(device))
        y_pred = torch.argmax(log_probs, axis=-1)
        loss = gnn_model.loss(log_probs, each.y)
        loss.backward()
        opt.step()
        opt.zero_grad() 
        train_losses.append(loss.detach().to(cpu_device))
        tqdm_obj.set_description_str(f"Train loss: {loss.item()}")
    
    return train_losses, embs, log_probs

def val_epoch(opt, gnn_model, val_dataloader):
    val_losses = []
    with torch.no_grad():
        gnn_model.eval()
        tqdm_obj = tqdm(val_dataloader)
        for each in tqdm_obj:
            embs, log_probs = gnn_model(each.to(device))
            y_pred = torch.argmax(log_probs, axis=-1)
            loss = gnn_model.loss(log_probs, each.y)
            val_losses.append(loss.detach().to(cpu_device))
            tqdm_obj.set_description_str(f"Val loss: {loss.item()}")
    
    return val_losses, embs, log_probs