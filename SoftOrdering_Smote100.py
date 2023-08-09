# Este código testa a arquitetura SoftOrdering CNN 1D para o dataset balanceado com SMOTE 100% e normalização Scaler.
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns

# Configuração para garantir a reprodutibilidade dos resultados
SEED = 2
# Definem a semente aleatória para as bibliotecas NumPy e PyTorch
np.random.seed(SEED)
torch.manual_seed(SEED) # CPU
torch.cuda.manual_seed(SEED) # GPU
torch.cuda.manual_seed_all(SEED) # GPUs

# Configuração para garantir que a biblioteca cuDNN do PyTorch gere resultados determinísticos (usado para aceleração em GPU)
torch.backends.cudnn.deterministic = True

# Verifica se há uma GPU disponível e define o dispositivo para "cuda" (GPU) ou "cpu" (CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# Carrega o Dataset
dataset = pd.read_csv("Datasets\dataset_SMOTE85_Scaler.csv")
print(f'dataset:\n{dataset}')