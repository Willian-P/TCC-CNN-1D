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


# Divisão do dataset em três conjuntos distintos: treinamento, validação e teste. Sendo 60% para treinamento, 20% para validação e 20% para teste.

# Cria um array para armazenar os índices do dataset original. Para embaralhar os índices e, posteriormente dividir os dados aleatoriamente.
index = np.array(dataset.index)

# Embaralha os índices de forma aleatória
np.random.shuffle(index)
# Número total de amostras no dataset
n = len(index)

# Seleciona os índices das primeiras 60% amostras embaralhadas para o conjunto de treinamento.
train_index = index[0:int(0.6*n)]
# As amostras da posição 60% até a posição 80% para o conjunto de validação
valid_index = index[int(0.6*n):int(0.8*n)]
# As amostras da posição 80% até o final para o conjunto de teste
test_index = index[int(0.8*n):]

# Cria um dataset para armazenar cada conjunto, treino, validação e teste, respectivamente e reindexa os índices
train_dset = dataset.loc[train_index].reset_index(drop=True)
valid_dset = dataset.loc[valid_index].reset_index(drop=True)
test_dset = dataset.loc[test_index].reset_index(drop=True)

# Obtendo os atributos (features) do dataset, excluindo a coluna 'EVOLUCAO'
input_features = dataset.columns.drop('EVOLUCAO').tolist()

# Obtendo os rótulos (target) do dataset
target = 'EVOLUCAO'
labels = dataset[target].tolist()


# Os dados são convertidos em tensores PyTorch
train_tensor_dset = TensorDataset(
    # converte os dados das colunas de entradas em tensores de ponto flutuante
    torch.tensor(train_dset[input_features].values, dtype=torch.float),
    # converte os dados da coluna de rótulos em tensores de ponto flutuante com formato de matriz
    torch.tensor(train_dset[target].values.reshape(-1,1), dtype=torch.float)
)

valid_tensor_dset = TensorDataset(
    torch.tensor(valid_dset[input_features].values, dtype=torch.float),
    torch.tensor(valid_dset[target].values.reshape(-1,1), dtype=torch.float)
)

test_tensor_dset = TensorDataset(
    torch.tensor(test_dset[input_features].values, dtype=torch.float),
    torch.tensor(test_dset[target].values.reshape(-1,1), dtype=torch.float) 
)


# Função para calcular a matriz de confusão e plotar a matriz de confusão como uma imagem
def plot_confusion_matrix(y_true, y_pred_labels):
    # Calcula a matriz de confusão usando as previsões e os rótulos verdadeiros
    cm = confusion_matrix(y_true, y_pred_labels)

    # Normaliza a matriz de confusão dividindo cada elemento pelo número total de amostras de teste
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plotar a matriz de confusão usando o seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix")

    # Salvar a figura como uma imagem
    confusion_matrix_image_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_image_path)

    # Fechar o plot para liberar recursos
    plt.close()


class SoftOrdering1DCNN(pl.LightningModule):

    def __init__(self, input_dim, output_dim, sign_size=16, cha_input=1, cha_hidden=64, 
                 K=2, dropout_input=0.2, dropout_hidden=0.2, dropout_output=0.2):
        super().__init__()

        # Calcula o tamanho da camada oculta multiplicando o tamanho do sinal (sign_size) 
        # pela quantidade de canais de entrada (cha_input).
        hidden_size = sign_size*cha_input
        # Armazena o tamanho do sinal
        sign_size1 = sign_size
        # Calcula o tamanho do sinal dividido por 2 e armazena o resultado na variável
        sign_size2 = sign_size//2

        #Calcula o tamanho da camada de saída multiplicando o tamanho do sinal dividido 
        # por 4 pela quantidade de canais ocultos (cha_hidden).
        output_size = (sign_size//4) * cha_hidden

        self.hidden_size = hidden_size # Tamanho do vetor oculto
        self.cha_input = cha_input # Número de canais de entrada
        self.cha_hidden = cha_hidden # Número de canais da camada oculta
        self.K = K # Fator de multiplicação utilizado na primeira camada convolucional
        self.sign_size1 = sign_size1 # Tamanho do sinal de entrada original
        self.sign_size2 = sign_size2 # Tamanho do sinal após a camada de pool adaptativa.
        self.output_size = output_size # Tamanho do vetor de saída após a última camada de convolução e a camada de pool
        self.dropout_input = dropout_input # Taxa de dropout aplicada à camada de entrada
        self.dropout_hidden = dropout_hidden # Taxa de dropout aplicada às camadas ocultas
        self.dropout_output = dropout_output # Taxa de dropout aplicada à camada de saída

        # Cria uma camada de normalização por lote (BatchNorm1d) com tamanho de entrada 
        # igual a input_dim e a atribui ao atributo batch_norm1 da classe.
        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        # Cria uma camada de dropout (Dropout) com taxa de dropout igual a dropout_input 
        # e a atribui ao atributo dropout1 da classe.
        self.dropout1 = nn.Dropout(dropout_input)
        # Cria uma camada densa (Linear) com tamanho de entrada igual a input_dim, tamanho de saída 
        # igual a hidden_size e sem viés (bias=False). A camada é armazenada temporariamente na variável dense1.
        dense1 = nn.Linear(input_dim, hidden_size, bias=False)
        # Aplica a normalização de peso (weight normalization) na camada dense1 e a atribui ao atributo dense1 da classe. 
        self.dense1 = nn.utils.weight_norm(dense1)

        # 1st conv layer
        self.batch_norm_c1 = nn.BatchNorm1d(cha_input)
        conv1 = conv1 = nn.Conv1d(
            cha_input, # canais de entrada
            cha_input*K, # canais de saída
            kernel_size=5, # tamanho de filtro
            stride = 1, # Deslocamento
            padding=2,  # Preenchimento
            groups=cha_input, # igual ao número de canais de entrada e sem viés
            bias=False) #  Essa camada aplica uma convolução em cada canal de entrada separadamente
        # camada convolucional definida anteriormente é normalizada pela norma dos pesos 
        self.conv1 = nn.utils.weight_norm(conv1, dim=None) 

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = sign_size2)

        # 2nd conv layer
        self.batch_norm_c2 = nn.BatchNorm1d(cha_input*K)
        self.dropout_c2 = nn.Dropout(dropout_hidden)
        conv2 = nn.Conv1d(
            cha_input*K, 
            cha_hidden, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False)
        self.conv2 = nn.utils.weight_norm(conv2, dim=None)

        # 3rd conv layer
        self.batch_norm_c3 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c3 = nn.Dropout(dropout_hidden)
        conv3 = nn.Conv1d(
            cha_hidden, 
            cha_hidden, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False)
        self.conv3 = nn.utils.weight_norm(conv3, dim=None)
        

        # 4th conv layer
        self.batch_norm_c4 = nn.BatchNorm1d(cha_hidden)
        conv4 = nn.Conv1d(
            cha_hidden, 
            cha_hidden, 
            kernel_size=5, 
            stride=1, 
            padding=2, 
            groups=cha_hidden, 
            bias=False)
        self.conv4 = nn.utils.weight_norm(conv4, dim=None)

        # cria uma camada de pooling médio unidimensional
        self.avg_po_c4 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        # cria uma camada de achatamento. É usada para transformar a saída das 
        # camadas convolucionais em um vetor unidimensional
        self.flt = nn.Flatten()

        self.batch_norm2 = nn.BatchNorm1d(output_size)
        self.dropout2 = nn.Dropout(dropout_output)
        dense2 = nn.Linear(output_size, output_dim, bias=False)
        self.dense2 = nn.utils.weight_norm(dense2)

        # Calcula a perda durante o treinamento
        self.loss = nn.BCEWithLogitsLoss()


    # Define a passagem direta (forward pass) do modelo (como os dados fluem pelas camadas)
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = nn.functional.celu(self.dense1(x))

        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)

        x = self.batch_norm_c1(x)
        x = nn.functional.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = nn.functional.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = nn.functional.relu(self.conv3(x))

        x = self.batch_norm_c4(x)
        x = self.conv4(x)
        x =  x + x_s
        x = nn.functional.relu(x)

        x = self.avg_po_c4(x)

        x = self.flt(x)

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.dense2(x)

        return x

    # Métricas
    def roc_auc(self, y_pred, y_true):
        # Calcula o ROC AUC usando sklearn.metrics.roc_auc_score
        roc_auc = roc_auc_score(y_true.cpu(), torch.sigmoid(y_pred).cpu())
        return torch.tensor(roc_auc)

    def accuracy(self, y_pred, y_true):
        # Arredonda as probabilidades previstas para obter as previsões binárias (0 ou 1)
        y_pred_labels = torch.round(torch.sigmoid(y_pred))
        # Calcula a acurácia usando sklearn.metrics.accuracy_score
        acc = accuracy_score(y_true.cpu(), y_pred_labels.cpu())
        return torch.tensor(acc)

    def f1(self, y_pred, y_true):
        # Arredonda as probabilidades previstas para obter as previsões binárias (0 ou 1)
        y_pred_labels = torch.round(torch.sigmoid(y_pred))
        # Calcula o F1-score usando sklearn.metrics.f1_score
        f1score = f1_score(y_true.cpu(), y_pred_labels.cpu())
        return torch.tensor(f1score)
    
    def calculate_confusion_matrix(self, y_pred, y_true):
        # Arredonda as probabilidades previstas para obter as previsões binárias (0 ou 1)
        y_pred_labels = torch.round(torch.sigmoid(y_pred))
        # Calcula a matriz de confusão usando as previsões e os rótulos verdadeiros
        cm = confusion_matrix(y_true.cpu(), y_pred_labels.cpu())
        return cm
    
    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['0', '1'], yticklabels=['0', '1'])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()

    # define os passos de treinamento do modelo.
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log('valid_loss', loss)
        
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_logit = self.forward(X)
        y_probs = torch.sigmoid(y_logit).detach().cpu().numpy()
        loss = self.loss(y_logit, y)
        metric = roc_auc_score(y.cpu().numpy(), y_probs)
        self.log('test_loss', loss)
        self.log('test_metric', metric)

        # MATRIZ
        # Calcula a matriz de confusão usando a função calculate_confusion_matrix
        cm = self.calculate_confusion_matrix(y_logit, y)
        # Plotar e salvar a matriz de confusão
        self.plot_confusion_matrix(cm)

        roc_auc = self.roc_auc(y_logit, y)
        acc = self.accuracy(y_logit, y)
        f1score = self.f1(y_logit, y)
        self.log('test_roc', roc_auc)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_f1', f1score, prog_bar=True)
        
    def configure_optimizers(self):
        # Cria um otimizador SGD
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer, 
                mode="min", 
                factor=0.5, 
                patience=5, 
                min_lr=1e-5),
            'interval': 'epoch',
            'frequency': 1,
            'reduce_on_plateau': True,
            'monitor': 'valid_loss',
        }
        return [optimizer], [scheduler]
    

model = SoftOrdering1DCNN(
    input_dim=len(input_features), 
    output_dim=1, # 0: Recuperado, 1: Óbito
    sign_size=16, # É utilizado em dados temporais, o dataset não possui essa característica
    cha_input=1,  # CNN1D
    cha_hidden=64, 
    K=2, 
    dropout_input=0.3, 
    dropout_hidden=0.3, 
    dropout_output=0.2
)
print("Número de atributos (input_dim):", len(input_features))


# interromper o treinamento prematuramente se a métrica de validação não melhorar, após x épocas
early_stop_callback = EarlyStopping(
   monitor='valid_loss',
   min_delta=.0,
   patience=21,
   verbose=True,
   mode='min'
)

trainer = pl.Trainer(
    callbacks=[early_stop_callback],
    min_epochs=10, 
    max_epochs=30, # 200 
    accelerator='gpu') # gpus=1


# Treinamento
trainer.fit(
    model, 
    DataLoader(train_tensor_dset, batch_size=2048, shuffle=True, num_workers=4),
    DataLoader(valid_tensor_dset, batch_size=2048, shuffle=False, num_workers=4)
)


# AUC on validation dataset
trainer.test(model, DataLoader(valid_tensor_dset, batch_size=2048, shuffle=False, num_workers=4))


# AUC on test dataset
trainer.test(model, DataLoader(test_tensor_dset, batch_size=2048, shuffle=False, num_workers=4))