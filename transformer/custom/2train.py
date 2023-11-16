from preprocessing import KinectDataset
from model import PosePredictionTransformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

#TensorBoard writer
writer = SummaryWriter()