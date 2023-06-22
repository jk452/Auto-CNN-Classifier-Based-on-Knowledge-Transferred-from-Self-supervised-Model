
from torchvision import models
from torchvision import transforms
from torchvision.datasets import CIFAR100
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from pl_bolts.datamodules import CIFAR10DataModule
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
import argparse
import os
from typing import List
from typing import Optional
import logging
import sys
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from pl_bolts.models.self_supervised import SimCLR
import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
from optuna.samplers import TPESampler
import plotly
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import Accuracy
from torchvision import models
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from pl_bolts.datamodules import CIFAR10DataModule
import pytorch_lightning as pl
#from torchmetrics.functional import accuracy
import argparse
import os
from typing import List
from typing import Optional
import logging
import sys
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from pl_bolts.models.self_supervised import SimCLR
import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
from optuna.samplers import TPESampler
import plotly
from pytorch_lightning.callbacks import LearningRateMonitor
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform

from pl_bolts.models.self_supervised import SimCLR
#from pytorch_lightning.metrics.functional import accuracy
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from torchmetrics import Accuracy
from optuna.samplers import RandomSampler
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import models
from torchvision import transforms
from torchvision.datasets import CIFAR100
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
#from pl_bolts.datamodules import CIFAR10DataModule
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
import argparse
import os
from typing import List
from typing import Optional
import logging
import sys
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms
from torch.optim import Adam
from torch.nn.functional import cross_entropy
#from pl_bolts.models.self_supervised import SimCLR
import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
from optuna.samplers import TPESampler
import plotly
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import Accuracy
from torchvision import models
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
#from pl_bolts.datamodules import CIFAR10DataModule
import pytorch_lightning as pl
#from torchmetrics.functional import accuracy
import argparse
import os
from typing import List
from typing import Optional
import logging
import sys
import optuna
#from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms
from torch.optim import Adam
import torch.optim
from torch.nn.functional import cross_entropy
#from pl_bolts.models.self_supervised import SimCLR
import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
#from optuna.samplers import TPESampler
import plotly
#from pytorch_lightning.callbacks import LearningRateMonitor
#from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
#from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform

#from pl_bolts.models.self_supervised import SimCLR
#from pytorch_lightning.metrics.functional import accuracy
#from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from torchmetrics import Accuracy
from optuna.samplers import RandomSampler
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import pandas as pd
import numpy as np

BATCHSIZE = 64
CLASSES = 100
EPOCHS = 15
DIR = os.getcwd()
#pl.seed_everything(42)
SEED = 42

np.random.seed(SEED)



class Net(nn.Module):
    def __init__(self, trial):
        super(Net, self).__init__()
        #self.layers = []
        layers = []
        #self.cnnlayers = []
        cnnlayers = []
        #self.dropouts = []
        input_dim_cnn = 2048
        #cnn_layers = 1
        cnn_layers = trial.suggest_int("cnn_layers", 1, 3,step=1)
        #kernal_size_cnn = 2
        kernal_size_cnn = trial.suggest_int("kernal_size_cnn",2,3,step=1)
        kernal_size_pool = trial.suggest_int("kernal_size_pool", 2,3,step=1)
        #kernal_size_pool = 2
        for i in range(cnn_layers):
            output_dim_cnn = trial.suggest_int("n_units_cnn_l{}".format(i), 64, 512, step = 64)
            #output_dim_cnn = 512
            
            cnnlayers.append(nn.Conv2d(input_dim_cnn, output_dim_cnn, kernal_size_cnn, stride=1, padding=1)) 
            cnnlayers.append(nn.BatchNorm2d(output_dim_cnn))
            cnnlayers.append(nn.ReLU(inplace=True))
            cnnlayers.append(nn.MaxPool2d(kernal_size_pool, stride=1))
            input_dim_cnn = output_dim_cnn
        #self.cnnlayers.append(nn.AdaptiveAvgPool2d((1,1)))
        self.cnnlayers: nn.Module = nn.Sequential(*cnnlayers)
        # We optimize the number of layers, hidden untis in each layer and drouputs.
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
       
        n_layers = trial.suggest_int("n_layers", 1, 2,step=1)
        #n_layers = 1
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6,step=.1)
        #dropout_rate = .8
        input_dim = input_dim_cnn
        for i in range(n_layers):
            output_dim = trial.suggest_int("n_units_l{}".format(i), 64, 512, step = 64)
            #output_dim = 640
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.BatchNorm1d(output_dim))
            #layers.append(nn.LayerNorm((output_dim), eps=1e-05, elementwise_affine=True))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, CLASSES))

        self.layers: nn.Module = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnnlayers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return F.log_softmax(x, dim=1)

class LightningNet(pl.LightningModule):
    def __init__(self,trial,learning_rate):
        super(LightningNet,self).__init__()
        self.save_hyperparameters()
        #self.accuracy = Accuracy()
        #weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
        #simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)        
        #self.backbone = simclr.encoder
        #swav = torch.hub.load('facebookresearch/swav:main', 'resnet50')  
        #swav = torch.hub.load('facebookresearch/swav:main', 'resnet50w2')    
        #self.backbone = nn.Sequential(*(list(swav.children())[:-2]))
        #self.backbone.fc = Identity()
        #self.backbone.avgpool = Identity()
        #self.backbone.layer4 = Identity()
        weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
        simclr = SimCLR.load_from_checkpoint(weight_path, strict=False) 
        self.backbone = simclr.encoder 
        self.backbone1 = nn.Sequential(*(list(self.backbone.children())[:-2])) 
        for param in self.backbone1.parameters():
            param.requires_grad = False
        self.classifier = Net(trial)
        

    #def forward(self, data: torch.Tensor) -> torch.Tensor:
        #return self.model(data.view(-1, 32*32))
    #def feature(self,batch,batch_idx):
      #data, target = batch
      #features = self.backbone(data)

    def training_step(self,batch,batch_idx):
      data, target = batch
      if self.trainer.current_epoch == 6:
            for param in self.backbone1.parameters():
                param.requires_grad = True
      features = self.backbone1(data)
      #features = features.view(features.size(0), -1)
      output = self.classifier(features)
      #return F.nll_loss(output, target)

      loss = cross_entropy( output, target)
      #preds = torch.argmax(output, dim=1)
      #acc = self.accuracy(preds,target) 
      #accuracy = pred.eq(target.view_as(pred)).float().mean()
   
      #self.log('train_loss',loss)
      #self.log('train_acc',acc)
      #self.log('train_acc',Taccuracy)
      return loss

    def validation_step(self, batch, batch_idx: int) -> None:
      data, target = batch
      features = self.backbone1(data)
      #features = features.view(features.size(0), -1)
      
      output = self.classifier(features)
      #loss =  F.nll_loss(output, target)
      #loss = cross_entropy( output, target)
      #preds = torch.argmax(output, dim=1)
      #acc = self.accuracy(preds,target) 
      #accuracy = pred.eq(target.view_as(pred)).float().mean()
   
      #self.log('val_loss',loss)
      #self.log('val_acc',acc)
      pred = output.argmax(dim=1, keepdim=True)
      accuracy = pred.eq(target.view_as(pred)).float().mean()
      self.log("val_acc", accuracy)
      self.log("hp_metric", accuracy, on_step=False, on_epoch=True)      
      #return loss 

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,weight_decay=1e-4)
        #optimizer = torch.optim.Adagrad(self.parameters(), lr=self.hparams.learning_rate,weight_decay=1e-4)
        #optimizer = torch.optim.Adadelta(self.parameters(), lr=self.hparams.learning_rate,weight_decay=1e-4)
        #Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,weight_decay=1e-4, momentum=0.9,nesterov=True)
        #scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=EPOCHS)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,weight_decay=1e-2, momentum=0.9,nesterov=True)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=EPOCHS)
        #scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=EPOCHS)
        return [optimizer],[scheduler]

from torchvision.transforms.functional import InterpolationMode
#The mean and std of ImageNet are: mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. 
#(tensor([0.5055, 0.4740, 0.4245]), tensor([0.1780, 0.1751, 0.1755]))
transform_train = transforms.Compose([
    #transforms.RandomResizedCrop(size = 224),
    transforms.Resize(size = 224,interpolation=3 ),
    #transforms.RandomCrop(224),
    #transforms.RandomResizedCrop(size=224),
    #transforms.RandomCrop(160),
    transforms.RandomHorizontalFlip(p=.5),
    transforms.RandomRotation(10),
    #transforms.CenterCrop(size=160),
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomApply([
        #transforms.ColorJitter(brightness=0.5,
                               #contrast=0.5,
                               #saturation=0.5,
                               #hue=0.1)
        #], p=0.8),
    #transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.1780, 0.1751, 0.1755)),
    #transforms.Normalize((0.5055, 0.4740, 0.4245), (0.1780, 0.1751, 0.1755)),
    #cifar100_normalization(),
    #transforms.Normalize(mean, std)
])
transform_test = transforms.Compose([
    transforms.Resize(size = 224,interpolation=3),
    #transforms.RandomResizedCrop(size = 160),
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomResizedCrop(size=224),
    #transforms.RandomHorizontalFlip(p=.5),
    #transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #transforms.Normalize((0.5055, 0.4740, 0.4245), (0.1780, 0.1751, 0.1755)),
    #transforms.Normalize((0.5071, 0.4867, 0.4408), (0.1780, 0.1751, 0.1755)),
    #cifar100_normalization(),
    #transforms.Normalize(mean, std)
])
CIFAR100_test = datasets.CIFAR100(root = DIR, train=False, download=True,transform=transform_test)
CIFAR100_full = datasets.CIFAR100(root = DIR, train=True, download=True)
CIFAR100_train, CIFAR100_val = random_split(CIFAR100_full, [45000, 5000])
CIFAR100_train.dataset.transform = transform_train
#CIFAR10_full.transform = transform_train
CIFAR100_val.dataset.transform = transform_test
train_dataloader = DataLoader(CIFAR100_train, batch_size=BATCHSIZE, shuffle=True)
#val_dataloader = DataLoader(CIFAR10_val, batch_size=BATCHSIZE, shuffle=True)
val_dataloader = DataLoader(CIFAR100_val, batch_size=BATCHSIZE, shuffle=False)
test_dataloader = DataLoader(CIFAR100_test, batch_size=BATCHSIZE, shuffle=False)

def objective(trial: optuna.trial.Trial) -> float:
    
    #logger = pl.loggers.TensorBoardLogger("tb_logs", name="swav_10")
    learning_rate = trial.suggest_float("learning_rate", 1e-4,1e-2, log=True)
    #learning_rate = .01879095700747313
    #logger = DictLogger(trial.number)
    model = LightningNet(trial,learning_rate)
    #checkpoint_callback = ModelCheckpoint(monitor="val_acc")
    #datamodule = FashionMNISTDataModule(data_dir=DIR, batch_size=BATCHSIZE)
    checkpoint_callback = ModelCheckpoint(monitor="val_acc")
    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing = True,
        max_epochs=EPOCHS,
        gpus=1 if torch.cuda.is_available() else None,
        #precision=16,
        #accelerator="gpu",
        #devices=1,
        callbacks=[checkpoint_callback],
        #callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
    )
    hyperparameters = dict(trial = trial,learning_rate = learning_rate)
    trainer.logger.log_hyperparams(hyperparameters)    
    trainer.fit(model,train_dataloader,val_dataloader)

    return trainer.callback_metrics["val_acc"].item()

study_name = "simclr_10_n_55"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
sampler = TPESampler(multivariate=True,n_startup_trials=20,seed=SEED)
study = optuna.create_study(direction="maximize",sampler=sampler,study_name=study_name, storage=storage_name,load_if_exists=True)
study.optimize(objective, n_trials=50,timeout=None)
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


print(df)
df.to_csv('simclr_10.csv',index=False)

df_saved_files = pd.read_csv('simclr_10.csv')
print(df_saved_files)



