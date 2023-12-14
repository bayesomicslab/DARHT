from Load_Models import LoadModel
from Attacks.AttackWrappersProtoSAGA import AutoAttackPytorchMatGPUWrapper as APGD
from Attacks.AttackWrappersWhiteBoxJelly import PGDNativePytorch as PGDJelly
from Attacks.AttackWrappersWhiteBoxSNN import PGDNativePytorch as PGDSNN
from Utilities import DataManagerPytorch as DMP
from Utilities.ModelPlus import ModelPlus
from Attacks.attack_utilities import GetFirstCorrectlyOverlappingSamplesBalancedSingle
import pdb
import torch
model_name = "ViT-L_16_tau2.bin"
#Fill these in as you loaded the models before
file = "/mnt/home/jierendeng/kd/adv_ML_KD/LoadModels/FAT/ViT-L_16_tau2.bin"
# file = "/mnt/home/jierendeng/kd/adv_ML_KD/LoadModels/Vanilla/{}.pth".format(model_name)
# file = "/mnt/home/jierendeng/kd/adv_ML_KD/LoadModels/Vanilla/{}.bin".format(model_name)
# file = "/mnt/home/jierendeng/kd/adv_ML_KD/LoadModels/Vanilla/ViT-L_16.bin"
dataset = "cifar10"
device = "cuda"

if dataset == "cifar10":
    nclasses = 10
elif dataset == "cifar100":
    nclasses = 100
elif dataset == "tiny": #Tiny Imagenet
    nclasses = 200

#Load the model
modelP, modelType, params, bs, size = LoadModel(file, dataset, device)

x = torch.load("adv_data_{}.pt".format(model_name))
y = torch.load("adv_gt_{}.pt".format(model_name))
torch.save(pred,"adv_pred_{}.pt".format(model_name))
# yPred = modelP.predictD((x,y))
# pdb.set_trace()
print("finish")