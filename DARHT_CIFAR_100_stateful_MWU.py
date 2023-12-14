import sys
sys.path.append('lib')
sys.path.insert(1,'/home/jid20004/ADV_KD/Multi_Teacher_Adv_KD-main/LoadModels')
print(sys.path)
from Load_Models import LoadModel
# from Attacks.AttackWrappersProtoSAGA import AutoAttackPytorchMatGPUWrapper as APGD
# from Attacks.AttackWrappersWhiteBoxJelly import PGDNativePytorch as PGDJelly
# from Attacks.AttackWrappersWhiteBoxSNN import PGDNativePytorch as PGDSNN
# from Utilities import DataManagerPytorch as DMP
# from Utilities.ModelPlus import ModelPlus
# from Attacks.attack_utilities import GetFirstCorrectlyOverlappingSamplesBalancedSingle
import tqdm
import os
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
import datetime
from models import *
from earlystop import *
import numpy as np
from utils import Logger
import attack_generator as attack
import pdb
# from random import sample
# from random import randint
import random
from collections import OrderedDict
import torch.nn.functional as F
from datetime import datetime
from torch.optim.lr_scheduler import _LRScheduler


print("load modules done")

parser = argparse.ArgumentParser(description='PyTorch Friendly Adversarial Training')
parser.add_argument('--epochs', type=int, default=350, metavar='N', help='number of epochs to train')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num_steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step_size', type=float, default=0.007, help='step size')
parser.add_argument('--seed', type=int, default=7, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="resnet_std",
                    help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--tau', type=int, default=0, help='step tau')
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn")
parser.add_argument('--rand_init', type=bool, default=True, help="whether to initialize adversarial sample with random noise")
parser.add_argument('--omega', type=float, default=0.001, help="random sample parameter for adv data generation")
parser.add_argument('--dynamictau', type=bool, default=True, help='whether to use dynamic tau')
parser.add_argument('--depth', type=int, default=32, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10, help='WRN width factor')
parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')
parser.add_argument('--out_dir', type=str, default='/home/jid20004/ADV_KD/Multi_Teacher_Adv_KD-main/DARHT_CIFAR_100_results/checkpoints', help='dir of output')
parser.add_argument('--resume', type=int, default='', help='whether to resume training, default: None')
parser.add_argument('--modelname', type=str, default='', required=True)
parser.add_argument('--kd_loss', type=str, default='', required=True)
args = parser.parse_args()
print(args)
# training settings
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True



out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def cal_kd_loss(teachers,std_knowledge,m_weights,target,train_data,epoch,args):

    std_kd_loss = torch.zeros(len(teachers))
    std_kd_loss.cuda()
    tch_acc = [0]*len(teachers)
    tch_loss = [0]*len(teachers)
    kd_loss = torch.tensor(0.0)
    mse_loss = nn.MSELoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    for i in range(len(teachers)):
        if isinstance(teachers[i], VisionTransformer):
                ### resize input for ViT only 
            temp_output = transforms.Resize(224)(train_data)
            temp_output = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),(0.2673342858792401, 0.2564384629170883, 0.27615047132568404))(temp_output)
            with torch.no_grad():
                tch_output = teachers[i](temp_output)[0]

        else:
            with torch.no_grad():
                tch_output = teachers[i](train_data)
    
        if args.kd_loss == "mse":
            #MSE loss
            std_kd_loss[i] = mse_loss(std_knowledge.view(-1,len(teachers),tch_output.shape[1])[:,i,:],tch_output)
        elif args.kd_loss == "kl":
            #KL loss
            std_kd_loss[i] = kl_loss(F.log_softmax(std_knowledge.view(-1,len(teachers),tch_output.shape[1])[:,i,:], dim=1),F.softmax(tch_output, dim=1))

  
        tch_pred = tch_output.max(1, keepdim=True)[1]
        tch_acc[i] += tch_pred.eq(target.view_as(tch_pred)).sum().item()   
        tch_loss[i] += nn.CrossEntropyLoss(reduction='mean')(tch_output, target)
    
    # m_weights = F.softmax(m_weights.float(),dim=0)
    m_weights = m_weights/m_weights.sum()

    current_m_weights = torch.exp(-1*torch.tensor([h.item() for h in tch_loss]))
    
    # m_weights = m_weights*current_m_weights
    
    for i in range(len(teachers)):
        kd_loss += current_m_weights[i]*std_kd_loss[i]
    # print("m_weights",m_weights)
    # print("tch_loss",tch_loss)
    return kd_loss,current_m_weights,tch_acc


def train(model, teachers, epoch, train_loader, optimizer,m_weights):

    if not torch.is_tensor(m_weights):
        m_weights = torch.tensor(m_weights)

    std_acc = 0
    cls_loss_sum = 0
    kd_loss_cum = 0

    # kd_loss_weight_sum = torch.tensor([0.0]*len(teachers))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        model.train()

        # sample data
        data_flag = random.choice([0,1])
        # print(data_flag)
        # if data_flag == 0:
        #     train_data, _, _, _ = earlystop(model, data, target, step_size=args.step_size,
        #                                                                 epsilon=args.epsilon, perturb_steps=args.num_steps, tau=0,
        #                                                                 randominit_type="uniform_randominit", loss_fn='cent', rand_init=args.rand_init, omega=args.omega)
        if data_flag == 1:
            # train_data = data
            train_data = mtard_inner_loss_ce(model, data, target, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=6.0)
        else:
            train_data = data
        # train_data = mtard_inner_loss_ce(model, data, target, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=6.0)
        optimizer.zero_grad()

        #student forward
        std_output,std_knowledge = model(train_data)
     
        std_pred = std_output.max(1, keepdim=True)[1]

        std_acc += std_pred.eq(target.view_as(std_pred)).sum().item()

        cls_loss = nn.CrossEntropyLoss(reduction='mean')(std_output, target)
        

        if cls_loss.item() > 8:  
            print("cls_loss",cls_loss.item())
            continue
       
        cls_loss_sum += cls_loss.item()
        
        if epoch < 50:
            loss = cls_loss

        elif epoch < 118 and epoch >= 50:

            kd_loss,m_weights,tch_acc = cal_kd_loss(teachers,std_knowledge,m_weights,target,train_data,epoch,args)

            loss = cls_loss + kd_loss

            kd_loss_cum += kd_loss.item()

        else:
            kd_loss,m_weights,tch_acc = cal_kd_loss(teachers,std_knowledge,m_weights,target,train_data,epoch,args)

            kd_loss_cum += kd_loss.item()

            loss = kd_loss
        
        loss.backward()
        optimizer.step()
        

    std_acc = std_acc*100/len(train_loader.dataset)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    print("Current Time =", current_time, "Epoch", epoch, "std_train_acc {:.2f} | std_kd_loss {:.4f}| | std_cls_loss {:.4f}  ".format(std_acc,kd_loss_cum, cls_loss_sum))

    
    return m_weights

    

def remove_module(module_in_checkpoint):
    new_state_dict = OrderedDict()
    for k, v in module_in_checkpoint.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict




def save_checkpoint(state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
batch_size = 256
print('==> Load Test Data')

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
### sample 1000 from testset
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test) 
subset_size = 1000
testset = torch.utils.data.random_split(testset, [subset_size, len(testset) - subset_size])[0]
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)





###### loading teachers


teacher_m_1 = WideResNet_70_16().cuda()
teacher_1_path = '/home/jid20004/ADV_KD/MTARD/teacher_models/cifar100_linf_wrn70-16_without.pt'
print("teacher_1_path", teacher_1_path)
teacher_m_1.load_state_dict(torch.load(teacher_1_path))

teacher_m_2 = VisionTransformer(config=CONFIGS["ViT-B_16"],img_size=224,zero_head=True,num_classes=100).cuda()
teacher_m_2_path ="/home/jid20004/ADV_KD/TrainedModels_rigel/ViT-B_16,cifar100,run1_15K_k10_c100_tau0__checkpoint.bin"
teacher_m_2.load_state_dict(torch.load(teacher_m_2_path))
print("teacher_m_2_path",teacher_m_2_path)

# teacher_m_3 = WideResNet(depth=22, num_classes=100, widen_factor=6, dropRate=0.0).cuda()
# teacher_3_path = '/home/jid20004/ADV_KD/MTARD/teacher_models/clean_cifar_100_resnet22.tar'
# print("teacher_3_path", teacher_3_path)
# teacher_m_3.load_state_dict(remove_module(torch.load(teacher_3_path)['state_dict']))

# teacher_m_4 = VisionTransformer(config=CONFIGS["ViT-B_16"],img_size=224,zero_head=True,num_classes=100).cuda()
# teacher_m_4_path ="/home/jid20004/ADV_KD/TrainedModels_rigel/ViT-B_16,cifar100,run1_15K_checkpoint.bin"
# teacher_m_4.load_state_dict(torch.load(teacher_m_4_path))
# print("teacher_m_4_path",teacher_m_4_path)

teacher_m_5= VisionTransformer(config=CONFIGS["ViT-B_16"],img_size=224,zero_head=True,num_classes=100).cuda()
teacher_m_5_path ="/home/jid20004/ADV_KD/TrainedModels_rigel/ViT-B_16,cifar100,run1_15K_k10_c100_tau1__checkpoint.bin"
teacher_m_5.load_state_dict(torch.load(teacher_m_5_path))
print("teacher_m_5_path",teacher_m_5_path)

# teacher_m_6= VisionTransformer(config=CONFIGS["ViT-B_32"],img_size=224,zero_head=True,num_classes=100).cuda()
# teacher_m_6_path ="/home/jid20004/ADV_KD/TrainedModels_rigel/ViT-B_32,cifar100,run1_15K_checkpoint.bin"
# teacher_m_6.load_state_dict(torch.load(teacher_m_6_path))
# print("teacher_m_6_path",teacher_m_6_path)

teacher_m_7= VisionTransformer(config=CONFIGS["ViT-B_32"],img_size=224,zero_head=True,num_classes=100).cuda()
teacher_m_7_path ="/home/jid20004/ADV_KD/TrainedModels_rigel/ViT-B_32,cifar100,run1_15K_k10_c100_tau0__checkpoint.bin"
teacher_m_7.load_state_dict(torch.load(teacher_m_7_path))
print("teacher_m_7_path",teacher_m_7_path)


# teacher_m_8_path="/home/jid20004/ADV_KD/TrainedModels_rigel/FAT-resnet164-164_cifar100_k10tau0_mom.pth.tar"
# print("teacher_m_8_path ", teacher_m_8_path)
# teacher_m_8, modelType, params, bs, size = LoadModel(teacher_m_8_path, "cifar100", "cuda")

# teacher_m_9_path="/home/jid20004/ADV_KD/TrainedModels_rigel/FAT-resnet164_R164_cifar100_k10tau2_mom.pth.tar"
# print("teacher_m_9_path ", teacher_m_9_path)
# teacher_m_9, modelType, params, bs, size = LoadModel(teacher_m_9_path, "cifar100", "cuda")


# teacher_m_10_path="/home/jid20004/ADV_KD/TrainedModels_rigel/FAT-resnet164_R164_cifar100_k10tau1_mom.pth.tar"
# print("teacher_m_10_path ", teacher_m_10_path)
# teacher_m_10, modelType, params, bs, size = LoadModel(teacher_m_10_path, "cifar100", "cuda")

# teachers = [teacher_m_1, teacher_m_2, teacher_m_3, teacher_m_4, teacher_m_5,teacher_m_6,teacher_m_7,teacher_m_8,teacher_m_9,teacher_m_10]
teachers = [teacher_m_1, teacher_m_2, teacher_m_5,teacher_m_7]



# pdb.set_trace()
 #Attack parameters
device = "cuda"
eps_max = .031 #attack strength
eps_step = .005 #attack step size
num_steps = 20 #number of steps
clipMin = 0 #maximum bound for attack, data is in [0,1]
clipMax = 1 #minimum bound for attack

print('==> Load Model')
if args.net == 'resnet_std':
    model = ResNet18_std(num_classes=100,num_teachers = len(teachers)).cuda()
elif args.net == 'mobilenet_std':
    model = Mobilenet_v2_std(num_classes=100,num_teachers = len(teachers)).cuda()
print(model)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[115, 160, 185], gamma=0.1)  # learning rates
iter_per_epoch = len(train_loader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * 1)  # warmup
# pdb.set_trace()

# pdb.set_trace()
log_path = os.path.join('/home/jid20004/ADV_KD/Multi_Teacher_Adv_KD-main/DARHT_CIFAR_100_results/logs', 'log_results_{}.txt').format(args.modelname)
if not os.path.exists(log_path):
    with open(log_path, 'w'):
        pass
    print(f"File '{log_path}' created successfully.")
else:
    print(f"File '{log_path}' already exists.")


mw = [1]*len(teachers)
start_epoch = 0
best_pgd_acc = 0
test_nat_acc = 0
fgsm_acc = 0
apgd_acc = 0
test_pgd20_acc = 0
cw_acc = 0
cls_loss = 0
best_epoch = 0
kd_loss = 0 
train_time = 0


# Resume
title = 'DARHT'
if args.resume:
    # resume directly point to checkpoint.pth.tar e.g., --resume='./out-dir/checkpoint.pth.tar'
    print('==> DARHT_MTARD Training Resuming from checkpoint ..')
    resume_path ="/home/jid20004/ADV_KD/Multi_Teacher_Adv_KD-main/DARHT_CIFAR_100_results/checkpoints/last_"+args.modelname
    print(resume_path)
    assert os.path.isfile(resume_path)
    out_dir = os.path.dirname(resume_path)
    checkpoint = torch.load(resume_path)
    start_epoch = checkpoint['epoch']
    # start_epoch = 50
    if 'pgd' in checkpoint.keys():
        best_pgd_acc = checkpoint['pgd']

    if 'mw' in checkpoint.keys():
        mw = checkpoint['mw']

    model.load_state_dict(remove_module(checkpoint['state_dict']))
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger_test = Logger(log_path, title=title, resume=True)
    logger_test.set_names(['Epoch', 'Natural Test Acc', 'FGSM Acc', 'PGD20 Acc', 'CW Acc'])
else:
    print('==> DARHT_MTARD Training')
    logger_test = Logger(log_path, title=title)
    logger_test.set_names(['Epoch', 'Natural Test Acc', 'FGSM Acc', 'PGD20 Acc', 'CW Acc'])



args.dynamictau = False
for epoch in range(start_epoch, args.epochs):
    starttime = datetime.now()
    
    # epoch = 52
    mw = train(model, teachers, epoch, train_loader, optimizer,mw)
    
    endtime = datetime.now()
    train_time = (endtime - starttime).seconds
    ## Evalutions the same as DAT.
    loss, test_nat_acc = attack.eval_clean(model, test_loader)
    loss, fgsm_acc = attack.eval_robust(model, test_loader, perturb_steps=1, epsilon=0.031, step_size=0.031,loss_fn="cent", category="Madry",rand_init=True)
    loss, test_pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=0.031, step_size=0.031 / 4,loss_fn="cent", category="Madry", rand_init=True)
    loss, cw_acc = attack.eval_robust(model, test_loader, perturb_steps=30, epsilon=0.031, step_size=0.031 / 4,loss_fn="cw", category="Madry", rand_init=True)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    # print("Current Time =", current_time)

    if epoch > 1:
        train_scheduler.step(epoch)
        
    print('Train Time: %.1f s | LR: %.4f | Natural Test Acc %.4f | FGSM Test Acc %.4f | PGD20 Test Acc %.4f | CW Test Acc %.4f |\n' % (
        train_time,
        optimizer.state_dict()['param_groups'][0]['lr'],
        test_nat_acc,
        fgsm_acc,
        test_pgd20_acc,
        cw_acc)
        )
    logger_test.append([epoch + 1, test_nat_acc, fgsm_acc, test_pgd20_acc, cw_acc])
    
    if test_pgd20_acc > best_pgd_acc:
        best_pgd_acc = test_pgd20_acc
        print('Saving.. new best PGD model')
        state = {
            'net': args.net,
            'epoch': epoch,
            'pgd': test_pgd20_acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'mw': mw,
            
        }
        save_checkpoint(state, checkpoint=out_dir, filename="pgd_"+args.modelname)
    
    
    save_checkpoint({
        'net': args.net,
        'epoch': epoch,
        'pgd': test_pgd20_acc,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'mw': mw,
    }, filename="last_"+args.modelname)
