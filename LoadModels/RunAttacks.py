from Load_Models import LoadModel
from Attacks.AttackWrappersProtoSAGA import AutoAttackPytorchMatGPUWrapper as APGD
from Attacks.AttackWrappersWhiteBoxJelly import PGDNativePytorch as PGDJelly
from Attacks.AttackWrappersWhiteBoxSNN import PGDNativePytorch as PGDSNN
from Utilities import DataManagerPytorch as DMP
from Utilities.ModelPlus import ModelPlus
from Attacks.attack_utilities import GetFirstCorrectlyOverlappingSamplesBalancedSingle
import argparse


def LoadData(size, dataset, bs):
    h = size[0]
    w = size[1]
    norm = True
    if dataset == "cifar10":
        test_loader = DMP.get_CIFAR10_loaders_test(img_size_H = h, img_size_W = w, bs = bs, norm = norm)
    elif dataset == "cifar100":
        test_loader = DMP.get_CIFAR100_loaders_test(img_size_H = h, img_size_W = w, bs = bs)
    elif dataset == "tiny":
        #Will need the tiny imagenet folder to load this
        test_loader = DMP.LoadTinyImageNetValidationData("data//tiny-imagenet-200", (h,w), bs)
    return test_loader

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch White-box Adversarial Attack Test')
    parser.add_argument('--model', type=str, default="WRN", help="decide which network to use,choose from smallcnn,resnet18,WRN")
    parser.add_argument('--model1', type=str, default="cifar10", help="choose from cifar10,svhn")
    parser.add_argument('--model2', type=str, default="cifar10", help="choose from cifar10,svhn")
    # model_name = "ViT-B_32"
    args = parser.parse_args()
    file = args.model 
    file1 = args.model1
    file2 = args.model2
    #Fill these in as you loaded the models before
    # file = "/mnt/home/jierendeng/kd/adv_ML_KD/LoadModels/FAT/FAT-resnet164.tar"
    # file = "/mnt/home/jierendeng/kd/adv_ML_KD/LoadModels/FAT/ViT-B_32_tau2.bin"
    # file1 = "/mnt/home/jierendeng/kd/adv_ML_KD/LoadModels/Vanilla/ViT-B_32.bin"
    # file = "/mnt/home/jierendeng/kd/adv_ML_KD/LoadModels/FAT/ViT-L_16_tau2.bin"
    # file = "/mnt/home/jierendeng/kd/adv_ML_KD/LoadModels/Vanilla/{}.bin".format(model_name)
    # file = "/mnt/home/jierendeng/kd/adv_ML_KD/LoadModels/Vanilla/{}.bin".format(model_name)
    # file1 = "/mnt/home/jierendeng/kd/adv_ML_KD/LoadModels/Vanilla/ViT-L_16.bin"

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
    model1, modelType, params, bs, size = LoadModel(file1, dataset, device)
    model2, modelType, params, bs, size = LoadModel(file2, dataset, device)

    #Load the data
    dataLoader = LoadData(size, dataset, bs)

    #Get 1000 samples that the model already correctly classifies
    cleanLoader, accuracy = GetFirstCorrectlyOverlappingSamplesBalancedSingle("cuda", 1000, nclasses, dataLoader, modelP)
    x,y = DMP.DataLoaderToTensor(cleanLoader)
    cleanLoader = DMP.TensorToDataLoader(x, y.long(), batchSize = bs)
    print(accuracy)

    #Attack parameters
    eps_max = .031 #attack strength
    eps_step = .005 #attack step size
    num_steps = 40 #number of steps
    clipMin = 0 #maximum bound for attack, data is in [0,1]
    clipMax = 1 #minimum bound for attack

    #Get attack data loaders
    if "jelly" in file:
        advLoader = PGDJelly(device, cleanLoader, modelP, eps_max, eps_step, num_steps, clipMin, clipMax, False)
    elif "snn" in file:
        advLoader = PGDSNN(device, cleanLoader, modelP, eps_max, eps_step, num_steps, clipMin, clipMax, False)
    else:
        advLoader = APGD(device, cleanLoader, modelP, eps_max, eps_step, num_steps, clipMin, clipMax) 
    
    import torch
    import pdb
    
    x,y = DMP.DataLoaderToTensor(advLoader)
    # pdb.set_trace()
    acc = modelP.validateD(advLoader)
    acc_1 = model1.validateD(advLoader)
    acc_2 = model2.validateD(advLoader)
    # pred =  modelP.predictD(advLoader) #get accuracy on adversarial samples
    # torch.save(x,"adv_data_{}.pt".format(model_name))
    # torch.save(y,"adv_gt_{}.pt".format(model_name))
    # torch.save(pred,"adv_pred_{}.pt".format(model_name))
    
    print("attack model ", file)
    print("model 1", file1)
    print("model 2", file2)
    print("attack model acc",acc)
    print("model 1 acc ",acc_1)
    print("model 2 acc ",acc_2)
    # print(model_name)
    # print(file)
    # print("finished")
    # x = torch.load("adv_data.pt")
    # y = torch.load("adv_gt.pt")
    # x = data["x"]
    # y = data["y"]
    # pdb.set_trace()
    print("finished")
    # advLoader = DMP.TensorToDataLoader(x,y, batchSize = bs)
