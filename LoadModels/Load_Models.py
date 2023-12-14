#External imports
import torch
from spikingjelly.clock_driven.model import spiking_resnet, sew_resnet
from spikingjelly.clock_driven import neuron, surrogate

#Internal Imports
from Defenses.Vanilla import TransformerModels, BigTransferModels
from Defenses.Vanilla.ResNet import resnet164
from Defenses.SNN.resnet_spiking import RESNET_SNN_STDB
from Defenses.BaRT.ValidateBart import BaRTWrapper
from Defenses.TIT.optdefense import OptNet
from Defenses.Vanilla.vgg import vggethan
from torchvision import transforms

from Utilities import DataManagerPytorch as DMP
from Utilities.ModelPlus import ModelPlus

def Fix_Dict(sd):
    out = {}
    for key in sd.keys():
        temp = key
        if "module." in key:
            temp = key[7:]
        out[temp] = sd[key]
    return out

#file is the file path of the model you are trying to load

#modelType is the type of model you want to load
#   resnet164 - for resnet 164 model, set params = ["FAT"] when loading the FAT version
#   transfer-snn - for the normal snn model, set params = [10] when loading
#   jelly - for the jelly snn
#   BiT - for the big transfer models
#   ViT - for the Vision Transformer models

#dataset is the dataset you want to use

#params is a list containing special parameters that some models or defenses need
#   transfer-snn - set params = [10]
#   FAT ResNet-164 - set params = ["FAT"]
#   BiT 50x1 - set params = ["TiT"] (this model was intended to only be used in the TiT defense, but also works as a vanilla model in isolation)

#returns the loaded model on cpu, the input size for the model as a tuple, and the batch size for the model (change as desired below)
def LoadFile(file, modelType, dataset = "cifar10", params = []):
    if dataset == "cifar10":
        classes = 10
    elif dataset == "cifar100":
        classes = 100
    else:
        classes = 200

    if modelType == "resnet164":
        data = torch.load(file)
        if len(params) > 0 and "FAT" in params[0]:
            dict = DMP.Fix_Dict(data["state_dict"])
            size = (32,32)
        else:
            dict = DMP.Fix_Dict(data["model"])
            size = (32,32)
        model = resnet164(size[0], classes)
        model.load_state_dict(dict)
        bs = 8

    if modelType == 'transfer-snn':
        model = model = RESNET_SNN_STDB(resnet_name = "resnet20", labels = classes, dataset = dataset.upper())
        state = torch.load(file)#torch.load(pretrained_snn, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state["state_dict"], strict=False)
        model.network_update(timesteps=int(params[0]), leak=1.0)
        #model = model.to("cuda")
        size = (32,32)
        bs = 32
    if modelType == "jelly":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sg = 'ATan'
        argsNeuron = 'MultiStepParametricLIFNode'
        arch = 'sew_resnet18'
        if dataset == "cifar10":
            timeStep = 4
            num_classes = 10
        elif dataset == "cifar100":
            num_classes = 100
            timeStep = 5
        surrogate_gradient = {
                            'ATan' : surrogate.ATan(),
                            'Sigmoid' : surrogate.Sigmoid(),
                            'PiecewiseLeakyReLU': surrogate.PiecewiseLeakyReLU(),
                            'S2NN': surrogate.S2NN(),
                            'QPseudoSpike': surrogate.QPseudoSpike()
                        }
        sg_type = surrogate_gradient[sg]
        neuron_dict = {
            'MultiStepIFNode'               : neuron.MultiStepIFNode,
            'MultiStepParametricLIFNode'    : neuron.MultiStepParametricLIFNode,
            'MultiStepEIFNode'              : neuron.MultiStepEIFNode,
            'MultiStepLIFNode'              : neuron.MultiStepLIFNode,
        }
        neuron_type = neuron_dict[argsNeuron]
        model_arch_dict = {
                        'sew_resnet18'       : sew_resnet.multi_step_sew_resnet18, 
                        'sew_resnet34'       : sew_resnet.multi_step_sew_resnet34, 
                        'sew_resnet50'       : sew_resnet.multi_step_sew_resnet50,
                        'spiking_resnet18'   : spiking_resnet.multi_step_spiking_resnet18, 
                        'spiking_resnet34'   : spiking_resnet.multi_step_spiking_resnet34, 
                        'spiking_resnet50'   : spiking_resnet.multi_step_spiking_resnet50,
        }
        model_type = model_arch_dict[arch]
        model = model_type(T=timeStep, num_classes=num_classes, cnf='ADD', multi_step_neuron=neuron_type, surrogate_function=sg_type)
        dir = file
        checkpoint = torch.load(dir)
        model.load_state_dict(checkpoint["snn_state_dict"], strict=True)
        #model.to(device)
        if dataset == "cifar10":
            size = [32,32]
            bs = 24
        elif dataset == "cifar100":
            size = [32,32]
            bs = 16
    if "BiT" in modelType:
        if len(params) > 0 and "TiT" in params[0]:
            model = BigTransferModels.KNOWN_MODELS["BiT-M-R50x1"](head_size=classes, zero_head=False)
        elif "BaRT" in file:
            model = BigTransferModels.KNOWN_MODELS["BiT-M-R101x3-BaRT"](head_size=classes, zero_head=False)
        else:
            model = BigTransferModels.KNOWN_MODELS["BiT-M-R101x3"](head_size=classes, zero_head=False)
        if "BaRT" in file:
            data = torch.load(file, map_location = "cpu")
            size = (128,128)
            bs = 2
        else:
            data = torch.load(file,  map_location = "cpu")["model"]
            size = (160, 128)
            if len(params) > 0 and "TiT" in params[0]:
                size = (224,224)
            bs = 8
        dic = {}
        for key in data:
            dic[key[7:]] = data[key]
        model.load_state_dict(dic)
        del(data)

    elif modelType == "ViT":
        if "ViT-L" in file:
            config = TransformerModels.CONFIGS["ViT-L_16"]
        else:
            config = TransformerModels.CONFIGS["ViT-B_32"]
        model =TransformerModels. VisionTransformer(config, 224, zero_head=True, num_classes=classes, vis = False)
        data = torch.load(file, map_location = "cpu")
        model.load_state_dict(data)
        del(data)
        size = (224, 224)
        bs = 8
    model = model.to("cpu")
    torch.cuda.empty_cache()
    return model, size, bs

def GetLoadingParams(file, dataset):
    trans = None
    if not "TiT" in file and ("BiT" in file or "ViT" in file or "BaRT" in file):
        #if "R50x1" in file: #only for the TiT defense
        #    return -1, -1, -1
        trans = None
        if "BiT" in file or "BaRT" in file:
            modelType = "BiT"
            if "BaRT" in file:
                if file[-7] == "5":
                    params = ["5"]
                    n = 5
                elif file[-7] == "1":
                    params = ["1"]
                    n = 1
                else:
                    params = ["10"]
                    n = 10
                modelType += "-BaRT"
                trans = lambda x ,init = False: transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))((BaRTWrapper(None, n, 128, dataset).generate(x, init)))
            elif "50" in file:
                params = ["TiT"]
            else:
                params = []
        else:
            modelType = "ViT"
            if "-L" in file:
                params = ["L"]
            else:
                params = ["B"]
            if "FAT" in file:
                params[0] += "-FAT"
    elif "jelly" in file:
        #trans = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        modelType = "jelly"
        params = []
    elif "resnet164" in file:
        modelType = "resnet164"
        if "FAT" in file:
            trans = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            params = ["FAT"]
        else:
            trans = None
            params = []
    elif "snn" in file:
        if dataset == "cifar10":
            trans = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        elif dataset == "cifar100":
            trans = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        modelType = "transfer-snn"
        params = ["10"]
    else:
        if not "TiT" in file:
            return -1, -1, -1

    #Handle TiT special case
    if "TiT" in file:
        if "BiT" in file:
            modelType = "ViT-BiT"
            params = ["TiT"]

        elif "Res" in file:
            modelType = "VGG-Res"
            params = ["TiT"]

    return modelType, params, trans

def LoadModel(file, dataset, device = "cuda"):
    if dataset == "cifar10":
        labels = 10
    elif dataset == "cifar100":
        labels = 100
    elif dataset == "tiny":
        labels = 200

    modelType, params, trans = GetLoadingParams(file, dataset)
   
    model, size, bs = LoadFile(file, modelType, dataset, params)

    modelP = ModelPlus(modelType + str(params[0]), model, device, size[0], size[1], bs, labels, trans, "jelly" in file, "TiT" in file)
    return modelP, modelType, params, bs, size

if __name__ == "__main__":
    pass