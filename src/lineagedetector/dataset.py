import torch, argparse, sys, json, os, warnings, re
from torch.utils.data import Dataset, ConcatDataset

sys.path.append('src')
os.chdir("src")
warnings.simplefilter(action='ignore', category=UserWarning)

from utils import *
from models.basic import *
from compare import get_feature

index_dict = {
"train" : torch.tensor([  0,   1,   2,   3,   4,   7,  10,  11,  12,  13,  14,  15,  16,
        17,  18,  21,  23,  25,  28,  29,  30,  32,  33,  34,  36,  37,
        40,  41,  43,  44,  46,  48,  49,  50,  51,  53,  54,  57,  58,
        59,  60,  61,  62,  63,  65,  68,  69,  70,  71,  72,  73,  74,
        76,  77,  78,  80,  82,  83,  84,  86,  87,  90,  91,  92,  93,
        95,  99, 100, 101, 102, 103, 104, 105, 106, 108, 109, 110, 111,
       112, 113, 115, 117, 118, 119, 120, 121, 123, 125, 127, 128, 129,
       130, 132, 133, 134, 135, 136, 137, 138, 139, 141, 142, 143, 144,
       145, 146, 147, 149, 150, 151, 152, 153, 155, 156, 158, 160, 161,
       162, 163, 167, 168, 169, 170, 171, 173, 175, 176, 177, 179, 182,
       183, 184, 185, 187, 188, 190, 192, 193, 195, 197, 199, 202, 203,
       204, 205, 207, 209, 210, 211, 212, 213, 214, 216, 217, 218, 220,
       221, 222, 223, 229, 230, 232, 234, 236, 237, 238, 239, 240, 241,
       242, 244, 246, 247, 248, 250, 251, 252, 253, 254, 255, 257, 258,
       260, 261, 263, 264, 265, 266, 268, 269, 270, 271, 274, 276, 277,
       278, 280, 282, 283, 285, 287, 288, 289, 290, 292, 294, 296, 297,
       298, 299, 300, 302, 303, 305, 306, 307, 308, 309, 312, 313, 315,
       316, 317, 318, 319, 320, 321, 323, 324, 327, 328, 329, 334, 335,
       339, 340, 341, 342, 343, 344, 347, 348, 349, 350, 351, 353, 354,
       355, 356, 357, 358, 359, 360, 362, 363, 364, 365, 366, 367, 369,
       370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 382, 384,
       386, 388, 389, 390, 391, 393, 394, 395, 396, 399, 400, 401, 403,
       405, 408, 409, 411, 412, 413, 415, 416, 417, 418, 420, 421, 422,
       423, 424, 425, 426, 429, 430, 431, 432, 436, 438, 439, 441, 442,
       443, 445, 447, 449, 450, 451, 452, 453, 454, 455, 458, 459, 460,
       461, 463, 464, 465, 466, 467, 468, 471, 472, 473, 474, 475, 478,
       483, 486, 487, 488, 489, 492, 493, 495, 496, 497, 498, 499], dtype = torch.long),
"val" : torch.tensor([ 22,  55,  66,  67,  85,  88,  96,  98, 124, 157, 159, 164, 172,
       178, 180, 181, 186, 189, 194, 196, 206, 208, 256, 295, 310, 311,
       314, 322, 326, 332, 333, 336, 361, 368, 383, 398, 404, 407, 419,
       428, 433, 434, 457, 469, 476, 479, 482, 484, 485, 494], dtype = torch.long),
"test" : torch.tensor([  5,   6,   8,   9,  19,  20,  24,  26,  27,  31,  35,  38,  39,
        42,  45,  47,  52,  56,  64,  75,  79,  81,  89,  94,  97, 107,
       114, 116, 122, 126, 131, 140, 148, 154, 165, 166, 174, 191, 198,
       200, 201, 215, 219, 224, 225, 226, 227, 228, 231, 233, 235, 243,
       245, 249, 259, 262, 267, 272, 273, 275, 279, 281, 284, 286, 291,
       293, 301, 304, 325, 330, 331, 337, 338, 345, 346, 352, 381, 385,
       387, 392, 397, 402, 406, 410, 414, 427, 435, 437, 440, 444, 446,
       448, 456, 462, 470, 477, 480, 481, 490, 491], dtype = torch.long),
}

class DetectorDataset(Dataset):
    def __init__(self, pmodelspath, cmodelspath, dataset, model, feature_index, weight_name):
        super().__init__()
        self.create_dataset(pmodelspath, cmodelspath, dataset, model, feature_index, weight_name)

    def get_weight(self, model, weight_names):
        for weight_name in weight_names.split("."):
            model = getattr(model,weight_name)
        return model
        
    def create_dataset(self, pmodelspath, cmodelspath, dataset, model, feature_index, weight_name):
        '''
            pfeatures: NumPModels x N x Nfeatures
            pfeatures_list : many pfeatures
            cfeatures: NumCModels x N x Nfeatures
            cfeatures_list : many cfeatures
            pweights: NumPmodels x Nparams
            pweight_lists: many pweights
            cweights: NumCmodels x Nparams
            cweight_lists: many cweights
        '''
        empty_dict = {}
        args = argparse.Namespace(**empty_dict)
        args.dataset = dataset
        args.batch_size = 64
        args.test_batch_size = 10
        args.model = model
        args.feature_index = feature_index
        args.finetune_flag = True

        train_data, test_data, num_classes, num_channels = get_loader(args, shuffle = False, use_cuda = True)
        args.num_classes = num_classes
        args.num_channels = num_channels
        
        json_file_p = open(pmodelspath, "r")
        pre_train_ckpt_paths_p = json.load(json_file_p)
        self.pweights = []
        self.pfeatures = []
        for pre_train_ckpt_path_p in pre_train_ckpt_paths_p:
            args.pre_train_ckpt = pre_train_ckpt_path_p
            modelp = get_model(args)
            self.pweights.append(self.get_weight(modelp, weight_name).weight.detach().cpu())
            modelp = get_submodel(modelp, args)
            self.pfeatures.append(get_feature(modelp, train_data, "cuda", 1).detach().cpu())

        self.pweights = torch.stack(self.pweights, dim = 0)
        self.pfeatures = torch.stack(self.pfeatures, dim = 0)
        
        json_file_c = open(cmodelspath, "r")
        pre_train_ckpt_paths_c = json.load(json_file_c)
        self.cweights = []
        self.cfeatures = []
        for pre_train_ckpt_path_c in pre_train_ckpt_paths_c:
            args.pre_train_ckpt = pre_train_ckpt_path_c
            modelc = get_model(args, "C")
            self.cweights.append(self.get_weight(modelc, weight_name).weight.detach().cpu())
            modelc = get_submodel(modelc, args)
            self.cfeatures.append(get_feature(modelc, train_data, "cuda", 1).detach().cpu())

        self.cweights = torch.stack(self.cweights, dim = 0)
        self.cfeatures = torch.stack(self.cfeatures, dim = 0)

        self.labels = get_real_parents(pre_train_ckpt_paths_p, pre_train_ckpt_paths_c)

        self.nump = len(self.pweights)

    def __getitem__(self, index):
        '''
            feature: NumPModels x N x Nfeatures
            weight: NumPModels x Nparams
        '''
        nump = self.nump
        cfeature = self.cfeatures[index].unsqueeze(0).unsqueeze(0).expand((nump,1,-1,-1))
        cweight = self.cweights[index].unsqueeze(0).unsqueeze(0).expand((nump,1,-1,-1))
        
        feature = torch.cat((self.pfeatures.unsqueeze(1),cfeature), dim = 1)
        weight = torch.cat((self.pweights.unsqueeze(1),cweight), dim = 1)
        label = self.labels[index]
        return feature, weight, label

    def __len__(self):
        return len(self.cfeatures)

def get_index(length, split):
    index = index_dict[split]
    return index[index < length]

def get_split(dataset):
    train_set = torch.utils.data.Subset(dataset, get_index(len(dataset), "train"))
    val_set = torch.utils.data.Subset(dataset, get_index(len(dataset), "val"))
    test_set = torch.utils.data.Subset(dataset, get_index(len(dataset), "test"))
    return train_set, val_set, test_set

def get_detector_loader(args, shuffle = True):
    train_kwargs = {'batch_size': args.batch_size, 'num_workers': 4,'shuffle': shuffle}
    val_kwargs = {'batch_size': args.test_batch_size, 'num_workers': 4,'shuffle': False}
    test_kwargs = {'batch_size': args.test_batch_size, 'num_workers': 4,'shuffle': False}

    trainsets = []
    valsets = []
    testsets = []
    for datasetname in args.datasets:
        if datasetname == "fc_MNIST":
            dataset = torch.load("path/to/dataset")
        else:
            raise NotImplementedError

        train_set, val_set, test_set = get_split(dataset)
        trainsets.append(train_set)
        valsets.append(val_set)
        testsets.append(test_set)
    
    train_loader = torch.utils.data.DataLoader(
        ConcatDataset(trainsets),
        **train_kwargs)
    val_loader = torch.utils.data.DataLoader(
        ConcatDataset(valsets),
        **val_kwargs)
    test_loaders = [
        torch.utils.data.DataLoader(test_set, **test_kwargs) for test_set in testsets
    ]

    return train_loader, val_loader, test_loaders, dataset.nump