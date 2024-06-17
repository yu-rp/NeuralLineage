import logging, os, sys, gc, time, re
from datetime import datetime
import torch, random, numpy
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset

from models.basic import *

def get_timestamp():
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_time

def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    print('Log directory: ', log_dir)
    return logger, formatter


def get_loader(args, shuffle = True, use_cuda = True):
    train_kwargs = {'batch_size': args.batch_size, 'num_workers': 4}
    test_kwargs = {'batch_size': args.test_batch_size, 'num_workers': 4}

    if shuffle:
        train_kwargs.update({'shuffle': True})
        test_kwargs.update({'shuffle': False})
    else:
        train_kwargs.update({'shuffle': False})
        test_kwargs.update({'shuffle': False})

    if args.dataset == "MNIST":
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
        test_dataset = datasets.MNIST('../data', train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 10
        num_channels = 1
    elif args.dataset == "EMNIST-Letters":
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1724,), (0.3311,))
            ])
        train_dataset = datasets.EMNIST('../data', split = "letters", train=True, download=True,
                        transform=transform)
        test_dataset = datasets.EMNIST('../data', split = "letters", train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 37
        num_channels = 1
    elif args.dataset == "EMNIST-Balanced":
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1753,), (0.3334,))
            ])
        train_dataset = datasets.EMNIST('../data', split = "balanced",  train=True, download=True,
                        transform=transform)
        test_dataset = datasets.EMNIST('../data', split = "balanced",  train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 47
        num_channels = 1
    elif args.dataset == "FMNIST":
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
            ])
        train_dataset = datasets.FashionMNIST('../data', train=True, download=True,
                        transform=transform)
        test_dataset = datasets.FashionMNIST('../data', train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 10
        num_channels = 1
    else:
        raise NotImplementedError

    return train_loader, test_loader, num_classes, num_channels

def get_model(args, mode = "P"):
    assert mode in ["P","C"]
    if args.model == "FC":
        model = FcNet()
        if args.finetune_flag and mode == "P":
            model.load_state_dict(torch.load(args.pre_train_ckpt, map_location= "cpu"))
        model = Fc_change_head(model, args.num_classes)
        if args.finetune_flag and mode == "C":
            model.load_state_dict(torch.load(args.pre_train_ckpt, map_location= "cpu"))
    else:
        raise NotImplementedError

    return model

def get_submodel(model, args):
    if args.feature_index == -1:
        return model
    else:
        if args.model == "FC":
            return SubFcNet(model, args.feature_index)
        else:
            raise NotImplementedError

def randomness_control(seed):
    print("seed",seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_matrix(matrix, path):

    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap='inferno')
    fig.colorbar(cax)
    fig.savefig(path)

def get_filename(path):
    base_name = os.path.basename(path)  # 获取文件名和扩展名: filename.extension
    file_name_without_extension = os.path.splitext(base_name)[0]
    return file_name_without_extension

def get_num_class_from_name(path):
    if "/FMNIST/" in path:
        return 10
    elif "/EMNIST-Letters/" in path:
        return 37
    elif "/EMNIST-Balanced/" in path:
        return 47
    elif "/MNIST/" in path:
        return 10
    else:
        raise NotImplementedError

def measure_time_memory(f):
    def wrapped(*args, **kwargs):
        if torch.cuda.is_available():
            start_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_max_memory_allocated()
        else:
            start_memory = 0

        start_time = time.time()

        result = f(*args, **kwargs)

        end_time = time.time()

        if torch.cuda.is_available():
            end_memory = torch.cuda.max_memory_allocated()
        else:
            end_memory = 0

        print(f"Function {f.__name__} executed in {end_time - start_time:.4f} seconds.")
        print(f"Memory usage increased by {(end_memory - start_memory) / (1024 ** 2):.2f} MB to {(end_memory) / (1024 ** 2):.2f} MB.")
        
        return result
    return wrapped

def get_real_parents(listp, listc):
    listp = [p.rsplit(".",1)[0] for p in listp]
    true_label = []
    for c in listc:
        flag = True
        for index, p in enumerate(listp):
            if p in c:
                true_label.append(index)
                flag = False
                break
        if flag:
            raise RuntimeError("Not True Label")
    return torch.tensor(true_label).long()

def get_accuracy(probability, label):
    prediction = probability.max(dim = 0)[1]
    return ((prediction == label).sum() / label.numel()).item()