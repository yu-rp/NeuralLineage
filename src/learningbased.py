from __future__ import print_function
import nni, os, sys
import argparse
import warnings
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

sys.path.append('src')
os.chdir("src")
from utils import *
from lineagedetector.dataset import *
from lineagedetector.model import *

warnings.simplefilter(action='ignore', category=UserWarning)

def train(args, model, device, train_loader, optimizer, epoch, logger, start_index):
    model.train()
    for batch_idx, (feature, weight, label) in enumerate(train_loader, start_index):
        feature, weight, label = feature.to(device), weight.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(feature, weight)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx - start_index) * len(feature), len(train_loader.dataset),
                100. * (batch_idx - start_index) / len(train_loader), loss.item()))
            if args.dry_run:
                break
        if args.save_model_along and (batch_idx + 1) % args.save_model_interval == 0:
            torch.save(model.state_dict(), f"{args.exp_id}/{args.operation}_{batch_idx + 1}.pt")
            logger.info(f"model was saved to {args.exp_id}/{args.operation}_{batch_idx + 1}.pt")
    return model

def test(model, device, test_loader, logger, name):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for feature, weight, label in test_loader:
            feature, weight, label = feature.to(device), weight.to(device), label.to(device)
            output = model(feature, weight)
            test_loss += F.cross_entropy(output, label, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info("\t"+name+' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)

def get_predict(model, device, test_loader):
    model.eval()
    prediction = None
    with torch.no_grad():
        for feature, weight, label in test_loader:
            feature, weight, label = feature.to(device), weight.to(device), label.to(device)
            output = model(feature, weight)
            if prediction is None:
                prediction = output
                labels = label
            else:
                prediction = torch.cat((prediction, output), dim = 0)
                labels = torch.cat((labels, label), dim = 0)

    return prediction, labels

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--exp_group', type=str, default="")
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--model', type=str, default="default", 
                        help='network structure')
    parser.add_argument('--datasets', type=str, nargs = "+", default=["fc_MNIST_FMNIST", "fc_MNIST_EMNIST-Letters"], 
                        help='dataset')
    parser.add_argument('--pre_train_ckpt', type=str, default="", 
                        help='path of the pretrained model')
    parser.add_argument('--test-batch-size', type=int, default=2, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', # 100 MNIST pretrain, 5 Finetune
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7, 1.0 for fewshot)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1314, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--embed_dim', type=int, default=32, metavar='N',
                        help='how many hidden dims')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--save-model-interval', type = int, default=-1, # pretrain -1, finetune 100
                        help='wheter save model along training')
    parser.add_argument('--kfold_split', type = int, default=5, 
                        help='number of folds used to validate')
    parser.add_argument('--no_parent', action='store_true', default=False,
                        help='whether no parent')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    randomness_control(args.seed)

    args.evaluation_flag = len(args.pre_train_ckpt) > 0
    args.operation = "Evaluation" if args.evaluation_flag else "Train"

    args.save_model_along = args.save_model_interval > 0

    args.exp_id = "../results/"+"__".join(args.datasets)+f"/{args.model}_{args.embed_dim}_{args.batch_size}_{args.lr}_{args.seed}"
    os.makedirs(args.exp_id, exist_ok = True)

    logger, formatter = get_logger(args.exp_id, None, "log.log", level=logging.INFO)

    train_loader, val_loader, test_loaders, nump = get_detector_loader(args)

    args.nump = nump

    model = get_model(args)
    model = model.to(device)

    if args.evaluation_flag:
        model.load_state_dict(torch.load(args.pre_train_ckpt, map_location= "cpu"))
        with torch.no_grad():
            for name, loader in zip(
                    ["train","test"] + [f"test no. {i}" for i in range(len(test_loaders))],
                    [train_loader, val_loader] + test_loaders
                    ):
                prediction, label = get_predict(model, device, loader)
                acc =  get_accuracy(prediction.T, label)
                logger.info(f"evaluation acc is {acc*100:.2f}%")
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, logger = logger, start_index = (epoch - 1) *len(train_loader))
            train_acc = test(model, device, train_loader, logger = logger, name = "train")
            val_acc = test(model, device, val_loader, logger = logger, name = "val")
            test_accs = []
            for no,test_loader in enumerate(test_loaders):
                test_accs.append(test(model, device, test_loader, logger = logger, name = f"test np. {no}"))
            scheduler.step()
            if args.dry_run:
                break

        if args.save_model:
            torch.save(model.state_dict(), f"{args.exp_id}/{args.operation}.pt")
            logger.info(f"model was saved to {args.exp_id}/{args.operation}.pt")

        logger.info(f"training process was finished")

if __name__ == '__main__':
    main()
