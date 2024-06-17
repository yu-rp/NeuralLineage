import argparse, json, os, warnings
import torch
from torchvision import datasets, transforms

from utils import *
from models.basic import *
from models.resnets import *
from compare import compare

warnings.simplefilter(action='ignore', category=UserWarning)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Model Lineage')
    parser.add_argument('--expgroup', type=str, default="")
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=2, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--model', type=str, default="FC", 
                        help='network structure')
    parser.add_argument('--dataset', type=str, default="MNIST", 
                        help='dataset')
    parser.add_argument('--datasplit', type=str, default="train", 
                        help='train or test')
    parser.add_argument('--pre-train-ckpt', type=str, default="", 
                        help='path of the pretrained model')
    parser.add_argument('--pre-train-ckpt-json-p', type=str, default="", 
                        help='path of the pretrained model')
    parser.add_argument('--pre-train-ckpt-json-c', type=str, default="", 
                        help='path of the pretrained model')
    parser.add_argument('--method', type = str, default="representation_cka", 
                        help='method to compute the similarity')
    parser.add_argument('--feature_index', type = int, default=2, 
                        help='feature to compute the similarity')
    parser.add_argument('--num_batch', type = int, default=1, 
                        help='number of batch used to calculate the similarity')
    parser.add_argument('--kfold_split', type = int, default=5, 
                        help='number of folds used to validate')
    parser.add_argument('--grad_scale', type = float, default=1.0, 
                        help='weight before gradient')
    parser.add_argument('--p', type = int, default=4, 
                        help='pnorm')
    parser.add_argument('--t', type = float, default=0.01, 
                        help='t log sum exp')
    args = parser.parse_args()

    exp_id = f"../results/{args.dataset}_{args.datasplit}_{args.model}"\
        f"/{get_filename(args.pre_train_ckpt_json_p)}_{get_filename(args.pre_train_ckpt_json_c)}"
    os.makedirs(exp_id, exist_ok = True)
    file_id = f"{args.method}_{args.grad_scale}_{args.feature_index}"
    logger, formatter = get_logger(exp_id, None, f"{file_id}.log", level=logging.INFO)
    args.logger = logger

    args.finetune_flag = True

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    randomness_control(args.seed)

    json_file_p = open(args.pre_train_ckpt_json_p, "r")
    pre_train_ckpt_paths_p = json.load(json_file_p)
    num_models_p = len(pre_train_ckpt_paths_p)
    storage_p = {}

    json_file_c = open(args.pre_train_ckpt_json_c, "r")
    pre_train_ckpt_paths_c = json.load(json_file_c)
    num_models_c = len(pre_train_ckpt_paths_c)
    storage_c = {}

    train_data, test_data, num_classes, num_channels = get_loader(args, shuffle = False, use_cuda = use_cuda)
    if args.datasplit == "train":
        data = train_data
    else:
        data = test_data
    args.num_classes = num_classes
    args.num_channels = num_channels

    similarity_matrix = torch.zeros(num_models_p, num_models_c)

    for p_idx in range(num_models_p):

        c_start_index = 0 if num_models_p != num_models_c else p_idx+1

        for c_idx in range(c_start_index, num_models_c):
            logger.info(f"For row {p_idx}, column {c_idx}.")

            if p_idx not in storage_p.keys():

                args.pre_train_ckpt = pre_train_ckpt_paths_p[p_idx]

                modelp = get_model(args)
                modelp = get_submodel(modelp, args)

                featurep = None

            else:
                featurep, modelp = storage_p[p_idx]

            if c_idx not in storage_c.keys():
            
                args.pre_train_ckpt = pre_train_ckpt_paths_c[c_idx]

                modelc = get_model(args,"C")
                modelc = get_submodel(modelc, args)

                featurec = None
            
            else:
                featurec, modelc = storage_c[c_idx]

            (similarity_matrix[p_idx, c_idx], _,
                featurep,featurec) = compare(
                    modelp, modelc, 
                    feature1 = featurep, feature2 = featurec,
                    data = data,
                    device = device,  
                    args = args
                    )

            else:
                raise NotImplementedError

        if modelp is not None:
            modelp.cpu()
            del modelp
        if featurep is not None:
            featurep.cpu()
            del featurep
        del storage_p[p_idx]
        torch.cuda.empty_cache()

    label = get_real_parents(pre_train_ckpt_paths_p, pre_train_ckpt_paths_c)
    acc =  get_accuracy(similarity_matrix, label)
    logger.info(f"acc = {acc*100:.5f}%")

if __name__ == '__main__':
    main()