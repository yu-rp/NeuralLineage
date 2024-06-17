import torch, numpy, math

from similarity.norm import *
from similarity.approx import *

from models.basic import SubFcNet

from utils import measure_time_memory

@measure_time_memory
def compare(
    model1 = None, model2 = None, 
    feature1 = None, feature2 = None, 
    data = None, 
    device = "cpu", 
    args = None):

    method = args.method
    num_batch = args.num_batch
    method_paras = method.split("_") # method space; method name; feature index; approximation or not
    method_space = method_paras[0]
    method_name = method_paras[1]

    if model2 is not None:
        model2.cuda()
    if feature2 is not None:
        feature2 = feature2.cuda()

    if method_space  == "representation" and "approx" in method_name:

        if feature1 is None:
            model1.eval()
            feature1 = get_feature_with_grad(model1, data, device, num_batch = num_batch)

        if feature2 is None:
            model2.eval()
            feature2 = get_feature(model2, data, device, num_batch = num_batch)

        if method_name == "approxl2":
            outcome, original = approxL2(feature1, feature2, model1, model2, grad_scale = args.grad_scale)
        elif method_name == "approxl1":
            outcome, original = approxL1(feature1, feature2, model1, model2, grad_scale = args.grad_scale)
        elif method_name == "approxlp":
            outcome, original = approxLp(feature1, feature2, model1, model2, grad_scale = args.grad_scale, p = args.p)
        elif method_name == "approxlse":
            outcome, original = approxLogSumExp(feature1, feature2, model1, model2, grad_scale = args.grad_scale, t = args.t)
        else:
            NotImplementedError

    elif method_space  == "representation":

        if feature1 is None:
            model1.eval()
            feature1 = get_feature(model1, data, device,  num_batch = num_batch)

        if feature2 is None:
            model2.eval()
            feature2 = get_feature(model2, data, device, num_batch = num_batch)

        if method_name == "l2":
            outcome = L2(feature1, feature2)
        elif method_name == "l1":
            outcome = L1(feature1, feature2)
        elif method_name == "li":
            outcome = Linfinity(feature1, feature2)
        else:
            NotImplementedError
        outcome = outcome
        original = 0.0

    else:
        NotImplementedError
    
    if method_space  == "representation" and "l2" in method_name:
        outcome = -numpy.sqrt(-outcome)
        original = -numpy.sqrt(-original)
    if method_space  == "representation" and "lp" in method_name:
        outcome = -numpy.power(-outcome, 1 / args.p)
        original = -numpy.power(-original, 1 / args.p)

    if math.isnan(original) or math.isinf(original):
        original = 1.0001

    if math.isnan(outcome) or math.isinf(outcome):
        outcome = original

    args.logger.info(f"The {method} similarity is {outcome:.4f}")
    args.logger.info(f"The {method}'s original similarity is {original:.4f}")

    return outcome, original, feature1, feature2

def parameter_distance(model1, model2, method):
    distances = []
    for param1, param2 in zip(list(model1.parameters())[:-2], list(model2.parameters())[:-2]):
        assert param1.shape == param2.shape
        diff = param1 - param2
        norm = (diff ** 2).mean()
        distances.append(norm.item())
    if method == "mean":
        return -numpy.mean(distances)
    elif method == "max":
        return -numpy.max(distances)
    
def get_feature(model, data, device, num_batch = 2):
    # feature_index is i, then the output of the i-th layer (non-linear) is used
    model.eval()
    model.to(device)

    output_feature = None

    with torch.no_grad():
        output_feature = None
        for e, (x,_) in enumerate(data):
            x = x.to(device)
            feature_batch = model(x)
            if output_feature is None:
                output_feature = feature_batch
            else:
                output_feature = torch.cat((output_feature, feature_batch), dim =0)
            if e == num_batch - 1:
                break
    del x
    del data
    del feature_batch
    del model
    torch.cuda.empty_cache()
    
    return output_feature

def get_feature_with_grad(model, data, device, num_batch = 2):
    # feature_index is i, then the output of the i-th layer (non-linear) is used
    model.eval()
    model.to(device)

    output_feature = None
    for e, (x,_) in enumerate(data):
        x = x.to(device)
        feature_batch = model(x)
        if output_feature is None:
            output_feature = feature_batch
        else:
            output_feature = torch.cat((output_feature, feature_batch), dim =0)
        if e == num_batch - 1:
            break
    del x
    del data
    torch.cuda.empty_cache()
    
    return output_feature