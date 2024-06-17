import torch

def L2(feature1, feature2):

    feature1, feature2 = feature1.cuda(), feature2.cuda()

    distance = torch.mean((feature1 - feature2)**2)
    similarity = - distance
    return similarity.item()

def L1(feature1, feature2):

    feature1, feature2 = feature1.cuda(), feature2.cuda()

    distance = torch.mean((feature1 - feature2).abs())
    similarity = - distance
    return similarity.item()

def Linfinity(feature1, feature2):

    feature1, feature2 = feature1.cuda().flatten(1), feature2.cuda().flatten(1)


    distance = torch.mean((feature1 - feature2).abs().max(dim = -1)[0])
    similarity = - distance
    return similarity.item()

def Lp(feature1, feature2, p = 4):

    feature1, feature2 = feature1.cuda(), feature2.cuda()

    distance = torch.mean((feature1 - feature2).abs()**p)
    similarity = - distance
    return similarity.item()

def LogSumExp(feature1, feature2, t = 0.02):

    feature1, feature2 = feature1.cuda().flatten(1), feature2.cuda().flatten(1)

    diff = t * (feature1 - feature2).abs()
    diff = diff.clamp(max = 10)
    
    exp = torch.exp(diff)
    sumexp = torch.sum(exp, dim = 1)
    logsumexp = torch.log(sumexp) / t
    distance = torch.sum(logsumexp) / feature1.numel()
    similarity = - distance
    return similarity.item()