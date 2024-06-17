import torch
from .norm import *

def mul_grad_params(grads, parameters1, parameters2):
    output = 0
    for grad, parameter1, parameter2 in zip(grads, parameters1, parameters2):
        if grad is not None:
            output += (grad * (parameter2 - parameter1)).sum()
    return output

def approxL2(feature1_grad, feature2_nograd, model1, model2, grad_scale):

    feature1_grad, feature2_nograd = feature1_grad.cuda(), feature2_nograd.cuda().detach()
    feature1_nograd = feature1_grad.detach()

    weight = - 2 * (feature1_nograd - feature2_nograd) / (feature1_nograd.numel())

    weight = weight * grad_scale

    param1 = [param for name, param in model1.named_parameters() if param.requires_grad]
    param2 = [param for name, param in model2.named_parameters() if param.requires_grad]

    grads = torch.autograd.grad(feature1_grad, param1, retain_graph = True, grad_outputs = weight, allow_unused=True)

    approximation_term = mul_grad_params(grads,parameters1 = param1, parameters2 = param2)
    original_term = L2(feature1_nograd, feature2_nograd)

    similarity = original_term + approximation_term
    similarity = torch.minimum(similarity, -torch.zeros(1).cuda())
    return similarity.item(), original_term

def approxL1(feature1_grad, feature2_nograd, model1, model2, grad_scale):

    feature1_grad, feature2_nograd = feature1_grad.cuda(), feature2_nograd.cuda().detach()
    feature1_nograd = feature1_grad.detach()

    weight = - torch.sign(feature1_nograd - feature2_nograd) / (feature1_nograd.numel())

    weight = weight * grad_scale

    param1 = [param for name, param in model1.named_parameters() if param.requires_grad]
    param2 = [param for name, param in model2.named_parameters() if param.requires_grad]

    grads = torch.autograd.grad(feature1_grad, param1, retain_graph = True, grad_outputs = weight, allow_unused=True)

    approximation_term = mul_grad_params(grads, parameters1 = param1, parameters2 = param2)
    original_term = L1(feature1_nograd, feature2_nograd)

    similarity = original_term + approximation_term
    return similarity.item(), original_term

def approxLp(feature1_grad, feature2_nograd, model1, model2, grad_scale, p = 4):

    feature1_grad, feature2_nograd = feature1_grad.cuda(), feature2_nograd.cuda().detach()
    feature1_nograd = feature1_grad.detach()

    weight = - p * torch.sign(feature1_nograd - feature2_nograd) * ((feature1_nograd - feature2_nograd).abs())**(p-1) / (feature1_nograd.numel())

    weight = weight * grad_scale

    param1 = [param for name, param in model1.named_parameters() if param.requires_grad]
    param2 = [param for name, param in model2.named_parameters() if param.requires_grad]

    grads = torch.autograd.grad(feature1_grad, param1, retain_graph = True, grad_outputs = weight, allow_unused=True)

    approximation_term = mul_grad_params(grads, parameters1 = param1, parameters2 = param2)
    original_term = Lp(feature1_nograd, feature2_nograd, p)

    similarity = original_term + approximation_term
    similarity = torch.minimum(similarity, -torch.zeros(1).cuda())
    return similarity.item(), original_term

def approxLogSumExp(feature1_grad, feature2_nograd, model1, model2, grad_scale, t = 0.02):

    feature1_grad, feature2_nograd = feature1_grad.cuda(), feature2_nograd.cuda().detach()
    feature1_nograd = feature1_grad.detach()

    weight = - torch.sign(feature1_nograd - feature2_nograd) * torch.nn.functional.softmax(t * (feature1_nograd - feature2_nograd).abs(), dim = 1) / (feature1_nograd.numel())

    weight = weight * grad_scale

    param1 = [param for name, param in model1.named_parameters() if param.requires_grad]
    param2 = [param for name, param in model2.named_parameters() if param.requires_grad]

    grads = torch.autograd.grad(feature1_grad, param1, retain_graph = True, grad_outputs = weight, allow_unused=True)

    approximation_term = mul_grad_params(grads, parameters1 = param1, parameters2 = param2)
    original_term = LogSumExp(feature1_nograd, feature2_nograd, t)

    similarity = original_term + approximation_term
    return similarity.item(), original_term