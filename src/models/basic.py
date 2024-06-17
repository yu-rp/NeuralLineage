import torch
import torch.nn as nn
import torch.nn.functional as F

class FcNet(nn.Module):
    def __init__(self):
        super(FcNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        # output = F.log_softmax(x, dim=1)
        output = x
        return output

def Fc_change_head(model, num_cls):
    if num_cls > 1:
        in_features, out_features = model.fc4.in_features, model.fc4.out_features
        if out_features != num_cls:
            model.fc4 = nn.Linear(in_features, num_cls)
            print(f"fc4 changed {in_features}x{out_features} -> {in_features}x{num_cls}")
    return model

class SubFcNet(nn.Module):
    def __init__(self, model, num_layer):
        super(SubFcNet, self).__init__()
        params_list = [
            ("fc1", 28*28, 1024),("fc2", 1024, 256),("fc3", 256, 128),("fc4", 128, 10)
        ]
        self.num_layer = num_layer

        self.model = nn.Sequential()
        for param in params_list[:num_layer]:
            self.model.add_module(param[0], nn.Linear(*param[1:]))

        for new_weight, ori_weight in zip(
            self.model.parameters(),
            list(model.parameters())[:num_layer * 2]
        ):
            new_weight.data = ori_weight.data.clone()

    def forward(self, x):
        x = torch.flatten(x, 1)

        for layer in self.model[:-1]:
            x = layer(x)
            x = F.relu(x)

        x = self.model[-1](x)
        if self.num_layer == 4:
            # output = F.log_softmax(x, dim=1)
            output = x
        else:
            output = F.relu(x)
        return output
