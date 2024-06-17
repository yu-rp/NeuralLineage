import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelWithClsToken(nn.Module):
    def __init__(self, embed_dim, type_vocab_size = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.ones(1, 1, embed_dim)*0.01)
        self.type_embeddings = nn.Embedding(type_vocab_size, embed_dim)

    def forward(self, x1, x2):
        #x1 NxL1xE, x2 NxL2xE
        batch_size = x1.size(0)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # Nx1xE

        type1_idx = torch.zeros(x1.size(0), dtype=torch.long, device=x1.device)# Nx1
        type2_idx = torch.ones(x2.size(0), dtype=torch.long, device=x2.device)# Nx1
        
        type1_embedding = self.type_embeddings(type1_idx)# NxE
        type2_embedding = self.type_embeddings(type2_idx)# NxE
        
        x1_with_type = x1 + type1_embedding.unsqueeze(1)# Nx1xE
        x2_with_type = x2 + type2_embedding.unsqueeze(1)# Nx1xE
        
        concatenated = torch.cat((x1_with_type, x2_with_type), dim=1)
        concatenated = torch.cat((cls_tokens, concatenated), dim=1)
        return concatenated

class Encoder(nn.Module):
    def __init__(self,embed_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(2, 3, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(3, embed_dim, kernel_size=3, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(embed_dim)  
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def initialize_conv1(self):
        self.conv1.weight.data[0,0,0,0] = 1
        self.conv1.weight.data[0,1,0,0] = -1
        self.conv1.weight.data[1,0,0,0] = -1
        self.conv1.weight.data[1,1,0,0] = 1

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = F.relu(self.bn2(self.conv2(x))) 
        x = self.pool(x)  
        return x

class LineageDetector(nn.Module):
    def __init__(self, embed_dim, nump, noparent = False):
        super().__init__()
        self.weightEnc = Encoder(embed_dim) 
        self.featureEnc = Encoder(embed_dim) 
        self.tokenization = ModelWithClsToken(embed_dim)
        self.attention = torch.nn.MultiheadAttention(embed_dim, 1, batch_first=True)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.nump = nump
        self.embed_dim = embed_dim
        self.noparent = noparent
        if noparent:
            self.noparent_token = torch.nn.Parameter(
                torch.randn(1,1,embed_dim, requires_grad = True)
            )

    def forward(self, weight, feature):
        # Nxnumpx2xwxh
        weightshape = weight.shape
        assert self.nump == weightshape[1]
        weight = weight.view(weightshape[0]*weightshape[1], *weightshape[2:]) # (Nxnump)x2xwxh
        featureshape = feature.shape
        assert self.nump == featureshape[1]
        feature = feature.view(featureshape[0]*featureshape[1], *featureshape[2:]) # (Nxnump)x2xwxh

        weight = self.weightEnc(weight) # (Nxnump)xE
        feature = self.featureEnc(feature) # (Nxnump)xE

        weight = weight.squeeze(-1).squeeze(-1)
        feature = feature.squeeze(-1).squeeze(-1)

        weight = weight.unsqueeze(1) # (Nxnump)x1xE
        feature = feature.unsqueeze(1) # (Nxnump)x1xE

        concatenated = self.tokenization(weight, feature) # (Nxnump)x3xE
        attn_output,_ = self.attention(concatenated,concatenated,concatenated,need_weights=False) # (Nxnump)x3xE
        attn_output = attn_output[:, 0, :] # (Nxnump)xE

        attn_output = attn_output.view(weightshape[0], self.nump, attn_output.shape[-1])

        attn_mean = attn_output.mean(dim = (1,2),keepdim = True)
        attn_std = attn_output.std(dim = (1,2),keepdim = True)
        attn_output = (attn_output - attn_mean)/attn_std

        if self.noparent:
            noparent_token = self.noparent_token.expand(weightshape[0],-1,-1)
            attn_output = torch.cat((attn_output, noparent_token), dim = 1)

        output = self.fc(attn_output)
        return output.squeeze(-1)


def get_model(args):
    if args.model == "default":
        return LineageDetector(args.embed_dim, args.nump, args.no_parent)
    else:
        NotImplementedError
        




