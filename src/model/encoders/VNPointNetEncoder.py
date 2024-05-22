import torch.nn.parallel
import torch.utils.data

from src.model.encoders.vn_layers import *

"""
SOURCES
@article{deng2021vn,
  title={Vector Neurons: a general framework for SO(3)-equivariant networks},
  author={Deng, Congyue and Litany, Or and Duan, Yueqi and Poulenard, Adrien and Tagliasacchi, Andrea and Guibas, Leonidas},
  journal={arXiv preprint arXiv:2104.12229},
  year={2021}
} 

https://github.com/FlyingGiraffe/vnn/tree/master
"""
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature_cross(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)

    feature = torch.cat((feature - x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()

    return feature

class STNkd(nn.Module):
    def __init__(self, pooling, d=64):
        super(STNkd, self).__init__()
        self.pooling = pooling
        
        self.conv1 = VNLinearLeakyReLU(d, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128//3, 1024//3, dim=4, negative_slope=0.0)

        self.fc1 = VNLinearLeakyReLU(1024//3, 512//3, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(512//3, 256//3, dim=3, negative_slope=0.0)
        
        if self.pooling == 'max':
            self.pool = VNMaxPool(1024//3)
        elif self.pooling == 'mean':
            self.pool = mean_pool
        
        self.fc3 = VNLinear(256//3, d)
        self.d = d

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x


class VNPointNetEncoder(nn.Module):
    def __init__(self, n_knn=20, pooling="max", global_feat=True, feature_transform=False):
        super(VNPointNetEncoder, self).__init__()
        self.n_knn = n_knn ######
        self.pooling = pooling
        
        self.conv_pos = VNLinearLeakyReLU(3, 64//3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0)
        
        self.conv3 = VNLinear(128//3, 1024//3)
        self.bn3 = VNBatchNorm(1024//3, dim=4)
        
        self.std_feature = VNStdFeature(1024//3, dim=4, normalize_frame=False, negative_slope=0.0) # *2
        
        if self.pooling == 'max':
            self.pool = VNMaxPool(64//3)
        elif self.pooling == 'mean':
            self.pool = mean_pool
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        
        if self.feature_transform:
            self.fstn = STNkd(self.pooling, d=64//3)

    def forward(self, x):
        B, D, N = x.size()
        
        x = x.unsqueeze(1)
        feat = get_graph_feature_cross(x, k=self.n_knn)
        x = self.conv_pos(feat)
        x = self.pool(x)
        
        x = self.conv1(x)
        
        if self.feature_transform:
            x_global = self.fstn(x).unsqueeze(-1).repeat(1,1,1,N)
            x = torch.cat((x, x_global), 1)
        
        x = self.conv2(x)
        x = self.bn3(self.conv3(x))
        
        # x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        # x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        x = x.view(B, -1, N)
        
        x = torch.max(x, -1, keepdim=False)[0]
        
        return x



if __name__ == "__main__":
    batches = 2
    n_points = 64

    model = VNPointNetEncoder(feature_transform=False) # Make feat trans from work <- todo

    x = torch.rand((batches, n_points, 3))

    x = x.transpose(2, 1)
    x2 = x

    label = torch.zeros((batches, 1023))
    label[:, 0] = 1

    out= model.forward(x)
    print(out.shape)

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.zero_grad()

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(out, label)
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)

    optimizer.step()




    out2 = model.forward(x)
    print(out2.shape)

    # Rotational invariance (COOLEST)
    print(out.isclose(out2).all(), "same output?")
    print(x.isclose(x2).all(), "same pcd?")

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
