import torch
import torch.nn as nn


# class Normalize(nn.Module):

#     def __init__(self, mean, std):
#         super(Normalize, self).__init__()
#         self.mean = mean
#         self.std = std

#     def forward(self, input):
#         size = input.size()
#         x = input.clone()
#         for i in range(size[1]):
#             x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]

#         return x


# class Permute(nn.Module):

#     def __init__(self, permutation=[2, 1, 0]):
#         super().__init__()
#         self.permutation = permutation

#     def forward(self, input):
#         return input[:, self.permutation]


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        # mean和std是长度为embed_dim的列表或tensor
        self.mean = torch.tensor(mean).view(1, 1, -1)  # (1, 1, embed_dim)
        self.std = torch.tensor(std).view(1, 1, -1)

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


class Permute(nn.Module):
    def __init__(self, permutation=(0, 2, 1)):
        super(Permute, self).__init__()
        self.permutation = permutation

    def forward(self, x):
        # x: 任意形状tensor
        return x.permute(self.permutation)