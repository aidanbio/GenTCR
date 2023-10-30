import unittest
import math
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn.utils import weight_norm

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
            (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


def get_activation_fn(name: str):
    if name == 'gelu':
        return gelu
    elif name == 'relu':
        return torch.nn.functional.relu
    elif name == 'swish':
        return swish
    else:
        raise ValueError(f"Unrecognized activation fn: {name}")


class SimpleMLP(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None))

    def forward(self, x):
        return self.main(x)

class SimpleConv(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm1d(in_dim),  # Added this
            weight_norm(nn.Conv1d(in_dim, hid_dim, 5, padding=2), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Conv1d(hid_dim, out_dim, 3, padding=1), dim=None))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.main(x)
        x = x.transpose(1, 2).contiguous()
        return x


class PredictionHeadTransform(nn.Module):
    def __init__(self,
                 hidden_size,
                 hidden_act='gelu',
                 layer_norm_eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if isinstance(hidden_act, str):
            self.transform_act_fn = get_activation_fn(hidden_act)
        else:
            self.transform_act_fn = hidden_act
        self.layer_norm = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states



class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
