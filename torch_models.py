from torch import nn
from torch.nn import (
    ReLU,
    BatchNorm1d,
    Linear,
    Tanhshrink,
    Softshrink,
    SiLU,
    Hardtanh,
    LeakyReLU,
    Sigmoid,
)


class ClassicAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = self._make_layer(encoder)
        self.decoder = self._make_layer(decoder)

    def _make_layer(self, layers):
        sequential = nn.Sequential()
        for layer in layers:
            for item in layer:
                sequential.append(globals()[item[0]](*item[1]))
        return sequential

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class NNClassifier(nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.layers = self._make_layer(nn)

    def _make_layer(self, layers):
        sequential = nn.Sequential()
        for layer in layers:
            for item in layer:
                sequential.append(globals()[item[0]](*item[1]))
        return sequential

    def forward(self, x):
        return self.layers(x)
