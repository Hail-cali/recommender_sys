import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.n_features =  n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.layer_1 = nn.Linear(in_features=self.n_features, out_features=self.hidden_dim)
        self.layer_2 = nn.Linear(in_features=self.hidden_dim, out_features=self.embedding_dim)



    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        x, (_,_) = self.layer_1(x)

        x, (hidden_n, _) = self.layer_2(x)

        return hidden_n.reshape((self.n_features, self.embedding_dim))


class Decoder(nn.Module):

    def __init__(self, input_dim=64, n_features=1):
        super(Decoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.layer_1 = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim)
        self.layer_2 = nn.Linear(in_features=self.hidden_dim, out_features=self.n_features)



        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.layer_1(x)
        x, (hidden_n, cell_n) = self.layer_2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))

        return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):

    def __init__(self,  n_features, embedding_dim=64, device='cpu'):
        super(RecurrentAutoencoder, self).__init__()

        self.encoder = Encoder(n_features, embedding_dim).to(device)
        self.decoder = Decoder(embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
