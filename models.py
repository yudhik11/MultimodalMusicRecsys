import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

features = 150

class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()
        self.enc1 = nn.Linear(in_features=11, out_features=128)
        self.enc2 = nn.Linear(in_features=128, out_features=features*2)
        self.dec1 = nn.Linear(in_features=features, out_features=128)
        self.dec2 = nn.Linear(in_features=128, out_features=11)
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x, f=0):
        # encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, features)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        if f == 1:
            return z
        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction, mu, log_var

class LinearVAE2(nn.Module):
    def __init__(self):
        super(LinearVAE2, self).__init__()
 
        # encoder
        self.enc1 = nn.Linear(in_features=300, out_features=64)
        self.enc2 = nn.Linear(in_features=64, out_features=features*2)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=features, out_features=64)
        self.dec2 = nn.Linear(in_features=64, out_features=300)
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x, f=0):
        # encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, features)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        if f == 1:
            return z
        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction, mu, log_var

def return_models(weights_matrix, weights_matrix1, weights_matrix2, weights_matrix3):
    m = nn.LogSoftmax(dim=1)
    class Encoder(nn.Module):
        def __init__(self,
                     input_dim: int,
                     emb_dim: int,
                     feat_dim, 
                     enc_hid_dim: int,
                     dec_hid_dim: int,
                     dropout: float):
            super().__init__()
            self.input_dim = input_dim
            self.emb_dim = emb_dim
            self.enc_hid_dim = enc_hid_dim
            self.dec_hid_dim = dec_hid_dim
            self.dropout = dropout
            self.evolve = nn.Linear(feat_dim, emb_dim)
            self.embedding = nn.Embedding(input_dim, emb_dim)
            self.embedding.load_state_dict({'weight': weights_matrix})
            self.embedding.weight.requires_grad = True
            self.rnn = nn.GRU(emb_dim, enc_hid_dim, num_layers = 1, bidirectional = True)
            self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self,
                    src: Tensor, lens, inds) -> Tuple[Tensor]:

            embedded = self.dropout(self.embedding(inds))
            embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lens, enforce_sorted = False)
            outputs, hidden = self.rnn(embedded)
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
            hidden = hidden[-2,:,:] + hidden[-1,:,:]
            outputs = (outputs[:, :, :self.enc_hid_dim] +
                       outputs[:, :, self.enc_hid_dim:])
            return outputs, hidden

    class Attention(nn.Module):
        def __init__(self,
                     enc_hid_dim: int,
                     dec_hid_dim: int,
                     attn_dim: int):
            super().__init__()

            self.enc_hid_dim = enc_hid_dim
            self.dec_hid_dim = dec_hid_dim
            self.attn_in = enc_hid_dim + dec_hid_dim
            self.attn = nn.Linear(self.attn_in, attn_dim)

        def forward(self,
                    decoder_hidden: Tensor,
                    encoder_outputs: Tensor) -> Tensor:

            src_len = encoder_outputs.shape[0]
            repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            energy = torch.tanh(self.attn(torch.cat((
                repeated_decoder_hidden,
                encoder_outputs),
                dim = 2)))
            attention = torch.sum(energy, dim=2)
            return m(attention)

    class Model(nn.Module):
        def __init__(self, encoder, attention, hidden_size, out_size):
            super(Model, self).__init__()
            self.encoder = encoder
            self.attention = attention
            self.embedding = nn.Embedding(encoder.input_dim, encoder.emb_dim)
            self.embedding.load_state_dict({'weight': weights_matrix2})
            self.embedding.weight.requires_grad = True
            self.embedding2 = nn.Embedding(encoder.input_dim, encoder.emb_dim)
            self.embedding2.load_state_dict({'weight': weights_matrix1})
            self.embedding2.weight.requires_grad = True
            self.embedding3 = nn.Embedding(encoder.input_dim, encoder.emb_dim)
            self.embedding3.load_state_dict({'weight': weights_matrix3})
            self.embedding3.weight.requires_grad = True
            self.feats = nn.Linear(encoder.emb_dim, hidden_size)
            self.out1 = nn.Linear(hidden_size * 4, 300)
            self.out2 = nn.Linear(300, out_size)
            self.relu = nn.LeakyReLU(0.18)

        def forward(self, src, lens, inds):
            batch_size = src.size(1)
            encoder_outputs, hidden = self.encoder(src, lens, inds)
            attn_weights = self.attention(hidden, encoder_outputs).unsqueeze(1)
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
            context1 = context.transpose(0, 1)[0]
            query = self.embedding(inds)
            context = attn_weights.bmm(query.transpose(0, 1))  # (B,1,N)
            context2 = context.transpose(0, 1)[0]
            query2 = self.embedding2(inds)
            context = attn_weights.bmm(query2.transpose(0, 1))  # (B,1,N)
            context3 = context.transpose(0, 1)[0]
            query3 = self.embedding3(inds)
            context = attn_weights.bmm(query3.transpose(0, 1))  # (B,1,N)
            context4 = context.transpose(0, 1)[0]
            temp = torch.cat((context1, context2, context3, context4), 1)
            out = self.relu(self.out1(temp))
            out = self.out2(out)
            return m(out)
    return Encoder, Attention, Model