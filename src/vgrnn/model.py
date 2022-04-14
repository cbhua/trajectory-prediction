from statistics import variance
from turtle import forward
from matplotlib.pyplot import scatter
from sympy import false
import torch
from zmq import device
from src.vgrnn.message_passing import MessagePassing
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_scatter import scatter_add
from src.vgrnn.utils import *


class GCNConv(MessagePassing):
    def __init__(self, in_dim, out_dim, device, act=F.relu, bias=False):
        super(GCNConv, self).__init__(device)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act = act
        self.device = device
        self.weight = Parameter(torch.Tensor(in_dim, out_dim).to(device))
        if bias:
            self.bias = Parameter(torch.Tensor(out_dim).to(device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_idx, edge_weight=None):
        # Initialize edge weights
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_idx.size(1), ), dtype=x.dtype, device=self.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_idx.size(1)
        # Add self-loop edge weights to edge_weight
        edge_idx, _ = add_self_loops(edge_idx, num_nodes=x.size(0))
        loop_weight = torch.full(
            (x.size(0), ), 1, dtype=x.dtype, device=self.device)
        edge_weight = torch.cat(
            [edge_weight, loop_weight], dim=0).to(self.device)
        # symmetrically normalized edge weights
        row, col = edge_idx     # start in row and end in col
        deg = scatter_add(edge_weight, row.to(
            self.device), dim=0, dim_size=x.size(0))
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]

        x = torch.matmul(x, self.weight)
        out = self.propagate('add', edge_idx, x=x, norm=norm)
        return self.act(out)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self) -> str:
        return '{}({}, {})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels)


class GRU(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, device, bias=True) -> None:
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.device = device
        # GRU weights
        self.weight_xz = [
            GCNConv(in_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_xr = [
            GCNConv(in_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_xh = [
            GCNConv(in_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_hz = [
            GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_hr = [
            GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_hh = [
            GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        for i in range(1, self.n_layer):
            self.weight_xz.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_xr.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_xh.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_hz.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_hr.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_hh.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))

    def forward(self, x, edge_idx, h, c):
        h_out = torch.zeros_like(h)
        for i in range(self.n_layer):
            # get the Update gate(X) and Reset gate(R)
            z_g = torch.sigmoid(
                self.weight_xz[i](x if i == 0 else h_out[i-1], edge_idx) +
                self.weight_hz[i](h[i], edge_idx))
            r_g = torch.sigmoid(
                self.weight_xr[i](x if i == 0 else h_out[i-1], edge_idx) +
                self.weight_hr[i](h[i], edge_idx))
            # get candidate hidden state
            h_tilde_g = torch.tanh(
                self.weight_xh[i](x if i == 0 else h_out[i-1], edge_idx) +
                self.weight_hh[i](r_g * h[i], edge_idx))
            # get new hidden state
            h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g

        return h_out, c

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, device, bias=True):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.device = device
        self.weight_xi = [
            GCNConv(in_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_xf = [
            GCNConv(in_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_xo = [
            GCNConv(in_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_xc = [
            GCNConv(in_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_hi = [
            GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_hf = [
            GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_ho = [
            GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        self.weight_hc = [
            GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias)]
        for i in range(1, self.n_layer):
            self.weight_xi.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_xf.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_xo.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_xc.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_hi.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_hf.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_ho.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))
            self.weight_hc.append(
                GCNConv(hidden_dim, hidden_dim, device, act=lambda x: x, bias=bias))

    def forward(self, x, edge_idx, h, c):
        h_out = torch.zeros_like(h)
        c_out = torch.zeros_like(c)
        for i in range(self.n_layer):
            I_g = torch.sigmoid(self.weight_xi[i](
                x if i == 0 else h_out[i-1], edge_idx) + self.weight_hi[i](h[i], edge_idx))
            F_g = torch.sigmoid(self.weight_xf[i](
                x if i == 0 else h_out[i-1], edge_idx) + self.weight_hf[i](h[i], edge_idx))
            O_g = torch.sigmoid(self.weight_xo[i](
                x if i == 0 else h_out[i-1], edge_idx) + self.weight_ho[i](h[i], edge_idx))
            c_tilde_g = torch.tanh(self.weight_xc[i](
                x if i == 0 else h_out[i-1], edge_idx) + self.weight_hc[i](h[i], edge_idx))
            c_out[i] = F_g * h[i] + I_g * c_tilde_g
            h_out[i] = O_g * torch.tanh(c[i])
        return h_out, c_out


class VGRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layer, device, bias=False):
        super(VGRNN, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layer = n_layer
        self.device = device

        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim), nn.ReLU()).to(device)
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim), nn.ReLU()).to(device)

        self.enc = GCNConv(2 * h_dim, h_dim, device)
        self.enc_mu = GCNConv(h_dim, z_dim, device, act=lambda x: x)
        self.enc_logvar = GCNConv(h_dim, z_dim, device, act=F.softplus)

        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.ReLU()).to(device)
        self.prior_mu = nn.Sequential(nn.Linear(h_dim, z_dim)).to(device)
        self.prior_logvar = nn.Sequential(
            nn.Linear(h_dim, z_dim), nn.Softplus()).to(device)
        # self.gru = GRU(2 * h_dim, h_dim, n_layer, device, bias)
        self.gru = LSTM(2 * h_dim, h_dim, n_layer, device, bias)

    def forward(self, x, edge_idx_list, hidden_in=None):
        '''
        Returns:
            kld_loss: <torch.tensor> single value; 
            nll_loss: <torch.tensor> single value; 
            all_enc_mu: <list> [6 * <torch.tensor> [#node, z_dim]];
            h: <torch.tensor> [1, #node, h_dim].
        '''
        # assert len(adj_orig_dense_list) == len(edge_idx_list)   # Snapshots
        kld_loss = 0
        nll_loss = 0
        all_enc_mu, all_enc_logvar = [], []
        all_prior_mu, all_prior_logvar = [], []
        all_dec_t, all_z_t = [], []
        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layer, x.size(1),
                         self.h_dim, device=self.device))
            c = Variable(torch.zeros(self.n_layer, x.size(1),
                         self.h_dim, device=self.device))
        else:
            h = Variable(hidden_in).to(self.device)
            c = Variable(hidden_in).to(self.device)
        # for t in range(x.size(0)): # t=0->6 go through all steps
        for t in range(len(x)): # t=0->6 go through all steps
            phi_x_t = self.phi_x(x[t]) # [663, 663] -> [663, 32]

            # Encoder
            enc_t = self.enc(
                torch.cat([phi_x_t, h[-1]], 1).to(self.device), edge_idx_list[t]) # ([663, 32 + 32], [2, 718]) -> [663, 32]
            enc_mu_t = self.enc_mu(enc_t, edge_idx_list[t]) # [663, 32], [2, 718] -> [663, 16]
            enc_logvar_t = self.enc_logvar(enc_t, edge_idx_list[t]) # [663, 32], [2, 718] -> [663, 16]

            # Prior
            prior_t = self.prior(h[-1]) # [663, 32] -> [663, 32]
            prior_mu_t = self.prior_mu(prior_t) # [663, 32] -> [663, 16]
            prior_logvar_t = self.prior_logvar(prior_t) # [663, 32] -> [663, 16]

            # Reparameterization
            z_t = self._reparameterized(enc_mu_t, enc_logvar_t) # -> [663, 16]
            phi_z_t = self.phi_z(z_t) # -> [663, 32]
            # Decode from z_t
            dec_t = self.decoder(z_t) # -> [663, 663]
            
            # Recurrence
            h, c = self.gru(
                torch.cat([phi_x_t, phi_z_t], 1).to(self.device),
                edge_idx_list[t], h, c)
                
            # Question: Is n_nodes dynamic? Yes
            n_nodes = x[t].size()[0]
            enc_mu_t = enc_mu_t[:n_nodes, :]
            enc_logvar_t = enc_logvar_t[:n_nodes, :]
            prior_mu_t = prior_mu_t[:n_nodes, :]
            prior_logvar_t = prior_logvar_t[:n_nodes, :]
            dec_t = dec_t[:n_nodes, :n_nodes]
            
            # Calculate and accumulate the KL-Divergence and Binary Cross-entropy
            # kld_loss = kld_loss + \
            #     self._kld_gauss(enc_mu_t, enc_logvar_t,
            #                     prior_mu_t, prior_logvar_t)
            # nll_loss = nll_loss + \
            #     self._nll_bernoulli(dec_t, adj_orig_dense_list[t])

            # Save the parameters learned at time t
            all_enc_mu.append(enc_mu_t)
            all_enc_logvar.append(enc_logvar_t)
            all_prior_mu.append(prior_mu_t)
            all_prior_logvar.append(prior_logvar_t)
            all_dec_t.append(dec_t)
            all_z_t.append(z_t)
        # return kld_loss, nll_loss, all_enc_mu, all_prior_mu, h, dec_t
        return all_enc_mu, all_prior_mu, h, dec_t

    def decoder(self, z):
        outputs = InnerProductDecoder(act=lambda x: x)(z)
        return outputs

    def _reparameterized(self, mu, logvar):
        epsilon = Variable(torch.FloatTensor(
            logvar.size()).normal_()).to(self.device)
        return epsilon.mul(logvar).add_(mu)

    def _kld_gauss(self, mu_1, logvar_1, mu_2, logvar_2):
        num_nodes = mu_1.size()[0]
        # kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
        #                (torch.pow(std_1, 2) + torch.pow(mean_1 - mean_2, 2)) /
        #                torch.pow(std_2, 2) - 1)
        kld_element = 2 * (logvar_2 - logvar_1) + (torch.exp(2*logvar_1) +
                                                   torch.pow(mu_1-mu_2, 2)) / torch.exp(2*logvar_2)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)

    def _nll_bernoulli(self, logits, target_adj_dense):
        temp_size = target_adj_dense.size()[0]
        temp_sum = target_adj_dense.sum()
        # Negtive Edges / Positive Edges, positive weight
        posw = float(temp_size * temp_size - temp_sum) / temp_sum
        norm = temp_size * temp_size / \
            float((temp_size * temp_size - temp_sum) * 2)
        nll_loss_mat = F.binary_cross_entropy_with_logits(
            input=logits, target=target_adj_dense.to(self.device), pos_weight=posw, reduction='none')
        nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0, 1])
        return -nll_loss


class InnerProductDecoder(nn.Module):
    def __init__(self, act=torch.sigmoid, dropout=0.5, training=True):
        super(InnerProductDecoder, self).__init__()
        self.act = act
        self.dropout = dropout
        self.training = training

    def forward(self, inp):
        inp = F.dropout(inp, self.dropout, training=self.training)
        x = torch.transpose(inp, dim0=0, dim1=1)
        x = torch.mm(inp, x)
        return self.act(x)
