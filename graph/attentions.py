from .utils import *
from torch.nn import Parameter, Linear
from torch import nn
class GATv2Attention(nn.Module):
    def __init__(self, embed_dim, nhead, aggr,leaky_relu_negative_slope: float = 0.2,
     dropout=.1, mult_attr=0, bias=True, concat=True, share_weights=False) -> None:
        super().__init__()
        print("GATv2Attention")
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.aggr = aggr
        self.mult_attr = mult_attr

        if concat:
            assert embed_dim % nhead == 0
            self.n_hidden = embed_dim // nhead
        
        self.linear_l = nn.Linear(embed_dim, self.n_hidden * nhead, bias=False)
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(embed_dim, self.n_hidden * nhead, bias=False)
        
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor):
        n_nodes = h.shape[0]
        g_l = self.linear_l(h).view(n_nodes, self.nhead, self.n_hidden)
        g_r = self.linear_r(h).view(n_nodes, self.nhead, self.n_hidden)

        g_l_repeat = g_l.repeat(n_nodes, 1, 1)
        g_r_interleave = g_r.repeat_interleave(n_nodes, dim=0)
        
        g_sum = g_l_repeat + g_r_interleave
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)

        e = self.attn(self.activation(g_sum))
        e = e.squeeze(-1)
        r, c, e_shape = edge_index[:, 0], edge_index[:, 1], edge_index.shape[0]

        # e has to be masked to -ve infinity if there is no connection
        # but if there is a fully connected graph is this needed?
        e = e.masked_fill()





class MultiHeadDotProduct(nn.Module):
    """
    Multi head attention like in transformers
    embed_dim: dimension of input embedding
    nhead: number of attention heads
    """

    def __init__(self, embed_dim, nhead, aggr, dropout=0.1, mult_attr=0):
        super(MultiHeadDotProduct, self).__init__()
        print("MultiHeadDotProduct")
        self.embed_dim = embed_dim
        self.hdim = embed_dim // nhead
        self.nhead = nhead
        self.aggr = aggr
        self.mult_attr = mult_attr

        # FC Layers for input
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # fc layer for concatenated output
        self.out = nn.Linear(embed_dim, embed_dim)

        self.reset_parameters()

    def forward(self, feats: torch.tensor, edge_index: torch.tensor,
                edge_attr: torch.tensor):
        q = k = v = feats
        bs = q.size(0)

        # FC layer and split into heads --> h * bs * embed_dim
        k = self.k_linear(k).view(bs, self.nhead, self.hdim).transpose(0, 1)
        q = self.q_linear(q).view(bs, self.nhead, self.hdim).transpose(0, 1)
        v = self.v_linear(v).view(bs, self.nhead, self.hdim).transpose(0, 1)

        # perform multi-head attention
        feats = self._attention(q, k, v, edge_index, edge_attr, bs)
        # concatenate heads and put through final linear layer
        feats = feats.transpose(0, 1).contiguous().view(
            bs, self.nhead * self.hdim)
        feats = self.out(feats)

        return feats  # , edge_index, edge_attr

    def _attention(self, q, k, v, edge_index=None, edge_attr=None, bs=None):
        r, c, e = edge_index[:, 0], edge_index[:, 1], edge_index.shape[0]

        scores = torch.matmul(
            q.index_select(1, c).unsqueeze(dim=-2),
            k.index_select(1, r).unsqueeze(dim=-1))
        scores = scores.view(self.nhead, e, 1) / math.sqrt(self.hdim)
        scores = softmax(scores, c, 1, bs)
        scores = self.dropout(scores)

        if self.mult_attr:
            scores = scores * edge_attr.unsqueeze(1)

        out = scores * v.index_select(1, r)  # H x e x hdim
        out = self.aggr(out, c, 1, bs)  # H x bs x hdim
        if type(out) == tuple:
            out = out[0]
        return out

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.constant_(self.q_linear.bias, 0.)

        nn.init.xavier_uniform_(self.v_linear.weight)
        nn.init.constant_(self.v_linear.bias, 0.)

        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.constant_(self.k_linear.bias, 0.)

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.)
