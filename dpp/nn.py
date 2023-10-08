import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import torch.nn.functional as F
from torch.nn import LayerNorm
from typing import Dict, Tuple, Optional, List
import math
from einops import rearrange
from .gau_layer import GAULayer


class BaseModule(nn.Module):
    """Wrapper around nn.Module that recursively sets history and embedding usage.

    All modules should inherit from this class.
    """
    def __init__(self):
        super().__init__()
        self._using_history = False
        self._using_embedding = False
        self._using_marks = False

    @property
    def using_history(self):
        return self._using_history

    @property
    def using_embedding(self):
        return self._using_embedding

    @property
    def using_marks(self):
        return self._using_marks

    def use_history(self, mode=True):
        """Recursively make all submodules use history."""
        self._using_history = mode
        for module in self.children():
            if isinstance(module, BaseModule):
                module.use_history(mode)

    def use_embedding(self, mode=True):
        """Recursively make all submodules use embeddings."""
        self._using_embedding = mode
        for module in self.children():
            if isinstance(module, BaseModule):
                module.use_embedding(mode)

    def use_marks(self, mode=True):
        """Recursively make all submodules use embeddings."""
        self._using_marks = mode
        for module in self.children():
            if isinstance(module, BaseModule):
                module.use_marks(mode)


class RNNLayer(BaseModule):
    """RNN for encoding the event history."""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.history_size
        self.rnn_type = config.rnn_type
        self.use_history(config.use_history)
        self.use_marks(config.use_marks)
        self.device = config.device
        self.use_sa = config.use_sa
        self.num_heads = config.num_heads
        self.attention_type = config.attention_type
        self.group_size = config.group_size
        self.use_prior = config.use_prior
        
        if self.use_prior:
            prior_data = np.load('./dataset/CS trajectory-15-5/history_information.npz')
            self.od_matrix = torch.tensor(prior_data['OD_matrix'].sum(axis=0), device=self.device).float()
            self.tt_matrix = torch.tensor(prior_data['ATT_matrix'].sum(axis=0), device=self.device).float()
            
            self.tt_matrix[self.tt_matrix<1e-5] = self.tt_matrix.max()
            self.tt_matrix = torch.softmax(torch.tensor(-self.tt_matrix), dim=-1)
            self.od_matrix = torch.softmax(self.od_matrix, dim=-1)
            
            self.prior_mark_fc = torch.nn.Linear(in_features=self.od_matrix.shape[1], out_features=self.hidden_size)
            self.prior_time_fc = torch.nn.Linear(in_features=self.tt_matrix.shape[1], out_features=self.hidden_size)
        else:
            self.od_matrix = None
            self.tt_matrix = None

        if config.use_marks:
            # Define mark embedding layer
            self.mark_embedding = nn.Embedding(config.num_classes, config.mark_embedding_size)
            # If we have marks, input is time + mark embedding vector
            self.in_features = config.mark_embedding_size + 1
        else:
            # Without marks, input is only time
            self.in_features = 1
        
        if self.use_sa:
            self.temporal_embed = nn.Linear(in_features=1, out_features=config.history_size)
            if self.attention_type == 'GAU':
                self.temporal_attn = GAULayer(hidden_size=config.history_size, intermediate_size=config.history_size*2)
                self.spatial_attn = GAULayer(hidden_size=config.history_size, intermediate_size=config.history_size*2)
            elif self.attention_type == 'inter_ma':
                self.temporal_attn = InterpretableMultiHeadAttention(n_head=self.num_heads, d_model=config.history_size, dropout=0.1)
                self.spatial_attn = InterpretableMultiHeadAttention(n_head=self.num_heads, d_model=config.history_size, dropout=0.1)
            elif self.attention_type == 'ma':
                self.temporal_attn = torch.nn.MultiheadAttention(config.history_size, self.num_heads, batch_first=True, dropout=0.1)
                self.spatial_attn = torch.nn.MultiheadAttention(config.history_size, self.num_heads, batch_first=True, dropout=0.1)
            
            if config.use_marks:
                self.in_features = config.history_size + config.mark_embedding_size
            else:
                self.in_features = config.history_size
            
            if self.attention_type == 'GAU':
                self.output_attn = GAULayer(hidden_size=config.history_size, intermediate_size=config.history_size*2)
            elif self.attention_type == 'ma':
                self.output_attn = torch.nn.MultiheadAttention(config.history_size, self.num_heads, batch_first=True, dropout=0.1)

        # Possible RNN types: 'RNN', 'GRU', 'LSTM'
        if self.rnn_type in ['RNN', 'GRU', 'LSTM']:
            self.rnn = getattr(nn, self.rnn_type)(self.in_features, self.hidden_size, batch_first=True)
        elif self.rnn_type == 'GGRU':
            if self.use_sa is False:
                self.temporal_embed_x = nn.Linear(in_features=1, out_features=config.history_size)
            self.rnn = GGRU(num_inputs=self.hidden_size, num_hiddens=self.hidden_size, dropout=0.1)
            # self.rnn = GGRU_v0(num_inputs=self.hidden_size, num_hiddens=self.hidden_size, dropout=0.1)

    def forward(self, input):
        """Encode the history of the given batch.

        Returns:
            h: History encoding, shape (batch_size, seq_len, self.hidden_size)
        """
        t = input.in_time
        length = input.length
        pre_len = t.shape[-1]
            
        if self.use_prior:
            od_emb = self.prior_mark_fc(self.od_matrix)
            prior_info_ = torch.matmul(od_emb, od_emb.T)
            prior_info_ = torch.softmax(prior_info_, dim=-1)
            prior_mark_info = torch.cat(
                [prior_info_[input.in_mark[i]][:, input.in_mark[i]].unsqueeze(dim=0) for i in range(input.in_mark.shape[0])], 
                dim=0).to(self.device).float()
            
            time_emb = self.prior_time_fc(self.tt_matrix)
            prior_info_ = torch.matmul(time_emb, time_emb.T)
            prior_info_ = torch.softmax(prior_info_, dim=-1)
            prior_time_info = torch.cat(
                [prior_info_[input.in_mark[i]][:, input.in_mark[i]].unsqueeze(dim=0) for i in range(input.in_mark.shape[0])], 
                dim=0).to(self.device).float()
        else:
            prior_time_info = None
            prior_mark_info = None
        
        if not self.using_history:
            return torch.zeros(t.shape[0], t.shape[1], self.hidden_size, device=t.device)
        
        x = t.unsqueeze(-1)
        if self.use_sa:
            x = self.temporal_embed(x)
            if self.attention_type == 'GAU':
                masked_matrix = torch.tril(torch.ones(size=(x.shape[0], pre_len, pre_len), device=x.device), diagonal=0)
                x = self.temporal_attn(x, output_attentions=False, attention_mask=masked_matrix, prior_info=prior_time_info)[0]
            elif self.attention_type == 'inter_ma':
                q, k, v = x, x, x
                masked_matrix = torch.triu(torch.ones(size=(pre_len, pre_len), device=q.device), diagonal=1).bool()
                x, weights = self.temporal_attn(q, k, v, mask=masked_matrix) # Q, K, V, attn_mask for causality
            elif self.attention_type == 'ma':
                q, k, v = x, x, x
                masked_matrix = torch.triu(torch.ones(x.shape[0]*self.num_heads, x.shape[1], x.shape[1], device=q.device), diagonal=1) * (-1e8)
                x, weights = self.temporal_attn(q, k, v, attn_mask=masked_matrix) # Q, K, V, attn_mask for causality
                
        if self.using_marks:
            mark = self.mark_embedding(input.in_mark)
            # self-attention
            if self.use_sa:
                if self.attention_type == 'GAU':
                    masked_matrix = torch.tril(torch.ones(size=(mark.shape[0], pre_len, pre_len), device=mark.device), diagonal=0)
                    mark = self.spatial_attn(mark, output_attentions=False, attention_mask=masked_matrix, prior_info=prior_mark_info)[0]
                elif self.attention_type == 'inter_ma':
                    q, k, v = mark, mark, mark
                    masked_matrix = torch.triu(torch.ones(size=(pre_len, pre_len), device=q.device), diagonal=1).bool()
                    mark, weights = self.spatial_attn(q, k, v, mask=masked_matrix) # Q, K, V, attn_mask for causality
                elif self.attention_type == 'ma':
                    q, k, v = mark, mark, mark
                    masked_matrix = torch.triu(torch.ones(mark.shape[0]*self.num_heads, mark.shape[1], mark.shape[1], device=q.device), diagonal=1) * (-1e8)
                    mark, weights = self.spatial_attn(q, k, v, attn_mask=masked_matrix) # Q, K, V, attn_mask for causality

        if self.rnn_type == 'GGRU':
            # h, _ = self.rnn(x, mark, input.in_time.unsqueeze(dim=-1))
            if self.use_sa is False:
                x = self.temporal_embed_x(x)
            h, _ = self.rnn(x, mark)
        else:
            x = torch.cat([x, mark], -1)
            # x = torch.cat([mark, x], -1)
            
            h_shape = (1, x.shape[0], self.hidden_size)
            if self.rnn_type == 'LSTM':
                # LSTM keeps two hidden states
                h0 = (torch.zeros(h_shape, device=t.device), torch.zeros(h_shape, device=t.device))
            else:
                # RNN and GRU have one hidden state
                h0 = torch.zeros(h_shape, device=t.device)
            
            x, batch_sizes = torch._C._VariableFunctions._pack_padded_sequence(x, length.cpu().long(), batch_first=True)
            x = torch.nn.utils.rnn.PackedSequence(x, batch_sizes)

            h, _ = self.rnn(x, h0)
            h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)

        if self.use_sa:
            if self.attention_type == 'GAU':
                masked_matrix = torch.tril(torch.ones(size=(h.shape[0], h.shape[1], h.shape[1]), device=h.device), diagonal=0)
                h = self.output_attn(h, output_attentions=False, attention_mask=masked_matrix)[0]
            elif self.attention_type == 'ma':
                q, k, v = h, h, h
                masked_matrix = torch.triu(torch.ones(h.shape[0]*self.num_heads, h.shape[1], h.shape[1], device=q.device), diagonal=1) * (-1e8)
                h, weights = self.output_attn(q, k, v, attn_mask=masked_matrix) # Q, K, V, attn_mask for causality

        return h, self.od_matrix

    def step(self, x, h):
        """Given input and hidden state produces the output and new state."""
        y, h = self.rnn(x, h)
        return y, h


class Hypernet(nn.Module):
    """Hypernetwork for incorporating conditional information.

    Args:
        config: Model configuration. See `dpp.model.ModelConfig`.
        hidden_sizes: Sizes of the hidden layers. [] corresponds to a linear layer.
        param_sizes: Sizes of the output parameters.
        activation: Activation function.
    """
    def __init__(self, config, hidden_sizes=None, param_sizes=None, activation=nn.Tanh()):
        if hidden_sizes is None:
            hidden_sizes = []
        if param_sizes is None:
            param_sizes = [1, 1]
        super().__init__()
        self.history_size = config.history_size
        self.embedding_size = config.embedding_size
        self.activation = activation

        # Indices for unpacking parameters
        ends = torch.cumsum(torch.tensor(param_sizes), dim=0)
        starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        self.param_slices = [slice(s.item(), e.item()) for s, e in zip(starts, ends)]

        self.output_size = sum(param_sizes)
        layer_sizes = list(hidden_sizes) + [self.output_size]

        # Bias used in the first linear layer
        self.first_bias = nn.Parameter(torch.empty(layer_sizes[0]).uniform_(-0.1, 0.1))
        if config.use_history:
            self.linear_rnn = nn.Linear(self.history_size, layer_sizes[0], bias=False)
        if config.use_embedding:
            self.linear_emb = nn.Linear(self.embedding_size, layer_sizes[0], bias=False)
        # Remaining linear layers
        self.linear_layers = nn.ModuleList()
        for idx, size in enumerate(layer_sizes[:-1]):
            self.linear_layers.append(nn.Linear(size, layer_sizes[idx + 1]))
        
    def reset_parameters(self):
        self.first_bias.data.fill_(0.0)
        if hasattr(self, 'linear_rnn'):
            self.linear_rnn.reset_parameters()
            nn.init.orthogonal_(self.linear_rnn.weight)
        if hasattr(self, 'linear_emb'):
            self.linear_emb.reset_parameters()
            nn.init.orthogonal_(self.linear_emb.weight)
        for layer in self.linear_layers:
            layer.reset_parameters()
            nn.init.orthogonal_(layer.weight)

    def forward(self, h=None, emb=None):
        """Generate model parameters from the embeddings.

        Args:
            h: History embedding, shape (*, history_size)
            emb: Sequence embedding, shape (*, embedding_size)

        Returns:
            params: Tuple of model parameters.
        """
        # Generate the output based on the input
        if h is None and emb is None:
            # If no history or emb are provided, return bias of the final layer
            # 0.0 is added to create a new node in the computational graph
            # in case the output will be modified by an inplace operation later
            if len(self.linear_layers) == 0:
                hidden = self.first_bias + 0.0
            else:
                hidden = self.linear_layers[-1].bias + 0.0
        else:
            hidden = self.first_bias
            if h is not None:
                hidden = hidden + self.linear_rnn(h)
            if emb is not None:
                hidden = hidden + self.linear_emb(emb)
            for layer in self.linear_layers:
                hidden = layer(self.activation(hidden))

        # Partition the output
        if len(self.param_slices) == 1:
            return hidden
        else:
            return tuple(hidden[..., s] for s in self.param_slices)


class MaybeLayerNorm(nn.Module):
    def __init__(self, output_size, hidden_size, eps):
        super().__init__()
        if output_size and output_size == 1:
            self.ln = nn.Identity()
        else:
            self.ln = LayerNorm(output_size or hidden_size, eps=eps)
    
    def forward(self, x):
        return self.ln(x)
    

class GLU(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.lin = nn.Linear(hidden_size, output_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        x = F.glu(x)
        return x


class GRAN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size, 
                 output_size=None,
                 context_hidden_size=None,
                 dropout=0):
        super().__init__()
        self.layer_norm = MaybeLayerNorm(output_size, hidden_size, eps=1e-3)
        self.lin_a = nn.Linear(input_size, hidden_size)
        if context_hidden_size is not None:
            self.lin_c = nn.Linear(context_hidden_size, hidden_size, bias=False)
        self.lin_i = nn.Linear(hidden_size, hidden_size)

        self.gau = GAULayer(hidden_size=hidden_size, intermediate_size=hidden_size*2)        
        # self.glu = GLU(hidden_size, output_size or hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_size, output_size) if output_size else None

    def forward(self, a: torch.Tensor, c: Optional[torch.Tensor] = None):
        x = self.lin_a(a)
        if c is not None:
            x = x + self.lin_c(c).unsqueeze(1)
        x = F.elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)

        # GAU
        masked_matrix = torch.tril(torch.ones(size=(x.shape[0], x.shape[1], x.shape[1]), device=x.device), diagonal=0)
        x = self.gau(x, output_attentions=False, attention_mask=masked_matrix)[0]
        # x = self.glu(x)
        
        y = self.out_proj(a) if self.out_proj else a
        x = x + y
        x = self.layer_norm(x)
        return x 


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = None, scale: bool = True):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else dropout
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale
    
    def forward(self, q, k, v, mask=None, prior_info=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap

        if self.scale:
            dimension = torch.as_tensor(k.size(-1), dtype=attn.dtype, device=attn.device).sqrt()
            attn = attn / dimension

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        
        if prior_info is not None:
            prior_info = prior_info.masked_fill(mask, -1e9)
            z = torch.sigmoid(attn+prior_info)
            attn = z*attn + (1-z)*prior_info
            
        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        
        return output, attn


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0):
        super(InterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)
                
    def forward(self, q, k, v, mask=None, prior_info=None) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        attns = []
        vs = self.v_layer(v)
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask, prior_info)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)

        return outputs, attn


class GGRU_v0(torch.nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01, dropout=0.0):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_inputs = num_inputs

        # parameters for inter-times
        init_weight_t = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        triple_t = lambda: (init_weight_t(num_inputs, num_hiddens),
                          init_weight_t(num_hiddens, num_hiddens),
                          nn.Parameter(torch.zeros(num_hiddens)))
        self.W_xz_t, self.W_hz_t, self.b_z_t = triple_t()  # Update gate
        self.W_xr_t, self.W_hr_t, self.b_r_t = triple_t()  # Reset gate
        self.W_xh_t, self.W_hh_t, self.b_h_t = triple_t()  # Candidate hidden state
        
        # parameters for marks
        init_weight_m = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        triple_m = lambda: (init_weight_m(num_inputs, num_hiddens),
                          init_weight_m(num_hiddens, num_hiddens),
                          nn.Parameter(torch.zeros(num_hiddens)))
        self.W_xz_m, self.W_hz_m, self.b_z_m = triple_m()  # Update gate
        self.W_xr_m, self.W_hr_m, self.b_r_m = triple_m()  # Reset gate
        self.W_xh_m, self.W_hh_m, self.b_h_m = triple_m()  # Candidate hidden state
        
        # fusion mechanism
        self.fus_W_t = nn.Parameter(torch.empty(num_hiddens, num_hiddens).uniform_(-0.1, 0.1))
        self.fus_W_m = nn.Parameter(torch.empty(num_hiddens, num_hiddens).uniform_(-0.1, 0.1))
        self.fus_bias = nn.Parameter(torch.empty(num_hiddens, ).uniform_(-0.1, 0.1))
        
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, tt_inputs, mark_inputs, H=None):
        if H is None:
            # Initial state with shape: (batch_size, num_hiddens)
            H = torch.zeros((tt_inputs.shape[0], self.num_hiddens),
                        device=tt_inputs.device)

        outputs = []
        for i in range(tt_inputs.shape[1]):
            X_t = tt_inputs[:, i]            
            Z_t = torch.sigmoid(torch.matmul(X_t, self.W_xz_t) +
                            torch.matmul(H, self.W_hz_t) + self.b_z_t)
            R_t = torch.sigmoid(torch.matmul(X_t, self.W_xr_t) +
                            torch.matmul(H, self.W_hr_t) + self.b_r_t)
            H_tilde_t = torch.tanh(torch.matmul(X_t, self.W_xh_t) +
                            torch.matmul(R_t * H, self.W_hh_t) + self.b_h_t)
            H_t = Z_t * H + (1 - Z_t) * H_tilde_t

            X_m = mark_inputs[:, i]
            Z_m = torch.sigmoid(torch.matmul(X_m, self.W_xz_m) +
                            torch.matmul(H, self.W_hz_m) + self.b_z_m)
            R_m = torch.sigmoid(torch.matmul(X_m, self.W_xr_m) +
                            torch.matmul(H, self.W_hr_m) + self.b_r_m)
            H_tilde_m = torch.tanh(torch.matmul(X_m, self.W_xh_m) +
                            torch.matmul(R_m * H, self.W_hh_m) + self.b_h_m)
            H_m = Z_m * H + (1 - Z_m) * H_tilde_m

            z = torch.sigmoid(torch.matmul(H_t, self.fus_W_t)+torch.matmul(H_m, self.fus_W_m)+self.fus_bias)
            H = z*H_t + (1-z)*H_m

            outputs.append(H)

        outputs = torch.stack(outputs).transpose(0, 1)
        outputs = self.dropout(outputs)

        return outputs, H


class GGRU(torch.nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01, dropout=0.0):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_inputs = num_inputs

        # parameters for inter-times
        init_weight_t = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        triple = lambda: (init_weight_t(num_inputs, num_hiddens),
                          init_weight_t(num_hiddens, num_hiddens),
                          nn.Parameter(torch.zeros(num_hiddens)))
        self.W_xz_t, self.W_hz_t, self.b_z_t = triple()  # Update gate
        self.W_xr_t, self.W_hr_t, self.b_r_t = triple()  # Reset gate

        # parameters for marks
        init_weight_m = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        self.W_xz_m, self.W_hz_m, self.b_z_m = triple()  # Update gate
        self.W_xr_m, self.W_hr_m, self.b_r_m = triple()  # Reset gate

        # fusion mechanism
        self.fus_W_rt = nn.Parameter(torch.empty(num_hiddens, num_hiddens).uniform_(-0.1, 0.1))
        self.fus_W_rm = nn.Parameter(torch.empty(num_hiddens, num_hiddens).uniform_(-0.1, 0.1))
        self.fus_rbias = nn.Parameter(torch.empty(num_hiddens, ).uniform_(-0.1, 0.1))
        
        self.fus_W_zt = nn.Parameter(torch.empty(num_hiddens, num_hiddens).uniform_(-0.1, 0.1))
        self.fus_W_zm = nn.Parameter(torch.empty(num_hiddens, num_hiddens).uniform_(-0.1, 0.1))
        self.fus_zbias = nn.Parameter(torch.empty(num_hiddens, ).uniform_(-0.1, 0.1))
        
        self.fus_W_xt = nn.Parameter(torch.empty(num_inputs, num_inputs).uniform_(-0.1, 0.1))
        self.fus_W_xm = nn.Parameter(torch.empty(num_inputs, num_inputs).uniform_(-0.1, 0.1))
        self.fus_xbias = nn.Parameter(torch.empty(num_inputs, ).uniform_(-0.1, 0.1))
        
        self.W_xh, self.W_hh, self.b_h = triple()  # Candidate hidden state
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, tt_inputs, mark_inputs, H=None):
        if H is None:
            # Initial state with shape: (batch_size, num_hiddens)
            H = torch.zeros((tt_inputs.shape[0], self.num_hiddens),
                        device=tt_inputs.device)

        outputs = []
        for i in range(tt_inputs.shape[1]):
            X_t = tt_inputs[:, i]
            X_m = mark_inputs[:, i]
            
            Z_t = torch.sigmoid(torch.matmul(X_t, self.W_xz_t) +
                            torch.matmul(H, self.W_hz_t) + self.b_z_t)
            R_t = torch.sigmoid(torch.matmul(X_t, self.W_xr_t) +
                            torch.matmul(H, self.W_hr_t) + self.b_r_t)
            Z_m = torch.sigmoid(torch.matmul(X_m, self.W_xz_m) +
                            torch.matmul(H, self.W_hz_m) + self.b_z_m)
            R_m = torch.sigmoid(torch.matmul(X_m, self.W_xr_m) +
                            torch.matmul(H, self.W_hr_m) + self.b_r_m)
            
            # fusion for Z
            z = torch.sigmoid(torch.matmul(Z_t, self.fus_W_zt)+torch.matmul(Z_m, self.fus_W_zm)+self.fus_zbias)
            Z = z*Z_t + (1-z)*Z_m
            # fusion for R
            r = torch.sigmoid(torch.matmul(R_t, self.fus_W_rt)+torch.matmul(R_m, self.fus_W_rm)+self.fus_rbias)
            R = r*R_t + (1-r)*R_m    
            # fusion for X
            x = torch.sigmoid(torch.matmul(X_t, self.fus_W_xt)+torch.matmul(X_m, self.fus_W_xm)+self.fus_xbias)
            X = x*X_t + (1-x)*X_m    
     
            H_tilde = torch.tanh(torch.matmul(X, self.W_xh) +
                            torch.matmul(R * H, self.W_hh) + self.b_h)
            H = Z * H + (1 - Z) * H_tilde

            outputs.append(H)
            
        outputs = torch.stack(outputs).transpose(0, 1)
        outputs = self.dropout(outputs)

        return outputs, H
