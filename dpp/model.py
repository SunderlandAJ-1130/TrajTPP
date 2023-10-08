import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dpp
from torch.distributions import Categorical
from dpp.utils import DotDict, diff
from dpp.nn import BaseModule, GRAN
import pandas as pd
from dpp.data import SimpleBatch
from torch.nn.functional import one_hot


class Model(BaseModule):
    """Base model class.

    Attributes:
        rnn: RNN for encoding the event history.
        embedding: Retrieve static embedding for each sequence.
        decoder: Compute log-likelihood of the inter-event times given hist and emb.

    Args:
        config: General model configuration (see dpp.model.ModelConfig).
        decoder: Model for computing log probability of t given history and embeddings.
            (see dpp.decoders for a list of possible choices)
    """
    def __init__(self, config, decoder):
        super().__init__()
        self.use_history(config.use_history)
        self.use_embedding(config.use_embedding)
        self.use_marks(config.use_marks)
        self.device = config.device
        self.use_sa = config.use_sa
        self.num_heads = config.num_heads
        self.use_timeofday = config.use_timeofday
        self.use_dayofweek = config.use_dayofweek
        self.use_driver = config.use_driver
        self.num_driver = config.num_driver
        self.history_size = config.history_size
        self.decoder_name = config.decoder_name
        self.embedding_size = config.embedding_size
        self.use_prior = config.use_prior
        self.joint_type = config.joint_type
        
        self.rnn = dpp.nn.RNNLayer(config)
        if self.using_embedding:
            self.embedding = nn.Embedding(config.num_embeddings, config.embedding_size)
            self.embedding.weight.data.fill_(0.0)
            
        if self.using_marks:
            self.num_classes = config.num_classes
            self.mark_layer = nn.Sequential(
                nn.Linear(config.history_size, config.history_size),
                nn.ReLU(),
                nn.Linear(config.history_size, self.num_classes)
            )
            if self.use_prior:
                self.od_emb = nn.Linear(self.num_classes, self.num_classes)
                
                self.lin_fc_prior = nn.Linear(self.num_classes, self.num_classes)
                self.lin_fc_mark = nn.Linear(self.num_classes, self.num_classes)
                self.lin_bias = nn.Parameter(torch.empty(self.num_classes, ))
            
        if self.use_driver:
            self.driver_embedding = nn.Embedding(self.num_driver, self.embedding_size)
        
        if self.use_timeofday:
            self.hour_embedding = nn.Embedding(24, self.embedding_size)
        
        if self.use_dayofweek:
            self.day_embedding = nn.Embedding(2, self.embedding_size)
        
        # enrichment gran
        context_hidden_size = self.embedding_size * (int(self.use_driver) + int(self.use_timeofday) + int(self.use_dayofweek))
        
        if self.use_driver or self.use_timeofday or self.use_dayofweek:
            self.enrichment_gran = GRAN(
                config.history_size,
                config.history_size,
                config.history_size,
                context_hidden_size=context_hidden_size,
                dropout=0.1)

        self.decoder = decoder

    def mark_nll(self, h, y, prior_info=None, input=None, return_val=False):
        """Compute log likelihood and accuracy of predicted marks

        Args:
            h: History vector
            y: Out marks, true label

        Returns:
            loss: Negative log-likelihood for marks, shape (batch_size, seq_len)
            accuracy: Percentage of correctly classified marks
        """
        mark_context = self.mark_layer(h)
        prior_info = None
        
        if prior_info is not None:
            prior_info = self.od_emb(prior_info)
            prior_info = torch.softmax(prior_info, dim=-1)
            
            prior_info = torch.cat(
                [prior_info[input.in_mark[i]].unsqueeze(dim=0) for i in range(input.in_mark.shape[0])], 
                dim=0).to(self.device).float()
            
            mark_context_ = self.lin_fc_mark(mark_context)
            prior_context_ = self.lin_fc_prior(prior_info)
            
            z = torch.sigmoid(mark_context_+prior_context_+self.lin_bias)
            h = z*mark_context_ + (1-z)*prior_context_
            mark_context = h
        
        mark_logits = torch.log_softmax(mark_context, dim=-1)  # (batch_size, seq_len, num_marks)
        mark_dist = Categorical(logits=mark_logits)
 
        if return_val:
            return mark_dist.log_prob(y), mark_logits.argmax(-1), mark_logits.exp()
        else:
            return mark_dist.log_prob(y)

    def get_external_features(self, input) -> torch.Tensor:
        external_context = []
        
        # static feature encoding        
        if self.use_driver:
            driver_embedding = self.driver_embedding(input.in_driver[:, 0].reshape(-1, 1)).reshape(input.in_driver.shape[0], -1)
            external_context.append(driver_embedding)
        if self.use_timeofday:
            hour_embedding = self.hour_embedding(input.in_timeofday[:, 0].reshape(-1, 1)).reshape(input.in_timeofday.shape[0], -1)
            external_context.append(hour_embedding)
        if self.use_dayofweek:
            day_embedding = self.day_embedding(input.in_dayofweek[:, 0].reshape(-1, 1)).reshape(input.in_dayofweek.shape[0], -1)
            external_context.append(day_embedding)
            
        external_context = torch.cat(external_context, dim=-1)
        
        return external_context
    
    def get_context(self, input):
        if self.using_history:
            h, prior_info = self.rnn(input) # has shape (batch_size, seq_len, rnn_hidden_size)
            
            # static enrichment
            if self.use_driver or self.use_timeofday or self.use_dayofweek:
                external_context = self.get_external_features(input)
                h = self.enrichment_gran(h, c=external_context)
            
            # if self.use_driver or self.use_timeofday or self.use_sa:
            #     h = torch.nn.Tanh()(h)
        else:
            h = None
            
        return h, prior_info

    def log_prob(self, input):
        """Compute log likelihood of the inter-event timesi in the batch.

        Args:
            input: Batch of data to score. See dpp.data.Input.

        Returns:
            time_log_prob: Log likelihood of each data point, shape (batch_size, seq_len)
            mark_nll: Negative log likelihood of marks, if using_marks is True
            accuracy: Accuracy of marks, if using_marks is True
        """
        # Encode the history with an RNN
        h, prior_info = self.get_context(input)
        
        # Get sequence embedding
        if self.using_embedding:
            # has shape (batch_size, seq_len, embedding_size)
            emb = self.embedding(input.index).unsqueeze(1).repeat(1, input.out_time.shape[1], 1)
        else:
            emb = None

        t = input.out_time  # has shape (batch_size, seq_len)
        time_log_prob = self.decoder.log_prob(t, h, emb)

        if self.using_marks:
            if self.joint_type == 'joint':
                multi_marks = one_hot(input.in_mark, num_classes=self.num_classes).float() # (batch_size, seq_len, num_marks)
                inter_times = t.unsqueeze_(-1)
                
                inter_times = inter_times.expand(
                    inter_times.shape[0],
                    inter_times.shape[1],
                    self.num_classes).clone() # (batch_size, seq_len, num_marks)
                time_log_prob = time_log_prob.unsqueeze(-1)

                pos_log_p = time_log_prob * multi_marks
                neg_log_p = time_log_prob * (1 - multi_marks)
                
                mark_context = self.mark_layer(h)
                mark_logits = torch.log_softmax(mark_context, dim=-1)  # (batch_size, seq_len, num_marks)
                mark_dist = Categorical(logits=mark_logits)
                log_mark = mark_dist.log_prob(input.out_mark)
                
                tot_nll = (pos_log_p + mark_logits * multi_marks).sum(-1)
                tot_nll = tot_nll.sum(-1)
                mark_class_joint = ((time_log_prob + mark_logits).argmax(-1)).float() #(batch_size, seq_len)
                log_time = pos_log_p.sum(-1)

                return log_time, log_mark
            else:
                mark_nll = self.mark_nll(h, input.out_mark, prior_info=prior_info, input=input)
                
                return time_log_prob, mark_nll
            
        return time_log_prob

    def aggregate(self, values, lengths):
        """Calculate masked average of values.

        Sequences may have different lengths, so it's necessary to exclude
        the masked values in the padded sequence when computing the average.

        Arguments:
            values (list[tensor]): List of batches where each batch contains
                padded values, shape (batch size, sequence length)
            lengths (list[tensor]): List of batches where each batch contains
                lengths of sequences in a batch, shape (batch size)

        Returns:
            mean (float): Average value in values taking padding into account
        """

        if not isinstance(values, list):
            values = [values]
        if not isinstance(lengths, list):
            lengths = [lengths]

        total = 0.0
        for batch, length in zip(values, lengths):
            length = length.long()
            mask = torch.arange(batch.shape[1], device=self.device)[None, :] < length[:, None]
            mask = mask.float().to(batch.device)

            batch[torch.isnan(batch)] = 0 # set NaNs to 0
            batch *= mask

            total += batch.sum()

        total_length = sum(x.sum() for x in lengths)

        return total / total_length.to(total.device)
    
    def sample(
        self, driver_info: torch.Tensor, t_start: float, t_end: float, 
        mean_in_train, std_in_train, dayofweek,
        batch_size: int = 1, context_init: torch.Tensor = None) -> dpp.data.Batch:
        # sourcery skip: avoid-builtin-shadow
        """Generate a batch of sequence from the model.
        Args:
            t_end: Size of the interval on which to simulate the TPP.
            batch_size: Number of independent sequences to simulate.
            context_init: Context vector for the first event.
                Can be used to condition the generator on past events,
                shape (context_size,)
        Returns;
            batch: Batch of sampled sequences. See dpp.data.batch.Batch.
        """
        if context_init is None:
            # Use the default context vector
            context_init = torch.zeros(self.history_size, device=self.device)
        else:
            # Use the provided context vector
            context_init = context_init.view(self.history_size)
        next_context = context_init[None, None, :].expand(batch_size, 1, -1)
        inter_times = torch.empty(batch_size, 0, device=self.device)
        taus = torch.empty(batch_size, 0, device=self.device)
        if self.num_classes > 1:
            marks = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)

        generated = False
        while not generated:
            inter_time_dist = self.decoder
            # next_tau is the actual travel time. not log, and not norm. 
            next_tau = inter_time_dist.sample(n_samples=1, h=next_context).squeeze(-1).to(self.device)
            next_inter_times = ((next_tau + 1e-8).log() - mean_in_train) / std_in_train

            taus = torch.cat([taus.to(self.device), next_tau], dim=1)  # (batch_size, seq_len)
            inter_times = torch.cat([inter_times.to(self.device), next_inter_times], dim=1)  # (batch_size, seq_len)
            # generate new mark
            if self.using_marks:
                mark_logits = torch.log_softmax(self.mark_layer(next_context), dim=-1)  # (batch_size, 1, num_marks)
                mark_logits = torch.nan_to_num(mark_logits, nan=0.0)
                mark_dist = Categorical(logits=mark_logits)
                next_marks = mark_dist.sample().to(self.device)  # (batch_size, 1)
                marks = torch.cat([marks, next_marks], dim=1)
            else:
                marks = None

            with torch.no_grad():
                generated = taus.sum(-1).min() >= t_end

            # generate external features
            # driver
            drivers = driver_info[0][0].item() * torch.ones(size=(inter_times.shape), device=self.device).long()
            # time of day
            hours = (torch.ones(size=inter_times.shape, device=inter_times.device) * t_start).long()
            # day of week
            days = (torch.ones(size=inter_times.shape, device=inter_times.device) * dayofweek).long()

            # generate input
            input = SimpleBatch(inter_times, marks, hours=hours, drivers=drivers, dayofweek=days, device=self.device)

            context = self.get_context(input)[0]  # (batch_size, seq_len, context_size)
            next_context = context[:, [-1], :]  # (batch_size, 1, context_size)   
            next_context = torch.nan_to_num(next_context, nan=0.0)

        arrival_times = taus.cumsum(-1)  # (batch_size, seq_len)
        taus = diff(arrival_times.clamp(max=t_end), dim=-1)
        mask = (arrival_times <= t_end).float()  # (batch_size, seq_len)
        if self.using_marks:
            marks = marks * mask  # (batch_size, seq_len)

        arrival_times = torch.cumsum(taus, axis=1).detach().cpu().numpy()
        delta_times = list(arrival_times)
        # delta_times = [np.concatenate([[1.0], np.ediff1d(time)]) for time in arrival_times]
        return SimpleBatch(delta_times, marks, device=self.device)

    def sample_v2(
        self, driver_info: torch.Tensor, t_start: float, t_end: float, 
        mean_in_train, std_in_train, dayofweek, start_mark,
        batch_size: int = 1, context_init: torch.Tensor = None) -> dpp.data.Batch:
        # sourcery skip: avoid-builtin-shadow
        """Generate a batch of sequence from the model.
        Args:
            t_end: Size of the interval on which to simulate the TPP.
            batch_size: Number of independent sequences to simulate.
            context_init: Context vector for the first event.
                Can be used to condition the generator on past events,
                shape (context_size,)
        Returns;
            batch: Batch of sampled sequences. See dpp.data.batch.Batch.
        """
        if context_init is None:
            # Use the default context vector
            context_init = torch.zeros(self.history_size, device=self.device)
        else:
            # Use the provided context vector
            context_init = context_init.view(self.history_size)
        next_context = context_init[None, None, :].expand(batch_size, 1, -1)
        inter_times = torch.empty(batch_size, 0, device=self.device)
        taus = torch.empty(batch_size, 0, device=self.device)
        if self.num_classes > 1:
            marks = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)

        generated = False
        while not generated:
            inter_time_dist = self.decoder
            # next_tau is the actual travel time. not log, and not norm. 
            next_tau = inter_time_dist.sample(n_samples=1, h=next_context).squeeze(-1).to(self.device)
            next_inter_times = ((next_tau + 1e-8).log() - mean_in_train) / std_in_train

            taus = torch.cat([taus.to(self.device), next_tau], dim=1)  # (batch_size, seq_len)
            inter_times = torch.cat([inter_times.to(self.device), next_inter_times], dim=1)  # (batch_size, seq_len)
            # generate new mark
            if self.using_marks:
                mark_logits = torch.log_softmax(self.mark_layer(next_context), dim=-1)  # (batch_size, 1, num_marks)
                mark_logits = torch.nan_to_num(mark_logits, nan=0.0)
                mark_dist = Categorical(logits=mark_logits)
                next_marks = mark_dist.sample().to(self.device)  # (batch_size, 1)
                marks = torch.cat([marks, next_marks], dim=1)
            else:
                marks = None

            with torch.no_grad():
                generated = taus.sum(-1).min() >= t_end

            # generate external features
            # driver
            drivers = driver_info[0][0].item() * torch.ones(size=(inter_times.shape), device=self.device).long()
            # time of day
            hours = (torch.ones(size=inter_times.shape, device=inter_times.device) * t_start).long()
            # day of week
            days = (torch.ones(size=inter_times.shape, device=inter_times.device) * dayofweek).long()

            # generate input
            input = SimpleBatch(inter_times, marks, hours=hours, drivers=drivers, dayofweek=days, device=self.device)

            context = self.get_context(input)  # (batch_size, seq_len, context_size)
            next_context = context[:, [-1], :]  # (batch_size, 1, context_size)   
            next_context = torch.nan_to_num(next_context, nan=0.0)

        arrival_times = taus.cumsum(-1)  # (batch_size, seq_len)
        taus = diff(arrival_times.clamp(max=t_end), dim=-1)
        mask = (arrival_times <= t_end).float()  # (batch_size, seq_len)
        if self.using_marks:
            marks = marks * mask  # (batch_size, seq_len)

        arrival_times = torch.cumsum(taus, axis=1).detach().cpu().numpy()
        delta_times = list(arrival_times)
        # delta_times = [np.concatenate([[1.0], np.ediff1d(time)]) for time in arrival_times]
        return SimpleBatch(delta_times, marks, device=self.device)


class ModelConfig(DotDict):
    """Configuration of the model.

    This config only contains parameters that need to be know by all the
    submodules. Submodule-specific parameters are passed to the respective
    constructors.

    Args:
        use_history: Should the model use the history embedding?
        history_size: Dimension of the history embedding.
        rnn_type: {'RNN', 'LSTM', 'GRU'}: RNN architecture to use.
        use_embedding: Should the model use the sequence embedding?
        embedding_size: Dimension of the sequence embedding.
        num_embeddings: Number of unique sequences in the dataset.
        use_marks: Should the model use the marks?
        mark_embedding_size: Dimension of the mark embedding.
        num_classes: Number of unique mark types, used as dimension of output
    """
    def __init__(self,
                 use_history=True,
                 history_size=32,
                 rnn_type='RNN',
                 use_embedding=False,
                 embedding_size=32,
                 num_embeddings=None,
                 use_marks=False,
                 mark_embedding_size=64,
                 device='cuda:0',
                 use_sa=False,
                 num_heads=8,
                 num_classes=None,
                 use_driver=False,
                 use_timeofday=False,
                 use_dayofweek=False,
                 decoder_name='LogNormMix',
                 attention_type='GAU',
                 min_clip=-3.0,
                 max_clip=5.0,
                 group_size=128,
                 use_prior=True,
                 mean_in_train=None,
                 std_in_train=None, 
                 joint_type=None, 
                 num_driver=0):
        super().__init__()
        # RNN parameters
        self.use_history = use_history
        self.history_size = history_size
        self.rnn_type = rnn_type
        self.device = device
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.decoder_name = decoder_name
        self.attention_type = attention_type
        self.group_size = group_size
        self.use_prior = use_prior

        # Sequence embedding parameters
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size
        if use_embedding and num_embeddings is None:
            raise ValueError("Number of embeddings has to be specified")
        self.num_embeddings = num_embeddings

        self.use_marks = use_marks
        self.mark_embedding_size = mark_embedding_size
        self.num_classes = num_classes
        
        self.use_sa = use_sa
        self.num_heads = num_heads
        
        self.use_driver = use_driver
        self.use_timeofday = use_timeofday
        self.num_driver = num_driver
        self.use_dayofweek = use_dayofweek
        
        self.mean_in_train = mean_in_train
        self.std_in_train = std_in_train
        self.joint_type = joint_type
        