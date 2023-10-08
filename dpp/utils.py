import numpy as np
import torch.nn as nn
import torch
import random
import logging
import os
import sys
import torch.nn.functional as F
import pickle


def get_inter_times(arrival_times):
    """Convert arrival times to interevent times."""
    return arrival_times - np.concatenate([[0], arrival_times[:-1]])


def get_arrival_times(inter_times):
    """Convert interevent times to arrival times."""
    return inter_times.cumsum()


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def clamp_preserve_gradients(x, min, max):
    """Clamp the tensor while preserving gradients in the clamped region."""
    return x + (x.clamp(min, max) - x).detach()


def evaluation_tpp(model, test_dataset, decoder_name, std_in_train,
                   mean_in_train, log_path):  # sourcery skip: low-code-quality
    prob_hat = []
    y_true = []
    quantile = np.linspace(0.05, 0.95, 19)
    y_mark_hat = []
    y_mark = []
    y_mean = []
    step_index = []
    y_mark_hat_prob = []
    
    upper_inter_time = 30
    lower_inter_time = 1e-2
    upper_boundary = (np.log(upper_inter_time)-mean_in_train) / std_in_train
    lower_boundary = (np.log(lower_inter_time)-mean_in_train) / std_in_train

    model.eval()
    with torch.no_grad():
        for batch in test_dataset:
            # h = model.rnn(batch)
            h, prior_info = model.get_context(batch)

            dist = model.decoder.base_dist
            mark_nll, label_hat, mark_logits = model.mark_nll(h, batch.out_mark, prior_info=prior_info, input=batch, return_val=True)
            y_mark.append(batch.out_mark.detach().cpu().numpy().tolist())
            y_mark_hat.append(label_hat.detach().cpu().numpy().tolist())
            y_mark_hat_prob.append(mark_logits.detach().cpu().numpy().tolist())
            inter_times = batch.out_time.detach().cpu().numpy()
            
            # mean
            if model.decoder_name == 'LogNormMix':
                prior_logits, means, log_scales = dist.get_params(h, None)
                prior = prior_logits.exp()
                scales_squared = (log_scales * 2).exp()
                affine = model.decoder.transforms[0]
                a = affine.log_scale.exp().item()
                b = affine.shift.item()
                y_mean_ = (prior * torch.exp(a * means + b +  0.5 * a**2 * scales_squared)).sum(-1).detach().cpu().numpy()
            else:
                y_mean_ = np.zeros(shape=inter_times.shape)
            
            for i in range(h.shape[0]):
                for ii in range(h.shape[1]):
                    x = torch.linspace(lower_boundary, upper_boundary, 10000, device=h.device)
                    # x = torch.linspace(batch.in_time.min(), batch.in_time.max(), 10000, device=h.device)
                    cdf = dist.log_cdf(x, h[i, ii, :], None).exp().detach().cpu().numpy()
                    x = (x*std_in_train+mean_in_train).exp().cpu().numpy()
                    idx_ = np.where(~np.isnan(cdf))[0]
                    x = x[idx_]
                    cdf = cdf[idx_]

                    y_hat = []
                    for iii in range(len(quantile)):
                        conf = quantile[iii]
                        conf_idx = np.argmin(np.abs(cdf-conf)).tolist()
                        y_hat.append(x[conf_idx])
                    prob_hat.append(y_hat)
                    y_true.append(inter_times[i, ii])
                    y_mean.append(y_mean_[i, ii])
                    step_index.append(ii)
    
    result = {}
    result['y_hat'] = prob_hat
    result['y_label'] = y_true
    result['y_mean'] = y_mean
    result['y_mark_hat'] = y_mark_hat
    result['y_mark'] = y_mark
    result['step_index'] = step_index
    result['y_mark_hat_prob'] = y_mark_hat_prob
    
    with open(f'{log_path}evaluation_tpp.pkl', 'wb') as file:
        pickle.dump(result, file)


# Function that calculates the loss for the entire dataloader
def get_total_loss(model, loader, use_marks=False):
    loader_log_prob, loader_lengths = [], []
    for input in loader:
        if use_marks:
            output = model.log_prob(input)
            loader_log_prob.append((output[0]+output[1]).detach())
        else:
            loader_log_prob.append(model.log_prob(input).detach())
        loader_lengths.append(input.length.detach())
    return -model.aggregate(loader_log_prob, loader_lengths)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_logger(log_dir,
               name,
               log_filename='info.log',
               level=logging.INFO,
               write_to_file=True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    if write_to_file is True:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)

    return logger


def diff(x, dim: int = -1):
    """Inverse of x.cumsum(dim=dim).
    Compute differences between subsequent elements of the tensor.
    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension over which to compute the difference, {-2, -1}.
    Returns:
        diff: Tensor of the the same shape as x.
    """
    if dim == -1:
        return x - F.pad(x, (1, 0))[..., :-1]
    elif dim == -2:
        return x - F.pad(x, (0, 0, 1, 0))[..., :-1, :]
    else:
        raise ValueError("dim must be equal to -1 or -2")
