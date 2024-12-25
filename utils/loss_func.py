import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from itertools import combinations
import pdb


class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.4, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def __call__(self, h, y, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps)
        # return CoxLoss(hazard_pred=h, survtime=y, censor=c)
        

# def R_set(x):
# 	n_sample = x.size(0)
# 	matrix_ones = torch.ones(n_sample, n_sample)
# 	indicator_matrix = torch.tril(matrix_ones)
# 	return(indicator_matrix)

# def CoxLoss(hazard_pred, survtime, censor):
#     n_observed = censor.sum(0)+1
#     ytime_indicator = R_set(survtime)
#     ytime_indicator = torch.FloatTensor(ytime_indicator).cuda()
#     risk_set_sum = ytime_indicator.mm(torch.exp(hazard_pred))
#     diff = hazard_pred - torch.log(risk_set_sum)
#     sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(censor.unsqueeze(1))
#     cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
#     print('\n')
#     print('hazard_pred, sum_diff_in_observed, hazard_pred, survtime, censor')
#     print(hazard_pred, sum_diff_in_observed, hazard_pred, survtime, censor)
#     return cost

# def nll_loss(hazards, Y, c, alpha=0.4, eps=1e-7):
#     batch_size = len(Y)
#     print('hazards ', hazards)
#     Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
#     c = c.view(batch_size, 1).float() #censorship status, 0 or 1
#     S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
#     print('S ', S)
#     # without padding, S(0) = S[0], h(0) = h[0]
#     S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
#     # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
#     #h[y] = h(1)
#     #S[1] = S(1)
#     print('S_padded', S_padded)
#     print('Y', Y)
#     print('eps', eps)
#     print('c ', c)
#     print('hazards ', hazards)
#     uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
#     censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
#     neg_l = censored_loss + uncensored_loss
#     loss = (1-alpha) * neg_l + alpha * uncensored_loss
#     loss = loss.mean()
#     return loss

# TODO: document better and clean up
def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='sum'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
        The weight on uncensored loss 
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    # print("h shape", h.shape)
    # print("y shape", y.shape)
    # print("c shape", c.shape)
    # print("h ", h)
    # print("y ", y)
    # print("c ", c)
    
    # make sure these are ints
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    # hazards = torch.sigmoid(h)
    hazards = h
    # print("hazards shape", hazards.shape)

    S = torch.cumprod(1 - hazards, dim=1)
    # print("S.shape", S.shape, S)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # hazards[y] = hazards(1)
    # S[1] = S(1)
    # TODO: document and check

    # print("S_padded.shape", S_padded.shape, S_padded)


    # TODO: document/better naming
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)
    
    # s_prev = S_padded
    # h_this = hazards
    # # s_this = S_padded
    # print('s_prev.s_prev', s_prev.shape, s_prev)
    # print('h_this.shape', h_this.shape, h_this)
    # print('s_this.shape', s_this.shape, s_this)

    # c = 1 means censored. Weight 0 in this case 
    # print('s_prev ', s_prev)
    # print('c ', c)
    # print('h_this ', h_this)
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)
    

    # print('uncensored_loss.shape', uncensored_loss.shape)
    # print('censored_loss.shape', censored_loss.shape)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss

class OrthogonalLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, P, P_hat, G, G_hat):
        pos_pairs = (1 - torch.abs(F.cosine_similarity(P.detach(), P_hat, dim=1))) + (
            1 - torch.abs(F.cosine_similarity(G.detach(), G_hat, dim=1))
        )
        neg_pairs = (
            torch.abs(F.cosine_similarity(P, G, dim=1))
            + torch.abs(F.cosine_similarity(P.detach(), G_hat, dim=1))
            + torch.abs(F.cosine_similarity(G.detach(), P_hat, dim=1))
        )

        loss = pos_pairs + self.gamma * neg_pairs
        return loss