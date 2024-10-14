from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse, os
import pandas as pd
import numpy as np
import random
import math
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.fft as fft
from .PhysNetNegPearsonLoss import Neg_Pearson as Physnet_Neg_Pearson
from .NegPearsonLoss import Neg_Pearson

def get_loss(name):
    # Todo: get all parameters from config
    max_shift = 15 #config.get('max_shift', 15)  # Adjust this if max_shift should be configurable

    if name == 'physnet_pearson':
        return Physnet_Neg_Pearson()
    elif name == 'pearson':
        return Neg_Pearson()
    elif name == 'mcc':
        return MCCLoss()
    elif name == 'macc':
        return NegMACC(max_shift)
    elif name == 'soft_macc':
        return NegSoftMACC(max_shift)
    elif name == 'soft_maacc':
        return NegSoftMAACC(max_shift)
    elif name == 'soft_msacc':
        return NegSoftMSACC(max_shift)
    elif name == 'lse_cc':
        return NegLseCC(max_shift)
    elif name == 'lse_scc':
        return NegLseSCC(max_shift)
    elif name == 'talos':
        return TALOSLoss(30, ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010', 'P011', 'P012', 'P014', 'P015', 'P016', 'P017', 'P018', 'P019', 'P020', 'P021', 'P023','P024'])
    else:
        raise ValueError(f'{name} is an invalid loss function')


class ShiftedCorrelationLoss(nn.Module):
    
    def __init__(self, max_shift):
        super(ShiftedCorrelationLoss, self).__init__()
        self.max_shift = max_shift

    def pearson_correlation(self, preds, labels):
        combined = torch.stack((preds, labels), dim=0)
        corr = torch.corrcoef(combined)[0, 1]
        return corr_sum

    def correlations_to_loss(self, correlations):
        raise NotImplementedError()

    def forward(self, preds, labels):
        batch_size = preds.shape[0]
        pad = (self.max_shift, self.max_shift)
        padded_preds = F.pad(preds, pad, mode='constant', value=0)
        padded_labels = F.pad(labels, pad, mode='constant', value=0)

        loss = 0
        for i in range(batch_size):
            correlations = []

            for shift in range(-self.max_shift, self.max_shift + 1):

                rolled_preds = torch.roll(padded_preds[i], shifts=shift, dims=-1)
                
                combined = torch.stack((rolled_preds, padded_labels[i]), dim=0)
                corr = torch.corrcoef(combined)[0, 1]
                
                correlations.append(corr)
            
            correlations = torch.stack(correlations)

            loss += self.correlations_to_loss(correlations)

        return loss / batch_size

class NegMACC(ShiftedCorrelationLoss):
    """
    The NegMACC loss function computes the negative maximum amplitude cross correlation.
    """
    def correlations_to_loss(self, correlations):
        return 1 - torch.max(correlations)

class NegSoftMACC(ShiftedCorrelationLoss):    
    """
    The NegSoftMACC loss function computes the negative soft max amplitude cross correlation.
    """
    def correlations_to_loss(self, correlations):
        return 1 - torch.sum(F.softmax(correlations,dim=0) * correlations)

class NegSoftMAACC(ShiftedCorrelationLoss):
    """
    1 - sum(softmax(abs(correlations)) * abs(correlations))
    """
    def correlations_to_loss(self, correlations):
        abs_correlations = torch.abs(correlations)
        return 1 - torch.sum(F.softmax(abs_correlations, dim=0) * abs_correlations)

class NegSoftMSACC(ShiftedCorrelationLoss):
    """
    1 - sum(softmax(square(correlations)) * square(correlations))
    """
    def correlations_to_loss(self, correlations):
        square_correlations = correlations ** 2
        return 1 - torch.sum(F.softmax(square_correlations, dim=0) * square_correlations)

class NegLseCC(ShiftedCorrelationLoss):
    """
    -logsumexp(correlations)
    """
    def correlations_to_loss(self, correlations):
        return -torch.logsumexp(correlations, dim=0)

class NegLseSCC(ShiftedCorrelationLoss):
    """
    -logsumexp(square(correlations))
    """
    def correlations_to_loss(self, correlations):
        square_correlations = correlations ** 2
        return -torch.logsumexp(square_correlations, dim=0)

class MCCLoss(torch.nn.Module):
    """
    Loss as in Gideon and Stent - 2021 - The Way to my Heart is through Contrastive Learning: Remote Photoplethysmography from Unlabelled Video
    """
    def __init__(self, bandpass_freq=(40/60, 250/60), sampling_rate=30):
        """
        Initialize the MCC loss function.
        
        Parameters:
        - bandpass_freq: The frequency range for the bandpass filter in Hz.
        - sampling_rate: The sampling rate of the signal in Hz.
        """
        super(MCCLoss, self).__init__()
        self.bandpass_freq = bandpass_freq
        self.sampling_rate = sampling_rate
    
    def bandpass_filter(self, freqs):
        """
        Create a bandpass filter mask based on the given frequency range.
        
        Parameters:
        - freqs: The frequency values from the FFT output.
        
        Returns:
        - A mask that filters out frequencies outside the specified heart rate range.
        """
        low_freq, high_freq = self.bandpass_freq
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        return mask

    def forward(self, y, y_hat):
        """
        Compute the Maximum Cross-Correlation (MCC) loss.
        
        Parameters:
        - y: The ground truth signal (batch, time).
        - y_hat: The predicted signal (batch, time).
        
        Returns:
        - The MCC loss value.
        """
        # Step 1: Zero-pad the inputs to twice their length to prevent circular correlation
        pad_len = y.size(-1)
        y = torch.nn.functional.pad(y, (0, pad_len), mode='constant', value=0)
        y_hat = torch.nn.functional.pad(y_hat, (0, pad_len), mode='constant', value=0)

        # Step 2: Subtract the means to simplify the calculation
        y = y - y.mean(dim=-1, keepdim=True)
        y_hat = y_hat - y_hat.mean(dim=-1, keepdim=True)

        # Step 3: Compute the FFT of both signals
        F_y = fft.fft(y)
        F_y_hat = fft.fft(y_hat)
        
        # Step 4: Multiply by the conjugate and apply the bandpass filter
        freqs = fft.fftfreq(F_y.size(-1), d=1/self.sampling_rate)
        bandpass_mask = self.bandpass_filter(freqs).to(F_y.device)
        
        cross_spectrum = F_y * torch.conj(F_y_hat)
        cross_spectrum_filtered = cross_spectrum * bandpass_mask

        # Step 5: Compute the inverse FFT to get cross-correlation
        cross_corr = fft.ifft(cross_spectrum_filtered).real

        # Step 6: Normalize by the product of the standard deviations
        sigma_y = torch.std(y, dim=-1, keepdim=True)
        sigma_y_hat = torch.std(y_hat, dim=-1, keepdim=True)
        norm_cross_corr = cross_corr / (sigma_y * sigma_y_hat)

        # Step 7: Find the maximum cross-correlation
        mcc = torch.max(norm_cross_corr, dim=-1).values

        # Step 8: Scale by the power ratio (c_pr)
        power_y = torch.sum(torch.abs(F_y)**2, dim=-1)
        power_y_hat = torch.sum(torch.abs(F_y_hat)**2, dim=-1)
        c_pr = torch.sum(bandpass_mask) / (torch.sum(power_y) * torch.sum(power_y_hat))

        return -c_pr * mcc.mean()  # Negative for minimizing

class TALOSLoss(nn.Module):
    def __init__(self, sampling_rate, participants, max_shift_frames=None):
        """
        Initialize the TALOS loss function.
        
        Parameters:
        - sampling_rate: The video frame rate (FPS).
        - participants: participants (for per-participant latent shift distribution).
        - max_shift_frames: Maximum possible frame shift (optional).
        """
        super(TALOSLoss, self).__init__()
        self.sampling_rate = sampling_rate
        if max_shift_frames is None:
            self.max_shift_frames = sampling_rate // 2  # Default max shift range is +/- half a second.
        else:
            self.max_shift_frames = max_shift_frames

        # Latent variable for shifts per participant (initialized as zeros for all participants)
        self.theta_s = nn.ParameterDict({
            participant: nn.Parameter(torch.ones(self.max_shift_frames * 2 + 1) / (self.max_shift_frames * 2 + 1))
            for participant in participants
        })
        for key in self.theta_s:
            self.theta_s[key] = self.theta_s[key].to(torch.device("cuda:0")) #get this from config


    def forward(self, y_pred, y_true, participant_ids, validation=False):
        """
        Compute the TALOS loss.
        
        Parameters:
        - y_pred: The predicted signal (batch, time).
        - y_true: The ground truth signal (batch, time).
        - participant_ids: List or tensor of participant IDs corresponding to each sample in the batch.
        - validation: Whether we are in validation mode (default: False).
        
        Returns:
        - The TALOS loss value.
        """
        # print(participant_ids)
        batch_size, T = y_pred.shape
        num_shifts = self.max_shift_frames * 2 + 1

        # Create all possible shifted versions of the ground truth
        shifts = torch.arange(-self.max_shift_frames, self.max_shift_frames + 1)
        shifted_ground_truths = []
        for shift in shifts:
            if shift < 0:
                pad = (-shift, 0)
                y_shifted = torch.nn.functional.pad(y_true, pad, mode='constant', value=0)[:, :T]
            else:
                pad = (0, shift)
                y_shifted = torch.nn.functional.pad(y_true, pad, mode='constant', value=0)[:, shift:]
            shifted_ground_truths.append(y_shifted)
        
        shifted_ground_truths = torch.stack(shifted_ground_truths, dim=0)  # Shape: (num_shifts, batch, T)
        
        # Compute MSE for each shift
        mse_per_shift = torch.mean((y_pred.unsqueeze(0) - shifted_ground_truths) ** 2, dim=-1)  # Shape: (num_shifts, batch)

        # Validation Mode: Minimize over all shifts
        if validation:
            talos_loss = torch.min(mse_per_shift, dim=0).values.mean()  # Minimize over shifts, average over batch
        else:
            # Training Mode: Compute the per-participant softmax distribution over the shifts
            talos_loss = 0
            for i, participant_id in enumerate(participant_ids):
                # print(self.theta_s[participant_id[4:8]].get_device())
                p_k = torch.nn.functional.softmax(self.theta_s[participant_id[4:8]], dim=0)  # Shape: (num_shifts,)
                # print(p_k.get_device())
                # print(mse_per_shift.get_device())
                p_k = p_k.to(torch.device("cuda:0"))
                # print(p_k.get_device())
                # Compute the weighted MSE by the shift probability for the current participant
                talos_loss += torch.sum(mse_per_shift[:, i] * p_k)
            
            talos_loss /= batch_size  # Average over the batch

        return talos_loss