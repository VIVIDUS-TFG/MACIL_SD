import torch
from tSNE import batch_tsne
import csv
import numpy as np
import os

def avce_test(dataloader, model_av, model_v, gt, e, args):
    with torch.no_grad():
        model_av.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pred = torch.zeros(0).to(device)
        if model_v is not None:
            model_v.eval()
        
        for i, (f_v, f_a) in enumerate(dataloader):
            f_v, f_a = f_v.to(device), f_a.to(device)
            _, _, _, av_logits, audio_rep, visual_rep = model_av(f_a, f_v, seq_len=None)
            av_logits = torch.squeeze(av_logits)
            av_logits = torch.sigmoid(av_logits)

            pred = torch.cat((pred, av_logits))

        pred = list(pred.cpu().detach().numpy())
        pred_binary = [1 if pred_value > 0.35 else 0 for pred_value in pred]
    return pred_binary

def save_results(results, filename):
    np.save(filename, results)
    
def parse_time(seconds):
    seconds = max(0, seconds)
    sec = seconds % 60
    if sec < 10:
        sec = "0" + str(sec)
    else:
        sec = str(sec)
    return str(seconds // 60) + ":" + sec

