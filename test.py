import torch
from tSNE import batch_tsne
import csv
import numpy as np
import os

def avce_test(dataloader, model_av, model_v, gt, e, args):
    with torch.no_grad():
        model_av.eval()
        pred = torch.zeros(0).cuda()
        if model_v is not None:
            model_v.eval()
            pred3 = torch.zeros(0).cuda()
        cur_index = 0
        for i, (f_v, f_a) in enumerate(dataloader):
            f_v, f_a = f_v.cuda(), f_a.cuda()
            _, _, _, av_logits, audio_rep, visual_rep = model_av(f_a, f_v, seq_len=None)
            av_logits = torch.squeeze(av_logits)
            av_logits = torch.sigmoid(av_logits)
            if av_logits.dim() > 1: # 5-crop
                av_logits = torch.mean(av_logits, 0)
            pred = torch.cat((pred, av_logits))

            visual_rep = torch.mean(visual_rep, 0)
            audio_rep = torch.mean(audio_rep, 0)
            if i == 10000:
                visual_rep = list(visual_rep.cpu().detach().numpy())
                audio_rep = list(audio_rep.cpu().detach().numpy())
                cur_gt = list(gt)[cur_index:cur_index+len(audio_rep)*16]
                cur_gt = cur_gt[::16]
                cur_index += len(audio_rep)*16
                batch_tsne(visual_rep, cur_gt, e, i, 'fig/visual/')
                batch_tsne(audio_rep, cur_gt, e, i, 'fig/audio/')

            if model_v is not None:
                v_logits = model_v(f_v, seq_len=None)
                v_logits = torch.squeeze(v_logits)
                v_logits = torch.sigmoid(v_logits)
                v_logits = torch.mean(v_logits, 0)
                pred3 = torch.cat((pred3, v_logits))

        pred = list(pred.cpu().detach().numpy())
        pred_binary = [1 if pred_value > 0.35 else 0 for pred_value in pred]

        video_duration = int(np.ceil(len(pred_binary) * 0.96)) # len(pred_binary) = video_duration / 0.96

        if any(pred == 1 for pred in pred_binary):
            message= "El video contiene violencia. "
            message_second = "Los intervalos con violencia son: "
            message_frames = "En un rango de [0-"+ str(len(pred_binary) - 1) +"] los frames con violencia son: "

            start_idx = None
            for i, pred in enumerate(pred_binary):
                if pred == 1:
                    if start_idx is None:
                        start_idx = i
                elif start_idx is not None:
                    message_frames += ("[" + str(start_idx) + " - " + str(i - 1) + "]" + ", ") if i-1 != start_idx else ("[" + str(start_idx) + "], ")
                    message_second += ("[" + parse_time(int(np.floor((start_idx + 1)* 0.96))) + " - " + parse_time(int(np.ceil(i * 0.96))) + "], ")
                    start_idx = None

            if start_idx is not None:
                message_frames += ("[" + str(start_idx) + " - " + str(len(pred_binary) - 1) + "]") if len(pred_binary) - 1 != start_idx else ("[" + str(start_idx) + "]")
                message_second += ("[" + parse_time(int(np.floor((start_idx + 1) * 0.96))) + " - " + parse_time(video_duration) + "]")
            else:
                message_frames = message_frames[:-2]              
                message_second = message_second[:-2]              

        else:
            message= "El video no contiene violencia."
            message_frames = ""            
            message_second = ""            

        if args.evaluate == 'true':
            # Create a list of dictionaries to store the data
            data = []
            data.append({
                'video_id': "IDVIDEO",
                'frame_number': pred_binary,
                "violence_label": "1" if any(pred == 1 for pred in pred_binary) else "0",
            })

            # Write the data to a CSV file
            csv_file = 'inference.csv'

            fieldnames = ['video_id', 'frame_number', 'violence_label']
            file_exists = os.path.isfile(csv_file)

            with open(csv_file, 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(data)
            
        return message, message_second, message_frames

def parse_time(seconds):
    seconds = max(0, seconds)
    sec = seconds % 60
    if sec < 10:
        sec = "0" + str(sec)
    else:
        sec = str(sec)
    return str(seconds // 60) + ":" + sec

