import torch
from tSNE import batch_tsne


def avce_test(dataloader, model_av, model_v, gt, e):
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
            av_logits = torch.mean(av_logits, 0)      # 5-crop
            pred = torch.cat((pred, av_logits))
            '''
'''
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

        if any(pred == 1 for pred in pred_binary):
            message= "El video contiene violencia"
            message_frames = "En un rango de [0-"+ str(len(pred_binary) - 1) +"], los frames con violencia son: "

            start_idx = None
            for i, pred in enumerate(pred_binary):
                if pred == 1:
                    if start_idx is None:
                        start_idx = i
                elif start_idx is not None:
                    message_frames += ("[" + str(start_idx) + " - " + str(i - 1) + "]" + ", ") if i-1 != start_idx else ("[" + str(start_idx) + "], ")
                    start_idx = None

            if start_idx is not None:
                message_frames += ("[" + str(start_idx) + " - " + str(len(pred_binary) - 1) + "]") if len(pred_binary) - 1 != start_idx else ("[" + str(start_idx) + "]")
            else:
                message_frames = message_frames[:-2]              

        else:
            message= "El video no contiene violencia"
            message_frames = "No hay frames con violencia"            

        return message, message_frames

