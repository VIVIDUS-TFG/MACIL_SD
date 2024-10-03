import os
from torch.utils.data import DataLoader
import torch
import numpy as np
from avce_network import AVCE_Model as Model
from avce_dataset import Dataset
from test import avce_test as test, save_results
import option
import time

if __name__ == '__main__':
    args = option.parser.parse_args()
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    model = Model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if device.type == 'cpu':
        model_dict = model.load_state_dict(
             {k.replace('module.', ''): v for k, v in torch.load('ckpt/macil_sd.pkl', map_location=torch.device('cpu')).items()})
    else:
        model_dict = model.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load('ckpt/macil_sd.pkl').items()})
    
    gt = np.load(args.gt)
    st = time.time()
    results = test(test_loader, model, None, gt, 0, args)
    save_results(results, os.path.join(args.output_path, 'results.npy'))
