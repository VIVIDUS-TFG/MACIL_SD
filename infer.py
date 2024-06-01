from torch.utils.data import DataLoader
import torch
import numpy as np
from avce_network import AVCE_Model as Model
from avce_dataset import Dataset
from test import avce_test as test
import option
import time

if __name__ == '__main__':
    args = option.parser.parse_args()
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=5, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    model = Model(args)
    model = model.cuda()
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('ckpt/macil_sd.pkl').items()})
    gt = np.load(args.gt)
    st = time.time()
    message, message_second, message_frames = test(test_loader, model, None, gt, 0)
    time_elapsed = time.time() - st
    print(message + message_frames)
    print(message + message_second)
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
