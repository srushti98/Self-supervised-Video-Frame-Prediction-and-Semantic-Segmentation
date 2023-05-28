import os
import gzip
import random
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torch.utils.data import random_split, Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True

def extract_num(s):
    return int(s.split('_')[1])

def extract_image_num(s):
    return int(s.split('_')[1].split('.')[0])

class MovingObjectDataSet(data.Dataset):
    def __init__(self, root, is_train=True, n_frames_input=11, n_frames_output=11, transform=None):
        super(MovingObjectDataSet, self).__init__()

        self.videos = []
        unlabelled_dirs = os.listdir(root)
        unlabelled_dirs = sorted(unlabelled_dirs, key=extract_num)

        for video in unlabelled_dirs:
            self.videos.extend([root + '/' + video + '/'])
        
        self.length = len(self.videos)

        self.is_train = is_train

        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        self.mean = 0
        self.std = 1

    def __getitem__(self, index):
        length = self.n_frames_input + self.n_frames_output
        # print(self.videos[index])
        video_folder = os.listdir(self.videos[index])
        video_folder = sorted(video_folder, key=extract_image_num)
        imgs = []
        for image in video_folder:
            imgs.append(np.array(Image.open(self.videos[index] + '/' + image)))

        #shape: torch.Size([10, 1, 64, 64])
        # print(len(imgs))

        past_clips = imgs[0:self.n_frames_input] #[11,160,240,3]
        future_clips = imgs[-self.n_frames_output:] #[11,160,240,3]

        past_clips = [torch.from_numpy(clip) for clip in past_clips]
        future_clips = [torch.from_numpy(clip) for clip in future_clips]
        # stack the tensors and permute the dimensions


        past_clips = torch.stack(past_clips).permute(0, 3, 1, 2)
        future_clips = torch.stack(future_clips).permute(0, 3, 1, 2)
        #we want [11,3,160,240]
        return (past_clips).contiguous().float(), (future_clips).contiguous().float()
    def __len__(self):
        return self.length

def load_moving_object(batch_size, val_batch_size,data_root, num_workers):

    whole_data = MovingObjectDataSet(root=data_root, is_train=True, n_frames_input=11, n_frames_output=11)

    train_size = int(0.9 * len(whole_data))
    val_size = int(0.09 * len(whole_data))
    test_size = int(0.01 * len(whole_data))
    print(train_size, val_size, test_size)
    train_data, val_data, test_data = random_split(whole_data, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(2021))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    mean, std = 0, 1
    return train_loader, val_loader, test_loader, mean, std
