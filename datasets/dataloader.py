import os
import random
import pickle as pkl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
            'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
            'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
            'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
            'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
            'Clapping']


def ids_to_multinomial(ids):
    """ label encoding
    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    id_to_idx = {id: index for index, id in enumerate(categories)}

    y = np.zeros(len(categories))
    for id in ids:
        index = id_to_idx[id]
        y[index] = 1
    return y


class GTLoader(object):
    def __init__(self, args, label):
        super(GTLoader, self).__init__()

        # load annotations
        self.df = pd.read_csv(label, header=0, sep='\t')
        self.df_a = pd.read_csv(args.eval_audio, header=0, sep='\t')
        self.df_v = pd.read_csv(args.eval_visual, header=0, sep='\t')

    def load(self, batch_idx):
        df, df_a, df_v = self.df, self.df_a, self.df_v 
        id_to_idx = {id: index for index, id in enumerate(categories)}

        # extract audio GT labels
        GT_a = np.zeros((25, 10))

        df_vid_a = df_a.loc[df_a['filename'] == df.loc[batch_idx, :][0]]
        filenames = df_vid_a["filename"]
        events = df_vid_a["event_labels"]
        onsets = df_vid_a["onset"]
        offsets = df_vid_a["offset"]
        num = len(filenames)
        if num > 0:
            for i in range(num):
                x1 = int(onsets[df_vid_a.index[i]])
                x2 = int(offsets[df_vid_a.index[i]])
                event = events[df_vid_a.index[i]]
                idx = id_to_idx[event]
                GT_a[idx, x1:x2] = 1

        # extract visual GT labels
        GT_v = np.zeros((25, 10))

        df_vid_v = df_v.loc[df_v['filename'] == df.loc[batch_idx, :][0]]
        filenames = df_vid_v["filename"]
        events = df_vid_v["event_labels"]
        onsets = df_vid_v["onset"]
        offsets = df_vid_v["offset"]
        num = len(filenames)
        if num > 0:
            for i in range(num):
                x1 = int(onsets[df_vid_v.index[i]])
                x2 = int(offsets[df_vid_v.index[i]])
                event = events[df_vid_v.index[i]]
                idx = id_to_idx[event]
                GT_v[idx, x1:x2] = 1

        GT_av = GT_a * GT_v

        return GT_a, GT_v, GT_av


class LLP(Dataset):

    def __init__(self, args, label, data_percent=1.0, use_tail=False):
        df = pd.read_csv(label, header=0, sep='\t')
        if 'search' in args.mode and use_tail:
            self.df = df[int(data_percent * len(df)):]
        else:
            self.df = df[:int(data_percent * len(df))]
        
        self.filenames = self.df['filename']
        self.audio_dir = args.audio_dir
        self.video_dir = args.video_dir
        self.st_dir = args.st_dir

        self.need_to_remove = (args.mode == 'research_ma' or args.mode == 'retrain_ma')
        if self.need_to_remove:
            assert args.label_ma is not None, 'Error, label_ma must be provided if in retrain mode.'
            self.need_to_remove_v, self.need_to_remove_a = pkl.load(open(args.label_ma, 'rb'))
            print('Loaded modality-aware labels from', args.label_ma)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row[0][:11]
        audio = np.load(os.path.join(self.audio_dir, name + '.npy'))
        video_s = np.load(os.path.join(self.video_dir, name + '.npy'))
        video_st = np.load(os.path.join(self.st_dir, name + '.npy'))
        ids = row[-1].split(',')
        label = ids_to_multinomial(ids)

        # label smoothing
        a = 1.0
        v = 0.9
        pa = a * label + (1 - a) * 0.5
        pv = v * label + (1 - v) * 0.5

        # We change modality-aware label here
        if self.need_to_remove:
            for c in range(len(categories)):
                if label[c] != 0:
                    if idx in self.need_to_remove_v[c]:
                        pv[c] = 0
                    if idx in self.need_to_remove_a[c]:
                        pa[c] = 0

        sample = {'audio': audio, 'video_s': video_s, 'video_st': video_st, 
                  'label': label, 'pa': pa, 'pv': pv}

        return sample



class EstimateLLP(Dataset):

    def __init__(self, args, label):
        self.df = pd.read_csv(label, header=0, sep='\t')
        self.filenames = self.df['filename']
        self.audio_dir = args.audio_dir
        self.video_dir = args.video_dir
        self.st_dir = args.st_dir

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        #! read sample 1
        row = self.df.iloc[idx]
        name = row[0][:11]
        audio = np.load(os.path.join(self.audio_dir, name + '.npy'))
        video_s = np.load(os.path.join(self.video_dir, name + '.npy'))
        video_st = np.load(os.path.join(self.st_dir, name + '.npy'))
        ids = row[-1].split(',')
        label = ids_to_multinomial(ids)

        # label smoothing
        a = 1.0
        v = 0.9 
        pa = a * label + (1 - a) * 0.5
        pv = v * label + (1 - v) * 0.5

        #! read sample 2
        while True:
            idx2 = random.randint(0, len(self.filenames)-1)
            row = self.df.loc[idx2, :]
            name = row[0][:11]
            ids = row[-1].split(',')
            label2 = ids_to_multinomial(ids)              
            intersection = np.logical_and(label, label2)  
            intersection = intersection.astype(int).sum() 
            if intersection == 0:
                break

        audio2 = np.load(os.path.join(self.audio_dir, name + '.npy'))
        video_s2 = np.load(os.path.join(self.video_dir, name + '.npy'))
        video_st2 = np.load(os.path.join(self.st_dir, name + '.npy'))
        ids = row[-1].split(',')
        label2 = ids_to_multinomial(ids)

        sample = {'audio': audio, 'video_s': video_s, 'video_st': video_st, 
                  'audio2': audio2, 'video_s2': video_s2, 'video_st2': video_st2, 
                  'label': label, 'label2': label2, 'idx': idx, 'idx2': idx2,
                  'pa': pa, 'pv': pv}

        return sample


