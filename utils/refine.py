import torch
import pickle
import numpy as np
import pandas as pd
from datasets.dataloader_avvp import categories


def get_modality_aware_label(args, datas, v_accs, a_accs, logger=None):

    v_err = 1 - torch.Tensor(v_accs)    # [25]
    a_err = 1 - torch.Tensor(a_accs)    # [25]

    v_class = v_err / torch.max(v_err)  # E_c
    a_class = a_err / torch.max(a_err)  # E_c
   
    need_to_remove_v = [[] for _ in range(25) ]     # [25] 
    need_to_remove_a = [[] for _ in range(25) ]
    total_a = 0
    total_v = 0
    changed_a = 0
    changed_v = 0
    for data in datas:  # n_steps
        a = data['a']
        v = data['v']
        a_v = data['a_v']
        v_v = data['v_v']
        a_a = data['a_a']
        v_a = data['v_a']
    
        label = data['label']   # [B, 25]
        idx = data['idx']
    
        a = a * label
        v = v * label
        a_v = a_v * label
        v_v = v_v * label
        a_a = a_a * label
        v_a = v_a * label
    
        for b in range(len(a)):
          for c in range(25):
            if label[b][c] != 0:
                if v_a[b][c] / v_class[c] < 0.5 and a_a[b][c] / v_class[c] < 0.5:
                    # visual is not correct, given original visual data is input.
                    need_to_remove_v[c].append(idx[b])

                if a_v[b][c]/a_class[c] < 0.5 and v_v[b][c] /a_class[c] < 0.5:
                    need_to_remove_a[c].append(idx[b])
       
    filepath = f'{args.result_dir}/ma_labels.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump([need_to_remove_v, need_to_remove_a], f)

    logger.info('Saved modality aware labels to: {}'.format(filepath))
    

def get_denoise_label(args, datas, a_prob_mean, v_prob_mean, logger=None):

    a_thres = 0.6
    v_thres = 1.8
    noise_num_v = np.zeros(25)
    noise_num_a = np.zeros(25)
    for data in datas:        # iters
        a = data['a']
        v = data['v']
        label = data['label']
        Pa = data['Pa']
        Pv = data['Pv']

        a = a * Pa
        v = v * Pv
        for b in range(len(a)): # batch size
            for c in range(25):
                if label[b][c] != 0:
                    if v[b][c] / v_prob_mean[c] < v_thres:
                        noise_num_v[c] += 1
                    if a[b][c] / a_prob_mean[c] < a_thres:
                        noise_num_a[c] += 1

    event_nums = np.zeros(25)
    labels = pd.read_csv(args.label_train, header=0, sep='\t')['event_labels'].values
    id_to_idx = {id: index for index, id in enumerate(categories)}
    for video_id, label in enumerate(labels):
        ls = label.split(',')
        label_id = [id_to_idx[l] for l in ls]
        for id in label_id:
            event_nums[id] += 1

    v_noise_ratio = np.divide(noise_num_v, event_nums)
    a_noise_ratio = np.divide(noise_num_a, event_nums)
    filepath = f'{args.result_dir}/noise_ratios.npz'
    np.savez(filepath, audio=a_noise_ratio, visual=v_noise_ratio)

    logger.info('Saved noise ratios to: {}'.format(filepath))
