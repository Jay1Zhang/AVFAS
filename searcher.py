import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dataloader import LLP, EstimateLLP, GTLoader
from utils.metrics import AVVP_Evaluator
from utils.refine import get_modality_aware_label, get_denoise_label


class AVVPSearcher():

    def __init__(self, args):
        self.args = args

        if 'search' in args.mode:
            data_percent = 0.9
            dataset_train = LLP(args, args.label_train, data_percent=data_percent, use_tail=False)
            dataset_dev = LLP(args, args.label_train, data_percent=data_percent, use_tail=True)

            # dataset_train = LLP(args, args.label_train)   
            # dataset_dev = LLP(args, args.label_train)
        else:
            dataset_train = LLP(args, args.label_train)   
            dataset_dev = LLP(args, args.label_val)

        dataset_val = LLP(args, args.label_test)    # label_val
        dataset_test = LLP(args, args.label_test)

        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, num_replicas=args.world_size, rank=args.dist_rank) if args.parallel else None
        train_loader = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                shuffle=(train_sampler is None),
                                                sampler=train_sampler)

        dev_sampler = torch.utils.data.distributed.DistributedSampler(dataset_dev, num_replicas=args.world_size, rank=args.dist_rank) if args.parallel else None
        dev_loader = torch.utils.data.DataLoader(dataset_dev,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             shuffle=(dev_sampler is None),
                                             sampler=dev_sampler)

        val_loader = torch.utils.data.DataLoader(dataset_val,
                                            batch_size=1,
                                            num_workers=args.num_workers,
                                            shuffle=False,
                                            pin_memory=True)

        test_loader = torch.utils.data.DataLoader(dataset_test,
                                            batch_size=1,
                                            num_workers=args.num_workers,
                                            shuffle=False,
                                            pin_memory=True)                    

        self.dataloaders = {'train': train_loader, 'dev': dev_loader, 'val': val_loader, 'test': test_loader}

        if 'estimate' in args.mode:
            dataset_estimate = EstimateLLP(args, args.label_train)
            estimate_loader = torch.utils.data.DataLoader(dataset_estimate,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                shuffle=True)
            self.dataloaders['estimate'] = estimate_loader

    def search_epoch(self, epoch, model, architect, optimizer, scheduler, logger):
        args = self.args
        train_loader = self.dataloaders['train']
        dev_loader = self.dataloaders['dev']

        model.train()
        if args.parallel:
            train_loader.sampler.set_epoch(epoch)
            dev_loader.sampler.set_epoch(epoch)

        epoch_loss_trn, epoch_lr_trn = 0., 0.
        epoch_loss_dev, epoch_lr_dev = 0., 0.
        with tqdm(train_loader) as t:
            for train_inputs, dev_inputs in zip(train_loader, dev_loader):
                
                # update alphas
                loss_dev, lr_dev = architect.step(dev_inputs)

                # update weights
                optimizer.zero_grad()
                loss_trn, outputs = model(train_inputs)
                loss_trn.backward()
                optimizer.step()
                lr_trn = optimizer.param_groups[0]['lr']

                # log
                if args.log:
                    t.set_postfix_str(
                        'loss_trn: {:.04f}, lr_trn: {:.08f}; loss_dev: {:.04f}, lr_dev: {:.08f}'.format(loss_trn.item(), lr_trn, loss_dev.item(), lr_dev)
                    )
                    t.update()

                epoch_loss_trn += loss_trn.item()
                epoch_lr_trn += lr_trn
                epoch_loss_dev += loss_dev.item()
                epoch_lr_dev += lr_dev

        epoch_loss_trn = epoch_loss_trn / len(train_loader)
        epoch_lr_trn = epoch_lr_trn / len(train_loader)
        epoch_loss_dev = epoch_loss_dev / len(dev_loader)
        epoch_lr_dev = epoch_lr_dev / len(dev_loader)
        scheduler.step()
     
        genotype, arch_params = model.module.genotype() if args.parallel else model.genotype()        
        if args.log:
            logger.info('[Search Epoch: {}]\t\t||\tTrain Loss: {:.4f}\t|\tLearning Rate: {:.8f}|\tDev Loss: {:.4f}\t|\tArch Learning Rate: {:.8f}'.format(epoch, epoch_loss_trn, epoch_lr_trn, epoch_loss_dev, epoch_lr_dev))
            logger.info('[Architecture] {}'.format(genotype))
            if args.tb_writer is not None:
                args.tb_writer.add_scalar(f'search/train_loss', epoch_loss_trn, epoch)
                args.tb_writer.add_scalar(f'search/train_lr', epoch_lr_trn, epoch)
                args.tb_writer.add_scalar(f'search/dev_loss', epoch_loss_dev, epoch)
                args.tb_writer.add_scalar(f'search/dev_lr', epoch_lr_dev, epoch)

        return genotype, arch_params


    def train_epoch(self, epoch, model, optimizer, scheduler, logger):
        args = self.args
        dataloader = self.dataloaders['train']

        model.train()
        if args.parallel:
            dataloader.sampler.set_epoch(epoch)

        epoch_loss, epoch_lr = 0., 0.
        with tqdm(dataloader) as t:
            for inputs in dataloader:
                # forward
                loss, outputs = model(inputs)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr = optimizer.param_groups[0]['lr']
                if args.log:
                    t.set_postfix_str(
                        'loss: {:.04f}, lr: {:.08f}'.format(loss.item(), lr)
                    )
                    t.update()

                epoch_loss += loss.item()
                epoch_lr += lr

        epoch_loss = epoch_loss / len(dataloader)
        epoch_lr = epoch_lr / len(dataloader)
        scheduler.step()
        
        if args.log:
            logger.info('[Train Epoch: {}]\t\t||\tTrain Loss: {:.4f}\t|\tLearning Rate: {:.8f}'.format(epoch, epoch_loss, epoch_lr))
            if args.tb_writer is not None:
                args.tb_writer.add_scalar(f'train/train_loss', epoch_loss, epoch)
                args.tb_writer.add_scalar(f'train/train_lr', epoch_lr, epoch)
                    
        return epoch_loss


    def dev_epoch(self, epoch, model, architect, logger):
        args = self.args
        dataloader = self.dataloaders['dev']
        
        model.train()
        if args.parallel:
            dataloader.sampler.set_epoch(epoch)

        epoch_loss, epoch_lr = 0., 0.
        with tqdm(dataloader) as t:
            for inputs in dataloader:
                
                loss, lr = architect.step(inputs)
                if args.log:
                    t.set_postfix_str(
                        'loss: {:.04f}, lr: {:.08f}'.format(loss.item(), lr)
                    )
                    t.update()

                epoch_loss += loss.item()
                epoch_lr += lr

        epoch_loss = epoch_loss / len(dataloader)
        epoch_lr = epoch_lr / len(dataloader)
        genotype, arch_params = model.module.genotype() if args.parallel else model.genotype()
        
        if args.log:
            logger.info('[Dev Epoch: {}]\t\t||\tDev Loss: {:.4f}\t|\tLearning Rate: {:.8f}'.format(epoch, epoch_loss, epoch_lr))
            logger.info('[Architecture] {}'.format(genotype))
            if args.tb_writer is not None:
                args.tb_writer.add_scalar(f'train/dev_loss', epoch_loss, epoch)
                args.tb_writer.add_scalar(f'train/dev_lr', epoch_lr, epoch)
                
        return epoch_loss, genotype, arch_params


    def val_epoch(self, epoch, model, logger):
        args = self.args
        dataloader = self.dataloaders['test']
        
        model.eval()
        epoch_loss = 0.0
        gt_loader = GTLoader(args, args.label_test)
        evaluator = AVVP_Evaluator()

        with tqdm(dataloader) as t:
            #! batch size equals 1
            for batch_idx, inputs in enumerate(dataloader):
                # forward
                loss, outputs = model(inputs)

                #! obtain event classification probability
                o = (outputs['global_prob'].cpu().detach().numpy() >= 0.5).astype(np.int_)
                Pa = outputs['frame_prob'][0, :, 0, :].cpu().detach().numpy()
                Pv = outputs['frame_prob'][0, :, 1, :].cpu().detach().numpy()
                # filter out false positive events with predicted weak labels
                Pa = (Pa >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)
                Pv = (Pv >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)
                # get prediction matrices
                SO_a = np.transpose(Pa)
                SO_v = np.transpose(Pv)
                SO_av = SO_a * SO_v

                #! obtain ground truth label
                GT_a, GT_v, GT_av = gt_loader.load(batch_idx)

                #! calculate metrics
                evaluator.calc_f1(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
                f1_seg_avg = evaluator.stats_segment_f1()[-2]
                f1_eve_avg = evaluator.stats_event_f1()[-2]

                if args.log:
                    t.set_postfix_str(
                        'loss: {:.04f}, segment-level avg@type: {:.04f}, event-level avg@type: {:.04f}'.format(loss.item(), f1_seg_avg, f1_eve_avg)
                    )
                    t.update()
        
                epoch_loss += loss.item()

        # log info
        epoch_loss = epoch_loss / len(dataloader)
        f1_seg_a, f1_seg_v, f1_seg_av, f1_seg_avg_type, f1_seg_avg_event = evaluator.stats_segment_f1()
        f1_eve_a, f1_eve_v, f1_eve_av, f1_eve_avg_type, f1_eve_avg_event = evaluator.stats_event_f1()
        if args.log:
            logger.info('[Val Epoch: {}]\t\t||\tVal Loss: {:.4f}'.format(epoch, epoch_loss))
            logger.info('[Segment-Level F1 Score]\t||\tAudio: {:.4f}\t｜\tVisual: {:.4f}\t｜\tAudio-Visual: {:.4f}\t｜\tType@Avg: {:.4f}\t|\tEvent@Avg: {:.4f}'.format(f1_seg_a, f1_seg_v, f1_seg_av, f1_seg_avg_type, f1_seg_avg_event))
            logger.info('[Event-Level F1 Score]\t||\tAudio: {:.4f}\t｜\tVisual: {:.4f}\t｜\tAudio-Visual: {:.4f}\t｜\tType@Avg: {:.4f}\t|\tEvent@Avg: {:.4f}'.format(f1_eve_a, f1_eve_v, f1_eve_av, f1_eve_avg_type, f1_eve_avg_event))
            
            if args.tb_writer is not None:
                args.tb_writer.add_scalar(f'train/val_loss', epoch_loss, epoch)

                args.tb_writer.add_scalar(f'segment/1-audio', f1_seg_a, epoch)
                args.tb_writer.add_scalar(f'segment/2-visual', f1_seg_v, epoch)
                args.tb_writer.add_scalar(f'segment/3-audio-visual', f1_seg_av, epoch)
                args.tb_writer.add_scalar(f'segment/4-type@avg', f1_seg_avg_type, epoch)

                args.tb_writer.add_scalar(f'event/1-audio', f1_eve_a, epoch)
                args.tb_writer.add_scalar(f'event/2-visual', f1_eve_v, epoch)
                args.tb_writer.add_scalar(f'event/3-audio-visual', f1_eve_av, epoch)
                args.tb_writer.add_scalar(f'event/4-type@avg', f1_eve_avg_type, epoch)
                   
        return f1_seg_avg_type, f1_eve_avg_type


    def search(self, model, architect, optimizer, scheduler, logger):
        args = self.args

        best_epoch, best_f1_seg, best_f1_eve = 0, 0, 0
        best_genotype = None
        for epoch in range(args.epochs):

            if args.log:
                logger.info("-" * 50)
                logger.info('Search Epoch: {}'.format(epoch))            
                logger.info('EXP Log at: {}'.format(args.result_dir))
            
            # #! search phase
            # genotype, arch_params = self.search_epoch(epoch, model, architect, optimizer, scheduler, logger) 
   
            # genotype, arch_params = model.module.genotype() if args.parallel else model.genotype()

            #! train phase
            with torch.set_grad_enabled(True):
                # train_loss = self.train_epoch(epoch, model, optimizer, scheduler, logger) 
                if args.mode == 'research_denoise':
                    assert args.label_denoise is not None, 'Error, the noise ratios file must be provided.'
                    train_loss = self.train_denoise_epoch(epoch, model, optimizer, scheduler, logger) 
                else:
                    train_loss = self.train_epoch(epoch, model, optimizer, scheduler, logger) 
       
            #! dev phase
            dev_loss, genotype, arch_params = self.dev_epoch(epoch, model, architect, logger)

            #! val phase
            with torch.no_grad():
                f1_seg, f1_eve = self.val_epoch(epoch, model, logger)
        
            #! log
            if args.log and f1_seg > best_f1_seg:
                # save model.pt, genotype.pkl, architect.pdf
                torch.save(model.state_dict(), f'{args.result_dir}/best/model.pt')
                with open(f'{args.result_dir}/best/genotype.pkl', 'wb') as geno_file:
                    pickle.dump(genotype, geno_file)
                logger.info('[Saved Model] Epoch: {}, current segment-level avg@type f1 score: {:.4f}, best f1 score: {:.4f}'.format(epoch, f1_seg, best_f1_seg))
                best_epoch = epoch
                best_f1_seg = f1_seg
                best_f1_eve = f1_eve
                best_genotype = genotype

                with open(f'{args.result_dir}/best/arch_params.json', 'w', newline='\n') as f:
                    data = json.dumps(arch_params, indent=1)
                    f.write(data)

        return best_epoch, best_f1_seg, best_f1_eve, best_genotype


    def train(self, model, optimizer, scheduler, logger):
        args = self.args

        best_epoch, best_f1_seg, best_f1_eve = 0, 0, 0
        for epoch in range(args.epochs):

            if args.log:
                logger.info("-" * 50)
                logger.info('Train Epoch: {}'.format(epoch))            
                logger.info('EXP Log at: {}'.format(args.result_dir))
            
            #! train phase
            with torch.set_grad_enabled(True):
                if args.mode == 'retrain_denoise':
                    assert args.label_denoise is not None, 'Error, the noise ratios file must be provided.'
                    train_loss = self.train_denoise_epoch(epoch, model, optimizer, scheduler, logger) 
                else:
                    train_loss = self.train_epoch(epoch, model, optimizer, scheduler, logger) 
   
            #! val phase
            with torch.no_grad():
                f1_seg, f1_eve = self.val_epoch(epoch, model, logger)
        
            #! log
            # if args.log and f1_seg > best_f1_seg:
            if args.log and f1_eve > best_f1_eve:
                torch.save(model.state_dict(), f'{args.result_dir}/best/model.pt')
                logger.info('[Saved Model] Epoch: {}, current segment-level avg@type f1 score: {:.4f}, best f1 score: {:.4f}'.format(epoch, f1_seg, best_f1_seg))
                best_epoch = epoch
                best_f1_seg = f1_seg
                best_f1_eve = f1_eve

        genotype = model.module.genotype() if args.parallel else model.genotype()
        with open(f'{args.result_dir}/best/genotype.pkl', 'wb') as geno_file:
            pickle.dump(genotype, geno_file)

        return best_epoch, best_f1_seg, best_f1_eve


    def eval(self, model, logger):
        args = self.args

        model.eval()
        dataloader = self.dataloaders['test']
        gt_loader = GTLoader(args, args.label_test)
        evaluator = AVVP_Evaluator()

        with torch.no_grad():
            with tqdm(dataloader) as t:
                #! batch size equals 1
                for batch_idx, inputs in enumerate(dataloader):
                    # forward
                    loss, outputs = model(inputs)

                    #! obtain event classification probability
                    o = (outputs['global_prob'].cpu().detach().numpy() >= 0.5).astype(np.int_)
                    Pa = outputs['frame_prob'][0, :, 0, :].cpu().detach().numpy()
                    Pv = outputs['frame_prob'][0, :, 1, :].cpu().detach().numpy()
                    # filter out false positive events with predicted weak labels
                    Pa = (Pa >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)
                    Pv = (Pv >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)
                    # get prediction matrices
                    SO_a = np.transpose(Pa)
                    SO_v = np.transpose(Pv)
                    SO_av = SO_a * SO_v

                    #! obtain ground truth label
                    GT_a, GT_v, GT_av = gt_loader.load(batch_idx)

                    #! calculate metrics
                    evaluator.calc_f1(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)

                    f1_seg_avg = evaluator.stats_segment_f1()[-2]
                    f1_eve_avg = evaluator.stats_event_f1()[-2]
                    if args.log:
                        t.set_postfix_str(
                            'loss: {:.04f}, segment-level avg@type: {:.04f}, event-level avg@type: {:.04f}'.format(loss.item(), f1_seg_avg, f1_eve_avg)
                        )
                        t.update()

        # log info
        f1_seg_a, f1_seg_v, f1_seg_av, f1_seg_avg_type, f1_seg_avg_event = evaluator.stats_segment_f1()
        f1_eve_a, f1_eve_v, f1_eve_av, f1_eve_avg_type, f1_eve_avg_event = evaluator.stats_event_f1()
        
        logger.info('[Segment-Level F1 Score]\t||\tAudio: {:.1f}\t｜\tVisual: {:.1f}\t｜\tAudio-Visual: {:.1f}\t｜\tType@Avg: {:.1f}\t|\tEvent@Avg: {:.1f}'.format(f1_seg_a, f1_seg_v, f1_seg_av, f1_seg_avg_type, f1_seg_avg_event))
        logger.info('[Event-Level F1 Score]\t||\tAudio: {:.1f}\t｜\tVisual: {:.1f}\t｜\tAudio-Visual: {:.1f}\t｜\tType@Avg: {:.1f}\t|\tEvent@Avg: {:.1f}'.format(f1_eve_a, f1_eve_v, f1_eve_av, f1_eve_avg_type, f1_eve_avg_event))
            

    def estimate_ma(self, model, logger):
        args = self.args

        model.eval()
        dataloader = self.dataloaders['estimate']

        das = []
        a_accs, v_accs = [], []
        with torch.no_grad():
            with tqdm(dataloader) as t:
                #! batch size equals 1
                for samples in tqdm(dataloader):
                    # nbatches = inputs['audio'].size(0)
                    #! exchange tracks
                    audio, video_s, video_st, label, idx = samples['audio'], samples['video_s'], samples['video_st'], samples['label'], samples['idx']
                    audio2, video_s2, video_st2, label2, idx2 = samples['audio2'], samples['video_s2'], samples['video_st2'], samples['label2'], samples['idx2']
                    Pa, Pv = samples['pa'], samples['pv']

                    inputs = {'audio': audio, 'video_s': video_s, 'video_st': video_st, 'label': label, 'pa': Pa, 'pv': Pv}
                    inputs_v = {'audio': audio, 'video_s': video_s2, 'video_st': video_st2, 'label': label, 'pa': Pa, 'pv': Pv}     # exchange visual tracks
                    inputs_a = {'audio': audio2, 'video_s': video_s, 'video_st': video_st, 'label': label, 'pa': Pa, 'pv': Pv}      # exchange audio tracks
                    
                    #! forward
                    _, outputs = model(inputs)
                    _, outputs_v = model(inputs_v)
                    _, outputs_a = model(inputs_a)
                    
                    #! stats datas
                    a_prob, v_prob = outputs['a_prob'], outputs['v_prob']
                    a_v, v_v = outputs_v['a_prob'], outputs_v['v_prob']
                    a_a, v_a = outputs_a['a_prob'], outputs_a['v_prob']
                    
                    da = {
                        'a': a_prob.cpu().detach(),
                        'v': v_prob.cpu().detach(),
                        'a_v': a_v.cpu().detach(),
                        'v_v': v_v.cpu().detach(),
                        'a_a': a_a.cpu().detach(),
                        'v_a': v_a.cpu().detach(), 
                        'label':label.cpu(), 'label2':label2.cpu(),
                        'idx': idx, 'idx2': idx
                    }
                    das.append(da)
                    
                    #! stats accuracies
                    a_prob = a_prob.clamp_(min=1e-7, max=1 - 1e-7).cpu()
                    v_prob = v_prob.clamp_(min=1e-7, max=1 - 1e-7).cpu()
                    v_acc = (v_prob>0.5) == label       # [B, 25]
                    a_acc = (a_prob>0.5) == label       # [B, 25]

                    v_acc = v_acc.float()
                    a_acc = a_acc.float()
                    v_accs.append(v_acc)
                    a_accs.append(a_acc)

        v_accs = torch.cat(v_accs, dim=0).mean(0)   # [25]
        a_accs = torch.cat(a_accs, dim=0).mean(0)   # [25]

        get_modality_aware_label(args, das, v_accs, a_accs, logger)


    def estimate_noise(self, model, logger):
        args = self.args

        model.eval()
        dataloader = self.dataloaders['train']

        das = []
        a_prob_list = []
        v_prob_list = []
        with torch.no_grad():
            for inputs in tqdm(dataloader):
                # forward
                loss, outputs = model(inputs)

                #! stats datas
                a_prob, v_prob = outputs['a_prob'], outputs['v_prob']
                Pa, Pv = inputs['pa'], inputs['pv']
                label = inputs['label']
                
                a_prob_list.append(torch.mean(a_prob, dim=0).detach().cpu().numpy())
                v_prob_list.append(torch.mean(v_prob, dim=0).detach().cpu().numpy())
        
                da = {
                    'a': a_prob.cpu().detach(),
                    'v': v_prob.cpu().detach(),
                    'Pa': Pa.cpu().detach(),
                    'Pv': Pv.cpu().detach(),
                    'label':label.cpu(), 
                }
                das.append(da)
                
        a_prob_mean = np.mean(a_prob_list, axis=0)
        v_prob_mean = np.mean(v_prob_list, axis=0)

        get_denoise_label(args, das, a_prob_mean, v_prob_mean, logger)
        

    def train_denoise_epoch(self, epoch, model, optimizer, scheduler, logger):
        args = self.args
        dataloader = self.dataloaders['train']

        model.train()
        if args.parallel:
            dataloader.sampler.set_epoch(epoch)

        noise_ratios = np.load(args.label_denoise)
        noise_ratios_a = torch.from_numpy(noise_ratios['audio']).to(args.device)
        noise_ratios_v = torch.from_numpy(noise_ratios['visual']).to(args.device)
        if args.log:
            print('Loaded denoise labels from', args.label_denoise)

        iters_per_epoch = len(dataloader)

        epoch_loss, epoch_lr = 0., 0.
        with tqdm(dataloader) as t:
            for batch_idx, inputs in enumerate(dataloader):
                #! label denoise
                if args.warm_up_epoch is not None:
                    noise_ratios_a = \
                        torch.min(
                            torch.cat(
                                (noise_ratios_a.reshape(1, -1),
                                noise_ratios_a.reshape(1, -1) *
                                ((epoch - 1) * iters_per_epoch + batch_idx) / (args.warm_up_epoch * iters_per_epoch)),
                                dim=0),
                            dim=0)[0]
                    noise_ratios_v = \
                        torch.min(
                            torch.cat(
                                (noise_ratios_v.reshape(1, -1),
                                noise_ratios_v.reshape(1, -1) *
                                ((epoch - 1) * iters_per_epoch + batch_idx) / (args.warm_up_epoch * iters_per_epoch)),
                                dim=0),
                            dim=0)[0]

                with torch.no_grad():
                    _, outputs = model(inputs, with_ca=False)

                    Pa = inputs['pa'].type(torch.FloatTensor).to(args.device)
                    Pv = inputs['pv'].type(torch.FloatTensor).to(args.device)

                    a_prob, v_prob = outputs['a_prob'], outputs['v_prob']
                    a_prob.clamp_(min=1e-7, max=1-1e-7)
                    v_prob.clamp_(min=1e-7, max=1-1e-7)

                    tmp_loss_a = nn.BCELoss(reduction='none')(a_prob, Pa)
                    tmp_loss_v = nn.BCELoss(reduction='none')(v_prob, Pv)
                    _, sort_index_a = torch.sort(tmp_loss_a, dim=0)
                    _, sort_index_v = torch.sort(tmp_loss_v, dim=0)

                    pos_index_a = Pa > 0.5
                    pos_index_v = Pv > 0.5

                    batch = len(a_prob)
                    for i in range(25):
                        pos_num_a = sum(pos_index_a[:, i].type(torch.IntTensor))
                        pos_num_v = sum(pos_index_v[:, i].type(torch.IntTensor))
                        numbers_a = torch.mul(noise_ratios_a[i], pos_num_a).type(torch.IntTensor)
                        numbers_v = torch.mul(noise_ratios_v[i], pos_num_v).type(torch.IntTensor)
                        # remove noise labels for visual
                        mask_a = torch.zeros(batch).to(args.device)
                        mask_v = torch.zeros(batch).to(args.device)
                        if numbers_v > 0:
                            mask_a[sort_index_a[pos_index_v[sort_index_a[:, i], i], i][:numbers_v]] = 1
                            mask_v[sort_index_v[pos_index_v[sort_index_v[:, i], i], i][-numbers_v:]] = 1
                        mask = torch.nonzero(torch.mul(mask_a, mask_v)).squeeze(-1).type(torch.LongTensor)
                        Pv[mask, i] = 0

                        # remove noise labels for audio
                        mask_a = torch.zeros(batch).to(args.device)
                        mask_v = torch.zeros(batch).to(args.device)
                        if numbers_a > 0:
                            mask_a[sort_index_a[pos_index_a[sort_index_a[:, i], i], i][-numbers_a:]] = 1
                            mask_v[sort_index_v[pos_index_a[sort_index_v[:, i], i], i][:numbers_a]] = 1
                        mask = torch.nonzero(torch.mul(mask_a, mask_v)).squeeze(-1).type(torch.LongTensor)
                        Pa[mask, i] = 0

                inputs['pa'] = Pa 
                inputs['pv'] = Pv

                # forward
                loss, outputs = model(inputs)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr = optimizer.param_groups[0]['lr']
                if args.log:
                    t.set_postfix_str(
                        'loss: {:.04f}, lr: {:.08f}'.format(loss.item(), lr)
                    )
                    t.update()

                epoch_loss += loss.item()
                epoch_lr += lr

        epoch_loss = epoch_loss / len(dataloader)
        epoch_lr = epoch_lr / len(dataloader)
        scheduler.step()
        
        if args.log:
            logger.info('[Train Epoch: {}]\t\t||\tTrain Loss: {:.4f}\t|\tLearning Rate: {:.8f}'.format(epoch, epoch_loss, epoch_lr))
            if args.tb_writer is not None:
                args.tb_writer.add_scalar(f'train/train_loss', epoch_loss, epoch)
                args.tb_writer.add_scalar(f'train/train_lr', epoch_lr, epoch)
                    
        return epoch_loss