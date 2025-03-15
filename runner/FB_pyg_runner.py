import os
import gc
import numpy as np
import pickle
import datetime
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from logging import Logger

from utils.train_helper import *
from utils.hooks import build_r_coeffs_hook


class FB_PYGRunner(object):
    def __init__(
        self,
        model_object: nn.Module,
        script_cfg: dict,
        logger: Logger,
        train_dataset: Dataset,
        dev_dataset: Dataset,
        test_dataset: Dataset=None,
        ):

        if not isinstance(script_cfg, dict):
            raise TypeError("Script config file is not in dict format")

        self.script_cfg = script_cfg
        self.train_cfg = script_cfg['train']
        self.test_cfg = script_cfg['test']
        self.use_gpu = script_cfg['use_gpu']
        self.gpus = script_cfg['gpus']

        self.model = model_object
        self.train_dataset = train_dataset    # all PthDataset objects
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.logger = logger


    def train(self):
        # data loaders
        collate_fn = lambda x: x    # list of non-uniform tensors, no stacking
        train_loader = DataLoader(dataset=self.train_dataset,
                                  batch_size=self.train_cfg['batch_size'],
                                  shuffle=self.train_cfg['shuffle'],
                                  num_workers=self.train_cfg['num_workers'],
                                  pin_memory=True if self.use_gpu else False,
                                  collate_fn=collate_fn,
                                 )

        dev_loader = DataLoader(dataset=self.dev_dataset,
                                batch_size=self.train_cfg['batch_size'],
                                num_workers=self.train_cfg['num_workers'],
                                pin_memory=True if self.use_gpu else False,
                                collate_fn=collate_fn
                                )

        # model
        model = self.model
        _eta = model.SC._eta
        
        # --- A.requires_grad() = False ---
        model.SC.A.requires_grad() = False
        if self.use_gpu:
            device = torch.device('cuda')
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()
        else:
            device = torch.device('cpu')

        # optimizer
        params = filter(lambda p: p.requires_grad and p is not self.model.SC.A,
                        self.model.parameters())
        params = list(params)

        if self.train_cfg['optimizer'] == 'SGD':
            optimizer = optim.SGD(
                {'params': params,
                'lr': self.train_cfg['lr'],
                'momentum': self.train_cfg['momentum'],
                'weight_decay': self.train_cfg['wd']},
                )

        elif self.train_cfg['optimizer'] == 'Adam':
            optimizer = optim.Adam(
                {'params': params,
                'lr': self.train_cfg['lr'],
                'weight_decay': self.train_cfg['wd']
                 }
                )

        else:
            raise ValueError("Non-supported optimizer!")

        early_stop = EarlyStopper([0.0], win_size=10, is_decrease=True)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.train_cfg['lr_decay_steps'],
            gamma=self.train_cfg['lr_decay'])

        # reset gradient
        optimizer.zero_grad()
        if self.train_cfg['is_resume']:
            load_model(model, self.train_cfg['resume_model'], optimizer=optimizer)   # mod call

        # training loop
        len_dev = len(self.dev_dataset)
        len_dev_loader = len(dev_loader)
        len_train = len(self.train_dataset)

        iter_count = 0
        best_val_loss = np.inf
        results = defaultdict(list)
       
        # === Constant hooks ===
        _r_f_batch, _r_f_batch_hook = build_r_f_batch_hook()
        _r_f_batch_handle = self.model.SC.reg`
        # === ////////////// ===

        # FB globals
        SC_K = self.model.SC.K
        SC_Q = self.model.SC.Q
        SC_partition = self.model.SC.partition
        R_length = self.model.SC.num_atoms * SC_K
        for epoch in range(self.train_cfg['max_epoch']):
            # validation
            if (epoch + 1) % self.train_cfg['valid_epoch'] == 0 or epoch == 0:
                model.eval()
                val_loss = 0
                avg_A_fidelity = 0
                avg_A_incoherence = 0
                correct = 0
                hook_trigger = True
                
                for data_dicts in tqdm(dev_loader):
                    # print(f"Reserved: {torch.cuda.memory_reserved()/1e9} GB")
                    if self.use_gpu:
                        data_dicts = [{k: v.to(device) if isinstance(v, torch.Tensor)
                        else v for k, v in d.items()} for d in data_dicts]
                    y_batch = torch.tensor([d["y"].item() for d in data_dicts], dtype=torch.long, device=device) #d

                    if hook_trigger:
                        _, _r_coeffs_hook = build_r_coeffs_hook(y_batch=y_batch,
                                                                partition=self.model.SC.partition)
                        _r_coeffs_handle = self.model.SC.register_forward_hook(_r_coeffs_hook)
                        
                    with torch.no_grad():
                        out, _, _ = model(data_dicts)    # FORWARD
                        # === GC ===
                        #gc.collect()
                        #torch.cuda.empty_cache()
                        # === GC ===
                        if hook_trigger:
                            _r_coeffs_handle.remove()
                            hook_trigger = False
                            
                        val_loss += F.cross_entropy(out, y_batch).cpu().item()
                        pred = out.max(dim=1)[1]
                        correct += pred.eq(y_batch).sum().cpu().item()

                val_loss = val_loss / len_dev_loader
                val_acc = correct / len_dev
                print("Avg. Validation CrossEntropy = {}".format(val_loss))     # dbg
                print("Avg. Validation Accuracy = {}".format(val_acc))
                results['val_epoch_loss'] += [val_loss]
                results['val_epoch_acc'] += [val_acc]

                # save best model
                if val_loss < best_val_loss:
                  best_val_loss = val_loss
                  snapshot(
                      model.module if self.use_gpu else model,
                      optimizer,
                      self.script_cfg,
                      epoch + 1,
                      tag='best')

                self.logger.info("Current Best Validation CrossEntropy = {}".format(best_val_loss))

                # check early stop
                if early_stop.tick([val_acc]):
                    print("STOPPING TIME DUE NOW")
                    snapshot(
                        model.module if self.use_gpu else model,
                        optimizer,
                        self.script_cfg,
                        epoch + 1,
                        tag='last')

                    break   # TODO: configure SC.compute_loss

            # training
            model.train()
            avg_A_fidelity = 0
            avg_A_incoherence = 0
            hook_trigger = True

            # FB Quadratic program
            R_1 = torch.zeros((R_length, R_length), device=device)
            R_2 = torch.zeros(R_length, device=device)
            for data_dicts in tqdm(train_loader):
                # print(torch.cuda.memory_allocated(device)/torch.cuda.max_memory_allocated(device))
                # print(f"Reserved: {torch.cuda.memory_reserved()/1e9} GB")
                optimizer.zero_grad()
                if self.use_gpu:
                    data_dicts = [{k: v.to(device) if isinstance(v, torch.Tensor)
                    else v for k, v in d.items()} for d in data_dicts]
                y_batch = torch.tensor([d["y"].item() for d in data_dicts], dtype=torch.long, device=device) # d

                if hook_trigger:
                    _, _r_coeffs_hook = build_r_coeffs_hook(y_batch=y_batch,
                                                            partition=self.model.SC.partition)
                    _r_coeffs_handle = self.model.SC.register_forward_hook(_r_coeffs_hook)

                out, A_fidelity, A_incoherence = model(data_dicts) # FORWARD

                if hook_trigger:
                    _r_coeffs_handle.remove()
                    hook_trigger = False
                
                # === Collect detached _r_batch, _f_batch ===
                _r_batch = _r_f_batch['_r_batch']
                _f_batch = _r_f_batch['_f_batch']
                # _R_r_batch
                _R_r_batch = get_R_bmm(data_dicts=data_dicts,
                                       _r_batch=_r_batch,
                                       K=SC_K)
                R_1 += torch.sum(
                                torch.bmm(_R_r_batch.transpose(-2, -1), _R_r_batch),dim=0
                                )
                R_2 += torch.sum(
                            torch.bmm(_R_r_batch.transpose(-2, -1), _f_batch),dim=0
                        )   # (B x CK x CK) x (B x CK x 1)

                start_batch = [SC_partition[y] for y in y_batch]
                end_batch = [SC_partition[y+1] for y in y_batch]
                sub_Q_batch = torch.stack(
                        [torch.cat((SC_Q[:, start:end],
                                    SC_Q[:, SC_partition[-2]:SC_partition[-1]]), dim=1 for start, end in zip(start_batch, end_batch)]
                _QQTr_batch = torch.bmm(
                        sub_Q_batch, torch.bmm(sub_Q_batch.transpose(-2,-1), _r_batch.unsqueeze(-1))
                    ) # val
                _R_QQTr_batch = get_R(data_dicts=data_dicts,
                                      _r_batch=_QQTr_batch,
                                      K=SC_K)
                R_1 += torch.sum(
                                torch.bmm(_R_QQTr_batch.transpose(-2,-1), _R_QQTr_batch)
                            )
                R_2 += torch.sum(
                        torch.bmm(_R_QQTr_batch.transpose(-2,-1), _f_batch),dim=0
                    )
                # === Incoherence penalty ===
                # === /////////////////// ===

                train_loss = F.cross_entropy(out, y_batch)
                torch.autograd.backward(
                            train_loss,
                            inputs=params)
                if (iter_count + 1) % self.train_cfg['display_iter'] == 0:
                    GIN_param = next(iter(self.model.GIN.parameters()))
                    # OUT_param = next(iter(self.model.OUT.parameters()))
                    print('\nGIN grad: {}'.format(torch.mean(torch.abs(GIN_param.grad))))
                    # print('OUT grad: {}'.format(torch.mean(torch.abs(OUT_param.grad))))
                # === GC ===
                #gc.collect()
                #torch.cuda.empty_cache()
                # === GC ===
                optimizer.step()
                avg_A_fidelity += A_fidelity.detach().cpu().item()
                avg_A_incoherence += A_incoherence.detach().cpu().item()

                # === GC ===
                # del data_dicts, y_batch, out, A_loss, A_fidelity, A_incoherence
                # === GC ===

                train_loss = float(train_loss.detach().cpu().numpy())
                results['train_loss'] += [train_loss]
                results['train_step'] += [iter_count]

                # display loss
                if (iter_count + 1) % self.train_cfg['display_iter'] == 0:
                    print("Training CrossEntropy at iteration {}: {}".format(iter_count, train_loss))
                    self.logger.info("Loss @ epoch {:04d} iteration {:08d} = {}".format(
                        epoch + 1, iter_count + 1, train_loss))

                iter_count += 1

            # === Parameter A update ===
            R_1_bmm = lambda x: torch.bmm(R_1, x) 
            A = self.model.SC.A.view(K, num_atoms, out_channels).transpose(0,1).reshape(-1, out_channels)
            A = cg_batch(R_1_bmm, R_2)
            self.model.SC.A = A.view(num_atoms, K, out_channels).transpose(0,1).reshape(-1, num_atoms*out_channels)
            # === ////////////////// ===
            if (epoch + 1) % self.train_cfg['snapshot_epoch'] == 0:
                self.logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
                snapshot(model.module
                         if self.use_gpu else model, optimizer, self.script_cfg, epoch + 1)
            
            avg_A_fidelity = avg_A_fidelity / len_train
            avg_A_incoherence = avg_A_incoherence / len_train
            results['train_A_fidelity'] += [avg_A_fidelity]
            results['train_A_incoherence'] += [avg_A_incoherence]
            lr_scheduler.step()
            
        results['best_val_loss'] += [best_val_loss]
        pickle.dump(results,
                    open(os.path.join(self.script_cfg['save_dir'], 'train_stats.p'), 'wb'))
        self.logger.info("Best Validation MSE = {}".format(best_val_loss))

        return best_val_loss


    def test(self):
        collate_fn = lambda x: x
        test_loader = DataLoader(dataset=self.test_dataset,
                                 batch_size=self.test_cfg['batch_size'],
                                 shuffle=False,
                                 num_workers=self.test_cfg['num_workers'],
                                 drop_last=False,
                                 collate_fn=collate_fn)
        model = self.model
        load_model(model, self.test_cfg['test_model'])
        if self.use_gpu:
            device = torch.device('cuda')
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        model.eval()
        test_loss = []

        for data_dicts in tqdm(test_loader):
            if self.use_gpu:
                data_dicts = [{k: v.to(device) if isinstance(v, torch.Tensor)
                else v for k, v in d.items()} for d in data_dicts]
            y_batch = torch.tensor([d["y"].item() for d in data_dicts], dtype=torch.long, device=device)  # d
            with torch.no_grad():
                out = model(data_dicts)
                curr_loss = F.cross_entropy(out, y_batch).cpu().numpy()
                test_loss += [curr_loss]

        test_loss = float(np.mean(np.concatenate(test_loss)))
        self.logger.info("Test MSE = {}".format(test_loss))

        return test_loss
