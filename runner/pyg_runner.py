import os
import numpy as np
import pickle
import datetime
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader

import logging
from utils.train_helper import *


logger = logging.getLogger(__name__)

class PYGRunner(object):
    def __init__(
        self,
        model_object: nn.Module,
        script_cfg: dict,
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


    def train(self):
        # data loaders
        collate_fn = lambda x: x    # list of non-uniform tensors, no stacking
        train_loader = DataLoader(dataset=self.train_dataset,
                                  batch_size=self.train_cfg['batch_size'],
                                  shuffle=self.train_cfg['shuffle'],
                                  num_workers=self.train_cfg['num_workers'],
                                  collate_fn=collate_fn
                                 )

        dev_loader = DataLoader(dataset=self.dev_dataset,
                                batch_size=self.train_cfg['batch_size'],
                                num_workers=self.train_cfg['num_workers'],
                                collate_fn=collate_fn
                                )

        # model
        model = self.model
        _eta = self.model.SC._eta
        if self.use_gpu:
            device = torch.device('cuda')
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()
        else:
            device = torch.device('cpu')

        # optimizer
        params = filter(lambda p: p.requires_grad and p is not self.model.SC.A,
                        self.model.parameters())
        params = list(params)
        A = self.model.SC.A

        if self.train_cfg['optimizer'] == 'SGD':
            optimizer = optim.SGD([
                {'params': params,
                'lr': self.train_cfg['lr'],
                'momentum': self.train_cfg['momentum'],
                'weight_decay': self.train_cfg['wd']},
                {'params': A,
                'lr': self.train_cfg['lr_A'],
                'momentum': self.train_cfg['momentum_A'],
                'weight_decay': self.train_cfg['wd_A']}
            ])

        elif self.train_cfg['optimizer'] == 'Adam':
            optimizer = optim.Adam([
                {'params': params,
                'lr': self.train_cfg['lr'],
                'weight_decay': self.train_cfg['wd']},
                {'params': A,
                'lr': self.train_cfg['lr_A'],
                'weight_decay': self.train_cfg['wd_A']}
            ])

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
            load_model(self.model, self.train_cfg['resume_model'], optimizer=optimizer)   # mod call

        # training loop
        iter_count = 0
        best_val_loss = np.inf
        results = defaultdict(list)
        for epoch in range(self.train_cfg['max_epoch']):
            epoch_loss = 0
            # validation
            if (epoch + 1) % self.train_cfg['valid_epoch'] == 0 or epoch == 0:
                model.eval()
                val_loss = 0
                A_fidelity = 0
                A_incoherence = 0
                correct = 0

                for data_dicts in tqdm(dev_loader):
                    if self.use_gpu:
                        data_dicts = [{k: v.to(device) if isinstance(v, torch.Tensor)
                        else v for k, v in d.items()} for d in data_dicts]
                    y = torch.tensor([d['y'].item() for d in data_dicts], dtype=torch.long, device=device)
                    with torch.no_grad():
                        out = model(data_dicts)    # FORWARD
                        val_loss += F.cross_entropy(out, y).item()    # TODO: test
                        A_fidelity += self.model.SC.A_fidelity
                        A_incoherence += self.model.SC.A_incoherence
                        pred = out.max(dim=1)[1]
                        correct += pred.eq(y).sum().item()

                len_loader = len(dev_loader)
                val_loss = val_loss / len_loader
                A_fidelity = A_fidelity / len_loader
                A_incoherence = A_incoherence / len_loader
                val_acc = correct / len(dev_loader.dataset)
                print("Avg. Validation CrossEntropy = {}".format(val_loss))
                print("Avg. Validation Accuracy = {}".format(val_acc))
                results['val_loss'] += [val_loss]
                results['A_fidelity'] += [A_fidelity]
                results['A_incoherence'] += [A_incoherence]
                results['val_acc'] += [val_acc]

                # save best model
                if val_loss < best_val_loss:
                  best_val_loss = val_loss
                  snapshot(
                      model.module if self.use_gpu else model,
                      optimizer,
                      self.script_cfg,
                      epoch + 1,
                      tag='best')

                logger.info("Current Best Validation CrossEntropy = {}".format(best_val_loss))

                # check early stop
                if early_stop.tick([val_acc]):
                  print("STOPPING TIME DUE NOW")
                  snapshot(
                      model.module if self.use_gpu else model,
                      optimizer,
                      self.script_cfg,
                      epoch + 1,
                      tag='last')
                  break

            # training
            # TODO: configre model.SC.compute_loss
            model.train()
            for data_dicts in train_loader:
                optimizer.zero_grad()
                if self.use_gpu:
                    data_dicts = [{k: v.to(device) if isinstance(v, torch.Tensor)
                    else v for k, v in d.items()} for d in data_dicts]
                y = torch.tensor([d['y'].item() for d in data_dicts], dtype=torch.long, device=device)
                out = model(data_dicts) # FORWARD

                A_loss = self.model.SC.A_fidelity + _eta * self.model.SC.A_incoherence
                A_loss.backward(retain_graph=True)
                train_loss = F.cross_entropy(out, y)
                torch.autograd.backward(
                            train_loss,
                            inputs=params)

                optimizer.step()
                train_loss = float(train_loss.data.cpu().numpy())
                results['train_loss'] += [train_loss]
                results['train_step'] += [iter_count]   # TODO: record A loss statistics
                epoch_loss += train_loss

                # display loss
                if (iter_count + 1) % self.train_cfg['display_iter'] == 0:
                    logger.info("Loss @ epoch {:04d} iteration {:08d} = {}".format(
                        epoch + 1, iter_count + 1, train_loss))

                iter_count += 1
            if (epoch + 1) % self.train_cfg['snapshot_epoch'] == 0:
                logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
                snapshot(model.module
                         if self.use_gpu else model, optimizer, self.script_cfg, epoch + 1)

            epoch_loss = epoch_loss / len(train_loader)
            print("Loss @ epoch {:04d} = {}".format(epoch+1, epoch_loss))
            lr_scheduler.step()

        results['best_val_loss'] += [best_val_loss]
        pickle.dump(results,
                    open(os.path.join(self.script_cfg['save_dir'], 'train_stats.p'), 'wb'))
        logger.info("Best Validation MSE = {}".format(best_val_loss))

        return best_val_loss


    def test(self):
        collate_fn = lambda x: x
        test_loader = DataLoader(dataset=self.test_dataset,
                                 batch_size=self.test_cfg['batch_size'],
                                 shuffle=False,
                                 num_workers=self.test_cfg['num_workers'],
                                 drop_last=False,
                                 collate_fn=collate_fn)
        load_model(self.model, self.test_cfg['test_model'])

        if self.use_gpu:
            device = torch.device('cuda')
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        model.eval()
        test_loss = []

        for data_dicts in tqdm(test_loader):
            if self.use_gpu:
                data_dicts = [{k: v.to(device) if isinstance(v, torch.Tensor)
                else v for k, v in d.items()} for d in data_dicts]
            y = torch.tensor([d['y'].item() for d in data_dicts], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(data_dicts)
                curr_loss = F.cross_entropy(out, y).cpu().numpy()
                test_loss += [curr_loss]

        test_loss = float(np.mean(np.concatenate(test_loss)))
        logger.info("Test MSE = {}".format(test_loss))

        return test_loss
