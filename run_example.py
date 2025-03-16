import os
import sys
import torch
import pprint
import logging
import datetime
import argparse
import traceback

from model.SRCNet import SRCNet
from runner.pyg_runner import PYGRunner
from data.pth_dataset import PthDataset
from utils.train_helper import get_config


def main():
    #1. Parser
    parser = argparse.ArgumentParser(
          description="Running graph classification experiment"
    )
    parser.add_argument('--script_cfg', type=str, default='./config/DEFAULT/DEF_config.json')
    parser.add_argument('--GIN_cfg', type=str, default='./config/DEFAULT/DEF_GIN_cfg.json')
    parser.add_argument('--SC_cfg', type=str, default='./config/DEFAULT/DEF_SC_cfg.json')
    parser.add_argument('--OUT_cfg', type=str, default='./config/DEFAULT/DEF_LE_cfg.json')
    parser.add_argument('--model_class', type=str, default='LeastEnergy')
    parser.add_argument('--dataset_load_dir', type=str, default='./data/PROTEINS/pth/')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help="Logging Level, \
                          DEBUG, \
                          INFO, \
                          WARNING, \
                          ERROR, \
                          CRITICAL")
    parser.add_argument('--comment', type=str, help="Experiment comment")
    parser.add_argument('--test', type=str, default='False')
    args = parser.parse_args()


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    #2. Load model
    script_cfg = get_config(args.script_cfg)
    GIN_cfg = get_config(args.GIN_cfg)
    SC_cfg = get_config(args.SC_cfg)
    OUT_cfg = get_config(args.OUT_cfg)

    model = SRCNet(
        GIN_cfg=GIN_cfg,
        SC_cfg=SC_cfg,
        OUT_cfg=OUT_cfg,
        model_class=args.model_class,
        device=device
    )
    if not os.path.exists(script_cfg["save_dir"]):
        os.makedirs(script_cfg["save_dir"])

    #3. Dataset
    train_dataset = PthDataset(load_dir=os.path.join(args.dataset_load_dir, 'train'))
    dev_dataset = PthDataset(load_dir=os.path.join(args.dataset_load_dir, 'val'))

    # 4. logger
    log_file = '../exp/pyg_SRC/log_{}.txt'.format(datetime.datetime.now())
    logging.basicConfig(level=args.log_level,
                        filename=log_file,
                        filemode='a',)
    logger = logging.getLogger(__name__)
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(script_cfg["run_id"]))
    logger.info("Exp comment = {}".format(args.comment))
    logger.info("Config =")
    print(">" * 80)
    pprint.pprint(script_cfg)
    print("<" * 80)

    # 5. Runner
    script_cfg["use_gpu"] = script_cfg["use_gpu"] and torch.cuda.is_available()
    
    runner = PYGRunner(model_object=model, script_cfg=script_cfg, logger=logger,
                       train_dataset=train_dataset, dev_dataset=dev_dataset)
    runner.train()    


if __name__ == '__main__':
    main()
