import os
import torch
import logging
import datetime
import argparse
import traceback

from model.SRCNet import SRCNet
from runner.pyg_runner import PYGRunner
from data/pth_dataset import PthDataset
from utils.train_helper import get_config


def main():
    #1. Parser
    parser = argparse.ArgumentParser(
          description="Running graph classification experiment"
    )
    parser.add_argument('--script_cfg', type=str, default='config/DEFAULT/DEF_config.json')
    parser.add_argument('--GIN_cfg', type=str, default='config/DEFAULT/DEF_GIN_cfg.json')
    parser.add_argument('--SC_cfg', type=str, default='config/DEFAULT/DEF_SDL_cfg.json')
    parser.add_argument('--OUT_cfg', type=str, default='config/DEFAULT')
    parser.add_argument('--dataset_load_dir', type=str, default='data/PROTEINS/pth')
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
        device=device
    )

    #3. Dataset
    dataset = PthDataset(load_dir=args.dataset_load_dir)

    torch.manual_seed(script_cfg.seed)
    torch.cuda.manual_seed_all(script_cfg.seed)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 4. logger
    logging.basicConfig(level=args.log_level,
                        filename='exp/log_{}'.format(datetime.datetime.now()),
                        filemode='a',)
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(script_cfg.run_id))
    logger.info("Exp comment = {}".format(args.comment))
    logger.info("Config =")
    print(">" * 80)
    pprint(script_cfg)
    print("<" * 80)

    # 6. Runner
    script_cfg.use_gpu = script_cfg.use_gpu and torch.cuda.is_available()
    try:
      runner = PYGRunner(model_object=model, script_cfg=script_cfg,
                         train_dataset=train_dataset, dev_dataset=dev_dataset)
      runner.train()

    except:
      logger.error(traceback.format_exc())

    sys.exit(0)


if __name__ == "__main__":
    main()
