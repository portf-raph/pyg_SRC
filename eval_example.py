import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader


from model.SRCNet import SRCNet
from data.pth_dataset import PthDataset
from utils.train_helper import load_model, get_config
from runner.pyg_runner import PYGRunner


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger = logging.getLogger(__name__)


# 1. Load data
train_dataset = PthDataset(load_dir='./data/PROTEINS/pth/train')
test_dataset = PthDataset(load_dir='./data/PROTEINS/pth/test')
data_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: x)

# 2. Load model
GIN_cfg = get_config('config/DEFAULT/DEF_GIN_cfg.json')
SC_cfg = get_config('config/DEFAULT/DEF_SC_cfg.json')
OUT_cfg = get_config('config/DEFAULT/DEF_LE_cfg.json')
model_class = "LeastEnergy"

model = SRCNet(GIN_cfg=GIN_cfg,
               SC_cfg=SC_cfg,
               OUT_cfg=OUT_cfg,
               model_class=model_class,
               device=device)
load_model(model=model, file_name='../exp/pyg_SRC/model_snapshot_best.pth', optimizer=None)
model.train()

# 3. Train forward
iter = 0
for data_dicts in data_loader:

    data_dicts = [{k: v.to(device) if isinstance(v, torch.Tensor)
        else v for k, v in d.items()} for d in data_dicts]
    out, A_fidelity, A_incoherence = model(data_dicts)
    out = torch.softmax(out, dim=1)
    print(out, A_fidelity, A_incoherence)
    iter += 1
    if iter == 10:
        break

# 4. Test forward
script_cfg = get_config('config/DEFAULT/DEF_config.json')
runner = PYGRunner(model_object=model, script_cfg=script_cfg, logger=logger,
        train_dataset=None, dev_dataset=None, test_dataset=test_dataset)
runner.test()

# 4. Load graphs
save_dir = '../exp/pyg_SRC'
with open(os.path.join(save_dir, 'train_stats.p'), 'rb') as handle:
    results = pickle.load(handle)
plt.figure()
plt.plot(results['train_step'], results['train_loss'], '-b', label='Perte')
plt.xlabel("Nombre d'itérations")
plt.ylabel("Perte d'entropie croisée")
plt.title('Perte de classification')
plt.savefig(os.path.join(save_dir, "Perte de classification"+".png"))
plt.show()
plt.close()

plt.figure()
plt.plot(results['train_A_fidelity'], '-b', label='Fidelite')
plt.xlabel("Nombre d'époques")
plt.ylabel("Erreur de fidélité")
plt.title("Fidélité")
plt.savefig(os.path.join(save_dir, "Fidélité"+".png"))
plt.show()
plt.close()

plt.figure()
plt.plot(results['train_A_incoherence'], '-r', label='Incohérence')
plt.xlabel("Nombre d'époques")
plt.ylabel("Erreur d'incohérence")
plt.title("Incohérence")
plt.savefig(os.path.join(save_dir, "Incohérence"+".png"))
plt.show()
plt.close()

plt.figure()
plt.plot(5*np.arange(1, len(results['val_epoch_acc'])+1), results['val_epoch_acc'], '-g', label="Exactitude de validation d'epoche")
plt.xlabel("Nombre d'epoques")
plt.ylabel("% de succès en classification")
plt.title("Validation")
plt.savefig(os.path.join(save_dir, "Validation"+".png"))
plt.show()
plt.close()


