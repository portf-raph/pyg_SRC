# pyg_SRC
Sparse representation-based classification of graph signals implemented in the pytorch-geometric framework

---

## training Script
```bash
python3 run_example.py --[FLAGS]
```

### available Flags

| argument             | type | default                             | description                                                    |
| -------------------- | ---- | ----------------------------------- | -------------------------------------------------------------- |
| `--script_cfg`       | str  | `./config/DEFAULT/DEF_config.json`  | Path to the main script configuration                          |
| `--GIN_cfg`          | str  | `./config/DEFAULT/DEF_GIN_cfg.json` | GIN architecture configuration                                 |
| `--SC_cfg`           | str  | `./config/DEFAULT/DEF_SC_cfg.json`  | Spectral component configuration                               |
| `--LE_cfg`           | str  | `./config/DEFAULT/DEF_LE_cfg.json`  | Least energy module configuration                              |
| `--OUT_cfg`          | str  | `./config/DEFAULT/DEF_LE_cfg.json`  | Output processing configuration (same as LE\_cfg)              |
| `--model_class`      | str  | `'LeastEnergy'`                     | Model class to instantiate                                     |
| `--dataset_load_dir` | str  | `./data/PROTEINS/pth/`              | Directory from which to load the dataset                       |
| `--log_level`        | str  | `'INFO'`                            | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `--comment`          | str  | `None`                              | Optional experiment comment                                    |
| `--test`             | str  | `'False'`                           | Whether to run in test mode                                    |

---
