import re
import yaml
import glob
import shutil
import numpy as np
from pathlib import Path
from itertools import repeat
from omegaconf import OmegaConf
from hydra.utils import instantiate
from subprocess import Popen, PIPE, STDOUT


def run(args, capture=None):
    res = []
    with Popen(args, shell=True, stdout=PIPE, stderr=STDOUT, text=True,
               bufsize=1, encoding='utf-8', errors='ignore') as p:
        for line in p.stdout:
            if capture is not None:
                line_capture = re.findall(capture, line)
                if len(line_capture):
                    res.append(line_capture)
            print(line, end='')
    if capture is not None:
        return res


def hierarchical(dict):
    tree = {}
    for key, value in dict.items():
        t = tree
        parts = key.split(".")
        for part in parts[:-1]:
            t = t.setdefault(part, {})
        t[parts[-1]] = value
    return tree


def save_hparams(config_file, sweep_dir):
    sweep_file = Path.joinpath(Path(sweep_dir), 'optimization_results.yaml')
    tune_conf = hierarchical(OmegaConf.load(sweep_file)['best_params'])
    base_conf = OmegaConf.load(config_file)
    [base_conf.pop(key, None) for key in tune_conf.keys()]
    with open(config_file, 'w') as f:
        f.write('# @package _global_\n')
        OmegaConf.save(base_conf, f)
        f.write('\n# hyperparameter tuning\n')
        OmegaConf.save(tune_conf, f)


def save_models(saved_path, sweep_dir, ecoregions=None):
    saved_path = Path(saved_path)
    saved_path.mkdir(parents=True, exist_ok=True)
    if ecoregions is None or ecoregions[0] == 'null':
        shutil.move(saved_path, sweep_dir)
        return saved_path
    source_dirs = [Path(i) for i in glob.iglob(sweep_dir + r'\*') if Path(i).is_dir()]
    target_dirs = [saved_path / ecoregions[int(d.name)] for d in source_dirs]
    for source_dir, target_dir in zip(source_dirs, target_dirs):
        shutil.move(source_dir, target_dir)
    return target_dirs


def consistent_split(cfg, dataset, datasplit):
    if cfg.dataset.eco is not None or cfg.datasplit.test_size == 0:
        return
    flow_eco = np.repeat(dataset.gages['ECO'], dataset.gages['t'] - dataset.gages['s'])
    train_idxs, test_idxs = [], []
    for eco in cfg.constant.shp.eco.keys():
        eco_dataset = instantiate(cfg.dataset, eco=eco)
        eco_datasplit = instantiate(cfg.datasplit, eco_dataset)
        idx_converter = np.where(flow_eco == eco)[0]
        train_idxs.append(idx_converter[eco_datasplit.splits[0][0]])
        test_idxs.append(idx_converter[eco_datasplit.splits[0][1]])
    datasplit.splits = [(np.concatenate(train_idxs), np.concatenate(test_idxs))]


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def write_yaml(content, fname):
    with fname.open('wt') as handle:
        yaml.dump(content, handle, indent=2, sort_keys=False)


def write_conf(config, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict.pop('constant', None)
    write_yaml(config_dict, save_path)
