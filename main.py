import geospace as gs
from pathlib import Path
from src.prepare import prepare
from src.simulate import simulate
from src.evaluate import basin_evaluate
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from src.utils import run, save_hparams, save_models

if __name__ == '__main__':
    # initial hydra config in jupyter notebook
    config_file = Path('configs/config.yaml')
    if not GlobalHydra.instance().is_initialized():
        initialize_config_dir(config_dir=str(config_file.parent.absolute()))
        cfg = compose(config_name=config_file.name, overrides=["work_dir=."])

    # preapare data first if data is unavailable
    if not Path(cfg.constant.csv.sta).exists():
        prepare(cfg['constant'])

    # hyperparameter tuning using cross validation
    script = ["python", "run.py", "-m"]
    extra = ["dataset.eco=NPL", "tuner=optuna"]
    if 'hyperparameter tuning' not in open(config_file).read():
        cmd = script + extra
        sweep_dir = run(cmd, capture='sweep output dir : (.*)\n')[0][0]
        save_hparams(config_file, sweep_dir)

    # evaluate separately in nine ecoregions using test_size=0.2
    models = ['vanilla', 'joint', 'front', 'rear', 'ealstm']
    ecoregions = ['CPL', 'NAP', 'NPL', 'SAP', 'SPL', 'TPL', 'UMW', 'WMT', 'XER']
    extra = [f"dataset.eco={','.join(ecoregions)}", "hydra/job_logging=disabled",
             "hydra/launcher=joblib", f"hydra.launcher.n_jobs=3"]
    for model in models:
        saved_path = Path(f'saved/evaluate/{model}')
        if saved_path.exists():
            continue
        cmd = script + [f"model={model}"] + extra
        if model == 'vanilla':
            cmd += ["dataset=dynamic"]
        sweep_dir = run(cmd, capture='sweep output dir : (.*)\n')[0][0]
        save_models(saved_path, sweep_dir, ecoregions)
        for eco in ecoregions:
            basin_evaluate(saved_path / f'{eco}/models/model_best.pth')

    # train on the full dataset using test_size=0
    model = 'front'
    saved_path = Path(f'saved/train/{model}')
    if not saved_path.exists():
        cmd = script + [f"model={model}"] + extra + ["datasplit=full"]
        sweep_dir = run(cmd, capture='sweep output dir : (.*)\n')[0][0]
        save_models(saved_path, sweep_dir, ecoregions)

    # simulate gridded baseflow from 1981 to 2020
    gridded_baseflow = 'saved/simulate/usa.tif'
    if not Path(gridded_baseflow).exists():
        Path(gridded_baseflow).parent.mkdir(exist_ok=True, parents=True)
        simulate_tifs = []
        for eco in ecoregions:
            simulate_tif = f'saved/simulate/{eco}.tif'
            simulate_tifs.append(simulate_tif)
            if Path(simulate_tif).exists():
                continue
            checkpoint = f'saved/train/{model}/{eco}/models/model_latest.pth'
            simulate_tifs.append(simulate(checkpoint))
        gs.mosaic(simulate_tifs, gridded_baseflow)
