import os
import pandas as pd
from source.utils import down_local, gee_static, local_dynamic, local_static, FAO_PET
from concurrent.futures import ThreadPoolExecutor, Future
from hydra.experimental import compose, initialize_config_dir


def preprocess(cfg):
    # 0. download data into local disk
    down_local.download_local_dynamic(cfg)
    down_local.download_local_static_vegetation_cover(cfg)
    down_local.download_local_static_vegetation_cover(cfg)
    down_local.global_surface_water(cfg)

    # 1. preprocess dynamic data in local disk
    # 1.1 monthly baseflow as the training target
    local_dynamic.daily_to_monthly_flow(cfg)

    # 1.2 Mean April baseflow for plotting
    local_dynamic.multi_year_average_flow(cfg)

    # 1.3 gridded tif of time series as the simulation inputs
    local_dynamic.tif_preprocess(cfg)

    # 1.4 calculate potential evaportranspiration
    dfs = []
    dfs.append(gee_static.dataset(cfg, "MERIT/DEM/v1_0_3", 'dem', resample='bilinear'))
    FAO_PET.estimate_PET(cfg)

    # 1.5 csv file of time series as the training inputs
    local_dynamic.csv_preprocess(cfg)

    # 2. preprocess static data from local disk and GEE
    # 2.1 Permeability
    local_static.permeability_tif(cfg)
    dfs.append(local_static.permeability_csv(cfg))

    # 2.2 water occurrence
    local_static.water_occurrence_tif(cfg)
    dfs.append(local_static.water_occurrence_csv(cfg))

    # 2.3 ERA5 multi-year average
    local_static.ERA5_tif(cfg)
    dfs.append(local_static.ERA5_csv(cfg))

    # 2.4 download datasets from google earth engine
    assets = ["NASA/ASTER_GED/AG100_003", "UMD/hansen/global_forest_change_2015_v1_3"]
    bands = ['ndvi', 'treecover2000']
    scales = [0.01, 1]
    with ThreadPoolExecutor(max_workers=100) as executor:
        dfs.append(executor.submit(gee_static.slope, cfg, "MERIT/DEM/v1_0_3"))
        for asset, band, scale in zip(assets, bands, scales):
            dfs.append(executor.submit(gee_static.dataset, cfg, asset, band, scale=scale))
        for band in ['bdod', 'cfvo', 'clay', 'sand', 'silt']:
            dfs.append(executor.submit(lambda x: gee_static.soilgrids(cfg, x), band))
    dfs = [df.result() if isinstance(df, Future) else df for df in dfs]
    df = pd.concat(dfs, axis=1)
    df.to_csv(cfg.csv.sta, index=None)


if __name__ == '__main__':
    initialize_config_dir(config_dir=os.path.join(
        os.getcwd(), "configs"), job_name="test_app")
    cfg = compose(config_name="config", overrides=["work_dir=."])
    preprocess(cfg)
