import os
import numpy as np
import pandas as pd
import geospace as gs
from osgeo import ogr
from functools import partial
from omegaconf import DictConfig, OmegaConf
from multiprocessing import Pool, cpu_count


def assign_worker(cfg, basin_id):
    filter_sql = f"STAID = '{basin_id}'"
    filter_shp = gs.shp_filter(cfg.shp.basins, filter_sql)
    ds_basin = ogr.Open(filter_shp)
    lyr_basin = ds_basin.GetLayer()
    f_basin = lyr_basin.GetFeature(0)
    geom_basin = f_basin.GetGeometryRef()

    intersect_area = np.zeros(len(cfg.shp.eco))
    for i, shp in enumerate(cfg.shp.eco.values()):
        ds_eco = ogr.Open(shp)
        lyr_eco = ds_eco.GetLayer()
        for f_eco in lyr_eco:
            geom = f_eco.GetGeometryRef()
            intersect_area[i] += geom.Intersection(geom_basin).Area()

    return np.argmax(intersect_area), *geom_basin.Centroid().GetPoint()[:2]


def assign(cfg):
    if (os.path.exists(cfg.gages) and
            'ECO' in pd.read_excel(cfg.gages, nrows=0).columns):
        return
    gages = pd.read_excel(cfg.gages, dtype={'STAID': str})

    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    with Pool(cpu_count() * 3 // 4) as p:
        res = p.map(partial(assign_worker, cfg), gages['STAID'])

    eco_names = np.array(list(cfg.shp.eco.keys()))
    res_array = np.array(res)
    gages['ECO'] = eco_names[res_array[:, 0].astype(int)]
    gages['Lon_Cent'] = res_array[:, 1]
    gages['Lat_Cent'] = res_array[:, 2]
    gages.to_excel(cfg.gages, index=None)
