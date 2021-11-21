import os
import glob
import shutil
import numpy as np
import pandas as pd
import geospace as gs
from osgeo import gdal
from functools import partial
from calendar import monthrange
from collections import OrderedDict
from omegaconf import DictConfig, OmegaConf
from multiprocessing import Pool, cpu_count


def daily_to_monthly_flow(cfg):
    """convert daily baseflow to monthly baseflow

    Args:
        cfg (omegaconfig): configs set in yaml
    """
    if os.path.exists(cfg.gages):
        return

    gages = pd.read_excel(cfg.gages_origin, dtype={'STAID': str})
    drop_index = []

    for index, g in gages.iterrows():
        f_in = os.path.join(cfg.csv.flow.daily, g['STAID'] + '.txt')
        f_out = os.path.join(cfg.csv.flow.monthly, g['STAID'] + '.csv')
        if os.path.exists(f_out):
            continue

        df = pd.read_csv(f_in, sep='\t', header=None,
                         names=['Y', 'M', 'D', 'B'])
        year = np.unique(df['Y'])
        year = year[(year >= cfg.beg.year) & (year <= cfg.end.year)]
        year = np.repeat(year, 12)
        month = np.arange(1, 13)
        month = np.tile(month, int(year.shape[0] / 12))
        first_year_delete = np.where(
            (month < cfg.beg.month) & (year == cfg.beg.year))[0]
        final_year_delete = np.where(
            (month > cfg.end.month) & (year == cfg.end.year))[0]
        delete_year = np.r_[first_year_delete, final_year_delete]
        year = np.delete(year, delete_year)
        month = np.delete(month, delete_year)
        flow = np.full(month.shape, np.nan)
        for i, (y, m) in enumerate(zip(year, month)):
            select = (df['Y'] == y) & (df['M'] == m)
            month_days = monthrange(y, m)[1]
            if np.sum(select) == month_days:
                # convert unit from m3/s to mm/day
                flow[i] = np.sum(df['B'][select]) * 86.4 / \
                    (g['Area'] * month_days)

        delete_rows = np.where(np.isnan(flow))
        year = np.delete(year, delete_rows)
        month = np.delete(month, delete_rows)
        flow = np.delete(flow, delete_rows)

        if flow.shape[0] < 36:
            drop_index.append(index)
            continue

        if not os.path.exists(cfg.csv.flow.monthly):
            os.makedirs(cfg.csv.flow.monthly)
        df_out = pd.DataFrame.from_dict(
            OrderedDict(zip(['Y', 'M', 'B'], [year, month, flow])))
        df_out.to_csv(f_out, index=False)

    gages = gages.drop(drop_index)
    gages.to_excel(cfg.gages, index=None)


def multi_year_average_flow(cfg):
    """calculate multi-year average baseflow for specific month

    Args:
        cfg (omegaconfig): configs set in yaml
    """
    month_str = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    if (os.path.exists(cfg.gages) and
            month_str[cfg.month_compared - 1] in
            pd.read_excel(cfg.gages, nrows=0).columns):
        return

    gages = pd.read_excel(cfg.gages, dtype={'STAID': str})
    flow_mean = []
    for i in gages['STAID']:
        flow_file = os.path.join(cfg.csv.flow.monthly, i + '.csv')
        flow_df = pd.read_csv(flow_file)
        flow_mean.append(
            flow_df[flow_df['M'] == cfg.month_compared]['B'].mean())
    gages[month_str[cfg.month_compared - 1]] = flow_mean
    gages.to_excel(cfg.gages, index=None)


def tif_preprocess(cfg):
    # example time series raster
    ds_eg = gdal.Open(os.path.join(cfg.raw.dyn, '2m_temperature.grib'))
    t = ds_eg.GetGeoTransform()

    # get study bound and grib file spatial reference
    bound = gs.grid_bound(ds_eg, cfg.shp.usa)[0]
    bound_srs = "+proj=longlat +datum=WGS84 +ellps=WGS84"

    # convert int8 to uint8
    rasters = glob.glob(os.path.join(cfg.raw.dyn, '*.tif')) + \
        glob.glob(os.path.join(cfg.raw.dyn, '*.grib')) + \
        glob.glob(os.path.join(cfg.raw.sta, '*.tif')) + \
        glob.glob(os.path.join(cfg.raw.sta, '*.grib'))
    with Pool(min(cpu_count() - 1, len(rasters))) as p:
        p.map(gs.convert_uint8, rasters)

    # convert grib to tif with wgs 84
    grib_rasters = [r for r in rasters if '.grib' in r]
    warp_option = dict(outputBounds=bound, dstSRS=bound_srs,
                       dstNodata=cfg.gdal.nodata, xRes=t[1], yRes=t[5])
    with Pool(min(cpu_count() - 1, len(grib_rasters))) as p:
        p.map(partial(gs.grib_to_tif, **warp_option), grib_rasters)

    # deal the 8 time series rasters
    out_path = cfg.tif.dyn
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # precipitation
    ras = os.path.join(cfg.raw.dyn, 'total_precipitation.tif')
    gs.map_calc(ras, 'A*1000', out_path)

    # snowfall
    ras = os.path.join(cfg.raw.dyn, 'snowfall.tif')
    gs.map_calc(ras, 'A*1000', out_path)

    # 2m dewpoint temperature
    ras = os.path.join(cfg.raw.dyn, '2m_dewpoint_temperature.tif')
    gs.map_calc(ras, 'A-273.15', out_path)
    out_file = gs.context_file(ras, out_path)
    if not os.path.exists(out_file):
        gs.Calc('A-273.15', out_file, creation_options=gs.CREATION,
                allBands='A', quiet=True, A=ras)

    # 2m temperature
    ras = os.path.join(cfg.raw.dyn, '2m_temperature.tif')
    gs.map_calc(ras, 'A-273.15', out_path)
    out_file = gs.context_file(ras, out_path)
    if not os.path.exists(out_file):
        gs.Calc('A-273.15', out_file, creation_options=gs.CREATION,
                allBands='A', quiet=True, A=ras)

    # 10m wind speed
    ras = os.path.join(cfg.raw.dyn, '10m_wind_speed.tif')
    out_file = gs.context_file(ras, out_path)
    if not os.path.exists(out_file):
        shutil.copy2(ras, out_file)

    # surface net radiation
    rasters = ['surface_net_solar_radiation.tif',
               'surface_net_thermal_radiation.tif']
    rasters = [os.path.join(cfg.raw.dyn, r) for r in rasters]
    out_file = os.path.join(cfg.tif.dyn, 'surface_net_radiation.tif')
    gs.map_calc(rasters, '(A+B)/86400', out_file)

    # leaf area index
    rasters_dyn = ['leaf_area_index_high_vegetation.tif',
                   'leaf_area_index_low_vegetation.tif']
    rasters_sta = ['high_vegetation_cover.tif', 'low_vegetation_cover.tif']
    rasters = [os.path.join(cfg.raw.dyn, r) for r in rasters_dyn] + \
        [os.path.join(cfg.raw.sta, r) for r in rasters_sta]
    out_file = os.path.join(cfg.tif.dyn, 'leaf_area_index.tif')
    n_band = gdal.Open(rasters[0]).RasterCount
    iter_idxs = np.repeat(np.arange(1, 1 + n_band).reshape(-1, 1), 2, axis=1)
    stat_idxs = np.ones(iter_idxs.shape)
    band_idxs = np.concatenate([iter_idxs, stat_idxs], axis=1)[:, [0, 1, 2, 3]]
    gs.map_calc(rasters, 'A*C+B*D', out_file, band_idxs=band_idxs)


def csv_worker(cfg, rasters, basin_id):
    file_path = os.path.join(cfg.csv.dyn, basin_id + '.csv')
    if os.path.exists(file_path):
        return

    filter_sql = f"STAID = '{basin_id}'"
    filter_shp = gs.shp_filter(cfg.shp.basins, filter_sql)
    df = pd.DataFrame()
    for ras in rasters:
        ras_name = os.path.splitext(os.path.basename(ras))[0]
        series = np.squeeze(gs.extract(ras, filter_shp, enlarge=10, ext=basin_id,
                                       stat=True, save_cache=True, new=False))
        df = pd.concat([df, pd.Series(data=series, name=ras_name)], axis=1)

    df.to_csv(file_path, index=False)


def csv_preprocess(cfg):
    basins_id = pd.read_excel(cfg.gages, usecols=['STAID'], dtype=str)['STAID']
    rasters = glob.glob(os.path.join(cfg.tif.dyn, '*.tif'))
    if not os.path.exists(cfg.csv.dyn):
        os.makedirs(cfg.csv.dyn)

    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    with Pool(cpu_count() * 3 // 4) as p:
        p.map(partial(csv_worker, cfg, rasters), basins_id)
    if os.path.exists('cache'):
        shutil.rmtree('cache')
