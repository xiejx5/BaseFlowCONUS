import os
import numpy as np
import pandas as pd
import geospace as gs
from osgeo import gdal
from functools import partial
from omegaconf import DictConfig, OmegaConf
from multiprocessing import Pool, cpu_count


def ERA5_tif_worker(cfg, ras, per_year=True):
    out_file = os.path.join(cfg.tif.sta, os.path.basename(ras))
    if os.path.exists(out_file):
        return
    ds_eg = gdal.Open(ras)
    array = ds_eg.ReadAsArray()
    nodata = ds_eg.GetRasterBand(1).GetNoDataValue()
    array_masked = np.ma.masked_equal(array, nodata)

    time_beg_str = f"{cfg.beg.year}-{'0' + str(cfg.beg.month) if cfg.beg.month < 10 else str(cfg.beg.month)}"
    time_end_str = f"{cfg.end.year}-{'0' + str(cfg.end.month) if cfg.end.month < 10 else str(cfg.end.month)}"
    time_beg = np.datetime64(time_beg_str)
    time_end = np.datetime64(time_end_str) + np.timedelta64(2, 'M')
    time_arr = np.arange(time_beg, time_end, dtype='datetime64[M]').astype('datetime64[D]')
    day_counts = (time_arr[1:] - time_arr[0:-1]).astype(int)

    array_day_mean = np.ma.average(array_masked, axis=0, weights=day_counts)
    if per_year:
        array_day_mean = array_day_mean * 365
    gs.tif_copy_assign(out_file, ds_eg, array_day_mean)


def ERA5_tif(cfg):
    rasters = [os.path.join(cfg.tif.dyn, i + '.tif') for i in
               ['total_precipitation', 'potential_evapotranspiration',
                '2m_temperature', 'snowfall']]
    per_year = [True] * 4
    per_year[2] = False
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    with Pool(min(cpu_count() * 3 // 4, len(rasters))) as p:
        p.starmap(partial(ERA5_tif_worker, cfg), zip(rasters, per_year))

    # read precipitation
    ds_P = gdal.Open(os.path.join(cfg.tif.sta, 'total_precipitation.tif'))
    P = ds_P.ReadAsArray()
    P = np.ma.masked_equal(P, ds_P.GetRasterBand(1).GetNoDataValue())

    # make snow percent
    out_file = os.path.join(cfg.tif.sta, 'snow_percent.tif')
    if not os.path.exists(out_file):
        snow_file = os.path.join(cfg.tif.sta, 'snowfall.tif')
        ds_snow = gdal.Open(snow_file)
        snow = ds_snow.ReadAsArray()
        snow = np.ma.masked_equal(snow, ds_snow.GetRasterBand(1).GetNoDataValue())
        snow_percent = snow / P * 100
        gs.tif_copy_assign(out_file, ds_P, snow_percent)
        ds_snow = None
        os.remove(snow_file)

    # make aridity index tif
    out_file = os.path.join(cfg.tif.sta, 'aridity_index.tif')
    if not os.path.exists(out_file):
        ds_PET = gdal.Open(os.path.join(cfg.tif.sta, 'potential_evapotranspiration.tif'))
        PET = ds_PET.ReadAsArray()
        PET = np.ma.masked_equal(PET, ds_PET.GetRasterBand(1).GetNoDataValue())
        AI = PET / P
        gs.tif_copy_assign(out_file, ds_P, AI)


def ERA5_csv(cfg):
    rasters = ['total_precipitation.tif', 'potential_evapotranspiration.tif',
               'aridity_index.tif', '2m_temperature.tif', 'snow_percent.tif']
    rasters = [os.path.join(cfg.tif.sta, os.path.basename(i)) for i in rasters]
    basins_id = pd.read_excel(cfg.gages, usecols=['STAID'], dtype=str)['STAID']
    return gs.basin_average(cfg.shp.basins, rasters, basins_id, save_cache=True, new=False)


def permeability_tif(cfg):
    perm_name = os.path.splitext(os.path.basename(cfg.shp.perm))[0]
    if os.path.exists(os.path.join(cfg.tif.sta, perm_name + '.tif')):
        return

    # example time series raster
    ds_eg = gdal.Open(os.path.join(cfg.raw.dyn, '2m_temperature.grib'))

    # get study bound and grib file spatial reference
    bound = gs.grid_bound(ds_eg, cfg.shp.usa)[0]
    srs = "+proj=longlat +datum=WGS84 +ellps=WGS84"
    tem_path = cfg.raw.sta
    if not os.path.exists(tem_path):
        os.makedirs(tem_path)
    gs.rasterize(cfg.shp.perm, 'logK_Ferr_', cfg.tif.sta, ds_eg, tem_path,
                 outputBounds=bound, dstSRS=srs, dstNodata=cfg.gdal.nodata)

    # deal with units
    ds_out = gdal.Open(os.path.join(cfg.tif.sta, perm_name + '.tif'), gdal.GA_Update)
    band_out = ds_out.GetRasterBand(1)
    no_data = band_out.GetNoDataValue()
    values = ds_out.ReadAsArray()
    values[values != no_data] = values[values != no_data] / 100
    band_out.WriteArray(values)

    tem_file = os.path.join(tem_path, perm_name + '.tif')
    ds_tem = gdal.Open(tem_file, gdal.GA_Update)
    band = ds_tem.GetRasterBand(1)
    no_data = band.GetNoDataValue()
    values = ds_tem.ReadAsArray()
    values[values != no_data] = values[values != no_data] / 100
    band.WriteArray(values)


def permeability_csv(cfg):
    perm_name = os.path.splitext(os.path.basename(cfg.shp.perm))[0]
    tem_file = os.path.join(cfg.raw.sta, perm_name + '.tif')
    basins_id = pd.read_excel(cfg.gages, usecols=['STAID'], dtype=str)['STAID']
    return gs.basin_average(cfg.shp.basins, tem_file, basins_id)


def water_occurrence_tif(cfg):
    ras_in = os.path.join(cfg.raw.sta, 'water_occurrence.tif')
    ras_out = os.path.join(cfg.tif.sta, 'water_occurrence.tif')

    if os.path.exists(ras_out):
        return

    # example time series raster
    ds_eg = gdal.Open(os.path.join(cfg.raw.dyn, '2m_temperature.grib'))
    t = ds_eg.GetGeoTransform()

    # get study bound and grib file spatial reference
    bound = gs.grid_bound(ds_eg, cfg.shp.usa)[0]

    gs.convert_uint8(ras_in)
    srs = "+proj=longlat +datum=WGS84 +ellps=WGS84"
    option = gdal.WarpOptions(multithread=True, dstSRS=srs,
                              creationOptions=gs.CREATION,
                              outputBounds=bound,
                              xRes=t[1], yRes=t[5],
                              resampleAlg=gdal.GRA_Average,
                              outputType=gdal.GDT_Float64,
                              dstNodata=cfg.gdal.nodata)
    gdal.Warp(ras_out, ras_in, options=option)


def water_occurrence_csv(cfg):
    water_file = os.path.join(cfg.raw.sta, 'water_occurrence.tif')
    basins_id = pd.read_excel(cfg.gages, usecols=['STAID'], dtype=str)['STAID']
    return gs.basin_average(cfg.shp.basins, water_file, basins_id)
