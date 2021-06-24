import os
import time
import cdsapi
import shutil
import requests
import geospace as gs
from osgeo import gdal
from osgeo.gdalconst import GA_Update
from concurrent.futures import ThreadPoolExecutor


def download_local_dynamic(cfg):
    """download ERA5 dynamic inputs into local disk

    Args:
        cfg (omegaconfig): configs in yaml
    """
    if not os.path.exists(cfg.raw.dyn):
        os.makedirs(cfg.raw.dyn)

    variable = [
        '10m_wind_speed', '2m_temperature', 'leaf_area_index_high_vegetation',
        'leaf_area_index_low_vegetation', 'snowfall', 'surface_net_solar_radiation',
        'total_precipitation', 'surface_net_thermal_radiation', '2m_dewpoint_temperature'
    ]
    with ThreadPoolExecutor(max_workers=len(variable)) as executor:
        executor.map(lambda x: cds_down(x, cfg), variable)


def download_local_static_vegetation_cover(cfg):
    """download vegetation cover of ERA5

    Args:
        cfg (omegaconfig): configs in yaml
    """
    if not os.path.exists(cfg.raw.sta):
        os.makedirs(cfg.raw.sta)

    # Non-time series
    variable = ['high_vegetation_cover', 'low_vegetation_cover']
    for v in variable:
        out_file = os.path.join(cfg.raw.sta, v + '.grib')
        if not os.path.exists(out_file):
            c = cdsapi.Client()
            c.retrieve(
                'reanalysis-era5-single-levels-monthly-means',
                {
                    'format': 'grib',
                    'product_type': 'monthly_averaged_reanalysis',
                    'variable': v,
                    'year': '2019',
                    'month': '06',
                    'time': '00:00'
                },
                out_file)


def cds_down(v, cfg):
    """batch function to download ERA5 from Climate Data Store (CDS)

    Args:
        v (str): variable to download
        cfg (config): configs in yaml
    """
    out_file = os.path.join(cfg.raw.dyn, v + '.grib')
    if os.path.exists(out_file):
        return

    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': v,
            'year': [
                '1979', '1980', '1981',
                '1982', '1983', '1984',
                '1985', '1986', '1987',
                '1988', '1989', '1990',
                '1991', '1992', '1993',
                '1994', '1995', '1996',
                '1997', '1998', '1999',
                '2000', '2001', '2002',
                '2003', '2004', '2005',
                '2006', '2007', '2008',
                '2009', '2010', '2011',
                '2012', '2013', '2014',
                '2015', '2016', '2017',
                '2018', '2019', '2020',
            ],
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12'
            ],
            'time': '00:00',
            'format': 'grib'
        },
        out_file)


def global_surface_water_worker(CACHE_FOLDER, DATASET_NAME, tile):
    filename = DATASET_NAME + "_" + tile + "v1_3_2020"
    temp_file = os.path.join(CACHE_FOLDER, filename + '.temp')
    tif_file = os.path.join(CACHE_FOLDER, filename + '.tif')
    url = ("http://storage.googleapis.com/global-surface-water/downloads2020/" +
           DATASET_NAME + "/" + filename + '.tif')

    while not os.path.exists(tif_file):
        try:
            s = requests.Session()
            s.head(url, stream=True, timeout=10)
            r = s.get(url, stream=True, timeout=10)
            if r.status_code == 200:
                with open(temp_file, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                os.rename(temp_file, tif_file)
            elif r.status_code == 401:
                print('Authentication Failed')
                break
            elif r.status_code == 404:
                break
        except BaseException:
            time.sleep(1)


def global_surface_water(cfg):
    out_file = os.path.join(cfg.raw.sta, 'water_occurrence.tif')
    if os.path.exists(out_file):
        return

    CACHE_FOLDER = os.path.join(cfg.data_dir, 'cache')
    if not os.path.exists(CACHE_FOLDER):
        print("Creating folder " + CACHE_FOLDER)
        os.makedirs(CACHE_FOLDER)
    DATASET_NAME = 'occurrence'

    tiles_lon, tiles_lat = gs.download_tiles(cfg.shp.usa, 10)
    tiles = [f"{abs(lon)}{'W' if lon < 0 else 'E'}_{abs(lat)}{'S' if lat < 0 else 'N'}"
             for lon, lat in zip(tiles_lon, tiles_lat)]
    with ThreadPoolExecutor(max_workers=len(tiles)) as executor:
        executor.map(lambda x: global_surface_water_worker(CACHE_FOLDER, DATASET_NAME, x), tiles)

    tiles_path = [os.path.join(CACHE_FOLDER, DATASET_NAME + "_" + tile + "v1_3_2020" + '.tif') for tile in tiles]
    ds_mosaic = gdal.BuildVRT('/vsimem/Mosaic.vrt', tiles_path)
    trans = ds_mosaic.GetGeoTransform()
    bound = gs.grid_bound(ds_mosaic, cfg.shp.usa)[0]
    nodata = 255

    option = gdal.WarpOptions(options=gs.CONFIG, creationOptions=gs.CREATION,
                              srcNodata=255, dstNodata=0, outputBounds=bound,
                              xRes=trans[1], yRes=trans[5], resampleAlg=gdal.GRA_NearestNeighbour)
    gdal.Warp(out_file, ds_mosaic, options=option)
    ds = gdal.Open(out_file, GA_Update)
    for i in range(ds.RasterCount):
        band = ds.GetRasterBand(i + 1)
        band.SetNoDataValue(nodata)
        band = None
    ds = None

    gs.masked_outside(cfg.shp.usa, out_file)
    shutil.rmtree(CACHE_FOLDER)
