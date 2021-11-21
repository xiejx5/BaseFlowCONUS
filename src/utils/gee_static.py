import os
import ee
import pandas as pd
import geospace as gs
from osgeo import gdal


def gee_export(cfg, image, filename):
    # export the image within the scope of usa
    ds_eg = gdal.Open(os.path.join(cfg.raw.dyn, '2m_temperature.grib'))
    crs_transform = gs.grid_bound(ds_eg, cfg.shp.usa)[1]
    usa = ee.FeatureCollection('users/' + cfg.gee.username + '/' + cfg.gee.usa)
    if not os.path.exists(cfg.tif.sta):
        os.makedirs(cfg.tif.sta)

    gs.gee_export_tif(image.unmask(cfg.gdal.nodata), filename, crs='EPSG:4326',
                      crs_transform=crs_transform, region=usa.geometry())

    ds = gdal.Open(filename, gdal.GA_Update)
    for i in range(ds.RasterCount):
        ds.GetRasterBand(i + 1).SetNoDataValue(cfg.gdal.nodata)
    ds = None

    # export csv containing the basin average
    basins = ee.FeatureCollection(
        'users/' + cfg.gee.username + '/' + cfg.gee.basins)
    url = gs.gee_export_csv(basins, image, fields=['STAID', '.*mean'], return_url=True)
    df = pd.read_csv(url, dtype={'STAID': str})
    df_gages = pd.read_excel(cfg.gages, usecols=['STAID'], dtype=str)
    df = pd.merge(df_gages, df, how='left', on='STAID')[['mean']].rename(
        columns={'mean': os.path.splitext(os.path.basename(filename))[0]})

    return df


def soilgrids(cfg, band):
    """download and preprocess soilgrids from google earth engine

    https://git.wur.nl/isric/soilgrids/soilgrids.notebooks/-/blob/master/markdown/access_on_gee.md

    Args:
        cfg (config): config in yaml
        band (string): one of ['bdod', 'cfvo', 'clay', 'sand', 'silt']

    Returns:
        DataFrame: the pandas dataframe contains the 'ORDER' and 'mean' fields
    """
    filename = os.path.join(cfg.tif.sta, band + '.tif')
    if os.path.exists(filename):
        return

    gs.gee_initial()
    image = ee.Image("projects/soilgrids-isric/" + band + '_mean')

    # calculate depth-weighted value
    RAW_NAMES = ['B5', 'B10', 'B15', 'B30', 'B40', 'B100']
    RAW_BANDS = ['0-5cm_mean', '5-15cm_mean', '15-30cm_mean',
                 '30-60cm_mean', '60-100cm_mean', '100-200cm_mean']
    RAW_BANDS = [band + '_' + i for i in RAW_BANDS]
    RAW_DICT = {k: image.select(v) for k, v in zip(RAW_NAMES, RAW_BANDS)}
    image = image.expression('5 * B5 + 10 * B10 + 15 * B15 + 30 * B30 + 40 * B40'
                             ' + 100 * B100', RAW_DICT).divide(200)
    image = image.reduceResolution(
        reducer=ee.Reducer.mean(), bestEffort=True, maxPixels=1024)

    return gee_export(cfg, image, filename)


def dataset(cfg, asset, band, resample='mean', scale=1):
    """download dataset from google earth engine

    Args:
        cfg (config): config in yaml
        asset (string): Earth Engine Snippet, e.g. ee.Image("MERIT/DEM/v1_0_3")
        band (string): band to be downloaded
        resample (string or ee.Reducer): bilinear or bicubic or ee.Reducer.mean

    Returns:
        DataFrame: the pandas dataframe contains the 'ORDER' and 'mean' fields
    """
    filename = os.path.join(cfg.tif.sta, band + '.tif')
    if os.path.exists(filename):
        return

    gs.gee_initial()
    try:
        image = ee.Image(asset).select(band)
        image.getInfo()
    except ee.EEException:
        images = ee.ImageCollection(asset).select(band)
        projection = images.first().projection()
        image = images.mean().setDefaultProjection(projection)

    if resample is not None:
        if resample in ['bilinear', 'bicubic']:
            image = image.resample(resample)
        else:
            image = image.reduceResolution(reducer=getattr(ee.Reducer, resample)(),
                                           bestEffort=True, maxPixels=1024)
    if scale != 1:
        image = image.multiply(scale)

    return gee_export(cfg, image, filename)


def slope(cfg, asset):
    filename = os.path.join(cfg.tif.sta, 'slope.tif')
    if os.path.exists(filename):
        return

    gs.gee_initial()
    dem = ee.Image(asset)
    slope = ee.Terrain.slope(dem).reduceResolution(
        reducer=ee.Reducer.mean(), bestEffort=True, maxPixels=1024)

    return gee_export(cfg, slope, filename)


def gee_show_map(fc, image):
    """show a map containing a featurecolleciton and an image

    Args:
        fc (ee.FeatureCollection): e.g. basins
        image (ee.Image): e.g. DEM
    """
    # Set visualization parameters.
    import geemap

    vis = {
        'min': 40,
        'max': 200,
        'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']
    }
    Map = geemap.Map()
    Map.addLayer(image.select('constant'), vis, 'SRTM DEM')
    Map.addLayer(fc, {}, 'GAGES-II Basins')
