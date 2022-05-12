import torch
import numpy as np
import pandas as pd
import geospace as gs
from osgeo import gdal
from pathlib import Path
from hydra.utils import instantiate
from src.evaluate import instantiate_model


def simulate(checkpoint):
    model, cfg = instantiate_model(checkpoint)
    device = next(model.parameters()).device

    # simulate
    SIMU_BATCH = 10000
    with torch.no_grad():
        outputs = []
        S, X, norm, ds_eg, grids = load_tif_data(cfg)
        for input in zip(torch.split(S, SIMU_BATCH), torch.split(X, SIMU_BATCH)):
            if isinstance(input, torch.Tensor):
                input = tuple([input.to(device)])
            else:
                input = tuple(i.to(device) for i in input)
            outputs.append(model(*input))
        y_pred = torch.cat(outputs).cpu().numpy().squeeze()
    return predict_tif(cfg, y_pred, norm, ds_eg, grids)


def load_tif_data(cfg):
    dataset = instantiate(cfg.dataset)
    norm = dataset.norm
    S_mean, S_std, X_mean, X_std = norm[0:4]
    shp = cfg.constant.shp.eco[cfg.dataset.eco] if cfg.dataset.eco else cfg.constant.shp.usa

    X_name = pd.read_csv(Path(cfg.dataset.X_dir) / (dataset.gages.iloc[0]['STAID'] + '.csv'), nrows=0).columns
    S_name = pd.read_csv(cfg.dataset.S_dir, dtype={'STAID': str}, nrows=0).columns[-cfg.dataset.input_size_sta:]

    # rows and columns of grids with valid dynamic values
    ds_eg = gdal.Open(gs.extract(str(Path(cfg.constant.tif.dyn) / (X_name[0] + '.tif')),
                                 shp, rect_file='/vsimem/example.tif'))
    band = ds_eg.GetRasterBand(1)
    grids = pd.DataFrame()
    grids['row'], grids['col'] = np.where(band.ReadAsArray() != band.GetNoDataValue())

    # delete grids without valid static values
    S_grids = pd.DataFrame(np.full([grids.shape[0], len(S_name)], np.nan), columns=S_name)
    for i, v in enumerate(S_name):
        ds = gdal.Open(gs.extract(str(Path(cfg.constant.tif.sta) / (v + '.tif')),
                                  shp, rect_file='/vsimem/' + v + '.tif'))
        S_grids[v] = ds.ReadAsArray()[grids['row'], grids['col']]
        S_grids.loc[S_grids[v] == ds.GetRasterBand(1).GetNoDataValue(), v] = np.nan
    idx_valid = ~np.any(np.isnan(S_grids), axis=1)
    grids = grids[idx_valid].reset_index(drop=True)
    S_grids = S_grids[idx_valid].reset_index(drop=True)

    # Construct a grid dataframe by treating the grid as a basin
    idx_beg = ((cfg.dataset.seq_length - 2) // 12 + 1) * 12
    idx_end = ds_eg.RasterCount // 12 * 12
    grids['num_months'] = idx_end - idx_beg
    grids['t'] = np.cumsum(grids['num_months'])
    grids['s'] = grids['t'].shift(1, fill_value=0)

    # S: input of static basin properties
    from_grid = np.digitize(np.arange(grids['t'].iloc[-1]), grids['t'])
    S = S_grids.iloc[from_grid]
    S = (S - S_mean) / S_std
    S = torch.Tensor(S.values.astype('float32'))

    # X: input of dynamic time series
    X = np.zeros((grids['t'].iloc[-1], cfg.dataset.seq_length, cfg.dataset.input_size_dyn))
    for i, v in enumerate(X_name):
        ds = gdal.Open(gs.extract(str(Path(cfg.constant.tif.dyn) / (v + '.tif')),
                                  shp, rect_file='/vsimem/' + v + '.tif'))
        array = ds.ReadAsArray()[idx_beg + 1 - cfg.dataset.seq_length:idx_end,
                                 grids['row'], grids['col']]
        idx_add = np.tile(np.arange(0, cfg.dataset.seq_length), idx_end - idx_beg)
        idx_rep = np.repeat(np.arange(0, idx_end - idx_beg, dtype=int), cfg.dataset.seq_length)
        X[:, :, i] = array[idx_rep + idx_add].flatten('F').reshape(X.shape[0:2])
    X = (X - X_mean) / X_std
    X = torch.Tensor(X.astype('float32'))

    return S, X, norm, ds_eg, grids


def predict_tif(cfg, y_pred, norm, ds_eg, grids):
    if cfg.dataset.eco:
        flow_file = str(Path('saved/simulate') / (cfg.dataset.eco + '.tif'))
    else:
        flow_file = str(Path('saved/simulate') / ('usa.tif'))

    y_mean, y_std = norm[-2:]
    num_months = int(grids.iloc[0, grids.columns.get_loc('num_months')])

    # predict
    y_pred = y_pred * y_std + y_mean
    y_pred = y_pred.reshape(grids.shape[0], num_months)

    flow_ds = gdal.GetDriverByName('GTiff').Create(
        flow_file, ds_eg.RasterXSize, ds_eg.RasterYSize,
        num_months, gdal.GDT_Float64, gs.CREATION)
    flow_ds.SetGeoTransform(ds_eg.GetGeoTransform())
    flow_ds.SetProjection(ds_eg.GetProjectionRef())

    for c in range(1, num_months + 1):
        flow_band = flow_ds.GetRasterBand(c)
        flow = np.full((ds_eg.RasterYSize, ds_eg.RasterXSize),
                       cfg.constant.gdal.nodata, dtype=float)
        flow[grids['row'], grids['col']] = y_pred[:, c - 1]
        flow_band.SetNoDataValue(cfg.constant.gdal.nodata)
        flow_band.WriteArray(flow)

    return flow_file
