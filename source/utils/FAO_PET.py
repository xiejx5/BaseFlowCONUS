import os
import numpy as np
from osgeo import gdal
import geospace as gs
from hydra.experimental import compose, initialize_config_dir


def delta_svp_estimator(t):
    """
    Estimate the slope of the saturation vapour pressure curve at a given
    temperature.
    Based on equation 13 in Allen et al (1998). If using in the Penman-Monteith
    *t* should be the mean air temperature.
    :param t: Air temperature [deg C]. Use mean air temperature for use in
        Penman-Monteith.
    :return: Saturation vapour pressure [kPa degC-1]
    :rtype: float
    """
    tmp = 4098 * (0.6108 * np.exp((17.27 * t) / (t + 237.3)))
    return tmp / np.square(t + 237.3)


def avp_from_tdew(tdew):
    """
    Estimate actual vapour pressure (*ea*) from dewpoint temperature.
    Based on equation 14 in Allen et al (1998). As the dewpoint temperature is
    the temperature to which air needs to be cooled to make it saturated, the
    actual vapour pressure is the saturation vapour pressure at the dewpoint
    temperature.
    This method is preferable to calculating vapour pressure from
    minimum temperature.
    :param tdew: Dewpoint temperature [deg C]
    :return: Actual vapour pressure [kPa]
    :rtype: float
    """
    return 0.6108 * np.exp((17.27 * tdew) / (tdew + 237.3))


def fao56_penman_monteith(net_rad, t, ws, svp, avp, delta_svp, psy, shf=0.0):
    """
    Estimate reference evapotranspiration (ETo) from a hypothetical
    short grass reference surface using the FAO-56 Penman-Monteith equation.
    Based on equation 6 in Allen et al (1998).
    :param net_rad: Net radiation at crop surface [MJ m-2 day-1]. If
        necessary this can be estimated using ``net_rad()``.
    :param t: Air temperature at 2 m height [deg Celsius].
    :param ws: Wind speed at 2 m height [m s-1]. If not measured at 2m,
        convert using ``wind_speed_at_2m()``.
    :param svp: Saturation vapour pressure [kPa]. Can be estimated using
        ``svp_from_t()''.
    :param avp: Actual vapour pressure [kPa]. Can be estimated using a range
        of functions with names beginning with 'avp_from'.
    :param delta_svp: Slope of saturation vapour pressure curve [kPa degC-1].
        Can be estimated using ``delta_svp()``.
    :param psy: Psychrometric constant [kPa deg C]. Can be estimatred using
        ``psy_const_of_psychrometer()`` or ``psy_const()``.
    :param shf: Soil heat flux (G) [MJ m-2 day-1] (default is 0.0, which is
        reasonable for a daily or 10-day time steps). For monthly time steps
        *shf* can be estimated using ``monthly_soil_heat_flux()`` or
        ``monthly_soil_heat_flux2()``.
    :return: Reference evapotranspiration (ETo) from a hypothetical
        grass reference surface [mm day-1].
    :rtype: float
    """
    a1 = (0.408 * (net_rad - shf) * delta_svp /
          (delta_svp + (psy * (1 + 0.34 * ws))))
    a2 = (900 * ws / (t + 273) * (svp - avp) * psy /
          (delta_svp + (psy * (1 + 0.34 * ws))))
    return a1 + a2


def monthly_soil_heat_flux(t_month_prev, t_month_next):
    """
    Estimate monthly soil heat flux (Gmonth) from the mean air temperature of
    the previous and next month, assuming a grass crop.
    Based on equation 43 in Allen et al (1998). If the air temperature of the
    next month is not known use ``monthly_soil_heat_flux2()`` instead. The
    resulting heat flux can be converted to equivalent evaporation [mm day-1]
    using ``energy2evap()``.
    :param t_month_prev: Mean air temperature of the previous month
        [deg Celsius]
    :param t_month2_next: Mean air temperature of the next month [deg Celsius]
    :return: Monthly soil heat flux (Gmonth) [MJ m-2 day-1]
    :rtype: float
    """
    return 0.07 * (t_month_next - t_month_prev)


def monthly_soil_heat_flux2(t_month_prev, t_month_cur):
    """
    Estimate monthly soil heat flux (Gmonth) [MJ m-2 day-1] from the mean
    air temperature of the previous and current month, assuming a grass crop.
    Based on equation 44 in Allen et al (1998). If the air temperature of the
    next month is available, use ``monthly_soil_heat_flux()`` instead. The
    resulting heat flux can be converted to equivalent evaporation [mm day-1]
    using ``energy2evap()``.
    Arguments:
    :param t_month_prev: Mean air temperature of the previous month
        [deg Celsius]
    :param t_month_cur: Mean air temperature of the current month [deg Celsius]
    :return: Monthly soil heat flux (Gmonth) [MJ m-2 day-1]
    :rtype: float
    """
    return 0.14 * (t_month_cur - t_month_prev)


def wind_speed_2m(ws, z):
    """
    Convert wind speed measured at different heights above the soil
    surface to wind speed at 2 m above the surface, assuming a short grass
    surface.
    Based on FAO equation 47 in Allen et al (1998).
    :param ws: Measured wind speed [m s-1]
    :param z: Height of wind measurement above ground surface [m]
    :return: Wind speed at 2 m above the surface [m s-1]
    :rtype: float
    """
    return ws * (4.87 / np.log((67.8 * z) - 5.42))


def svp_from_t(t):
    """
    Estimate saturation vapour pressure (*es*) from air temperature.
    Based on equations 11 and 12 in Allen et al (1998).
    :param t: Temperature [deg C]
    :return: Saturation vapour pressure [kPa]
    :rtype: float
    """
    return 0.6108 * np.exp((17.27 * t) / (t + 237.3))


def atm_pressure(altitude):
    """
    Estimate atmospheric pressure from altitude.
    Calculated using a simplification of the ideal gas law, assuming 20 degrees
    Celsius for a standard atmosphere. Based on equation 7, page 62 in Allen
    et al (1998).
    :param altitude: Elevation/altitude above sea level [m]
    :return: atmospheric pressure [kPa]
    :rtype: float
    """
    tmp = (293.0 - (0.0065 * altitude)) / 293.0
    return np.power(tmp, 5.26) * 101.3


def psy_const(atmos_pres):
    """
    Calculate the psychrometric constant.
    This method assumes that the air is saturated with water vapour at the
    minimum daily temperature. This assumption may not hold in arid areas.
    Based on equation 8, page 95 in Allen et al (1998).
    :param atmos_pres: Atmospheric pressure [kPa]. Can be estimated using
        ``atm_pressure()``.
    :return: Psychrometric constant [kPa degC-1].
    :rtype: float
    """
    return 0.000665 * atmos_pres


def estimate_PET(cfg):
    ds_Rn = gdal.Open(os.path.join(cfg.tif.dyn, 'surface_net_radiation.tif'))
    Rn = ds_Rn.ReadAsArray()
    net_rad = Rn * 86400 / 1000000

    ds_t = gdal.Open(os.path.join(cfg.tif.dyn, '2m_temperature.tif'))
    t = ds_t.ReadAsArray()

    ds_tdew = gdal.Open(os.path.join(
        cfg.tif.dyn, '2m_dewpoint_temperature.tif'))
    tdew = ds_tdew.ReadAsArray()

    t_month_prev = ds_t.ReadAsArray()
    t_month_prev = np.r_[t_month_prev[11:12], t_month_prev[:-1]]

    t_month_next = ds_t.ReadAsArray()
    t_month_next = np.r_[t_month_next[1:], t_month_next[-13:-12]]

    ds_wind = gdal.Open(os.path.join(cfg.tif.dyn, '10m_wind_speed.tif'))
    wind_speed_10m = ds_wind.ReadAsArray()

    ds_altitude = gdal.Open(os.path.join(cfg.tif.sta, 'dem.tif'))
    altitude = ds_altitude.ReadAsArray()
    altitude = np.ma.masked_equal(
        altitude, ds_altitude.GetRasterBand(1).GetNoDataValue())

    ws = wind_speed_2m(wind_speed_10m, 10)
    svp = svp_from_t(t)
    avp = avp_from_tdew(tdew)
    delta_svp = delta_svp_estimator(t)
    psy = psy_const(atm_pressure(altitude))
    shf = monthly_soil_heat_flux(t_month_prev, t_month_next)
    pet = fao56_penman_monteith(
        net_rad, t, ws, svp, avp, delta_svp, psy, shf=shf)
    pet[pet.filled(1) < 0] = 0
    pet = pet.filled(cfg.gdal.nodata)

    out_file = os.path.join(cfg.tif.dyn, 'potential_evapotranspiration.tif')
    gs.tif_copy_assign(out_file, ds_t, pet, no_data=cfg.gdal.nodata)
