'''
The funcition and workflow related to DataFrame x DEM. e.g. 

- load_gdf: load (standarlize, clip) a datframe by a DEM.
- get_value_point: get the value at the points on DEM. The value can be interpolated or zonal statistics under the footprint.
- get_dh_dem: produce dh (before and after coreg)
- get_dh_by_footprint: a mini version of get_value_point.
- get_dh_by_shift_px_dem: get dh over a DEM by giving the shifing matrix to DEM(instead of getting a matrix by coreg)
- get_dh_by_shift_px_gdf: get dh over a DEM by giving the shifing matrix to DataFrame(instead of getting a matrix by coreg)
- pipeline_compare: comparing different settings for corege, it yield results from get_dh_dem(para_dict).
- best_footprint: get the best footprint size by gradient descending algorithm.
- best_best_shift_px: get the best shift_px by gradient descending algorithm. This function is faster than xdem.NuthKaab coreg if Points are few.
- dem_profiler: get a profile over DEMs by interpolate a dataframe that contains ICESat-2 ground tracks.
- dem_difference: ddem = dem_obs - dem_ref
- dem_difference_plot: plot the difference map and histrogram before and after NuthKaab coreg between dem_ref and dem_obs
- sc_sf_dem: the production version of get_dh_dem, for parallel processing over dems, producing the final results: (1) snowfree dh (2) snowcover dh
- sc_dem: similar to sc_sf_dem. But no Coreg, just update the snow cover dh.

'''

import xarray as xr
from rasterio.enums import Resampling
from audioop import bias
import contextlib
from dataclasses import asdict
from distutils.log import error
from os import stat
from sys import stderr
from unittest import result
import pandas as pd
from pandas import DataFrame
import geopandas as gpd
from geopandas.geodataframe import GeoDataFrame
import numpy as np
from sympy import false, true
from traitlets import Bool
import matplotlib.pyplot as plt
import pprint

import xdem
from xdem.dem import DEM 

import pyproj
from pyproj import Transformer

import shapely.speedups
shapely.speedups.enable()
from shapely.geometry import Point,Polygon,LineString

from xsnow.goplot import normal_statistics, final_histogram
from xsnow.misc import extend_geosegment, interp_points, points_to_footprint,poly_zonal_stats,dem_shift,gdf_shift,df_sampling_from_dem
from noisyopt import minimizeCompass
import importlib

importlib.reload(xdem)

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

def yield_gdf(df:DataFrame or GeoDataFrame,
             dem_fid_list:list,
             z_name='h_te_best_fit',
             gdf:bool = False):
    '''
    Same with load_gdf but yield over a list of dem.

    return a iterator of gdf_subseted
    '''

    for dem in dem_fid_list:
        yield load_gdf(df,xdem.DEM(dem),z_name=z_name,gdf=gdf)

def load_gdf(df:DataFrame or GeoDataFrame,
             dem:DEM,
             z_name='h_te_best_fit',
             gdf: bool = True):
    '''
    Normalize the dataset from ICESat-2 into the one could be processed easily:
    (1) project into grid, and subset according dem
    (2) (if not geodataframe) turn into geodataframe, and adding crs
    (3) decide which h to be used in coregistration by z_name: 'h_te_interp','h_te_best_fit'...
    (4) mask out nodata according to dem msak.
    
    :param df: DataFrame or GeoDataFrame
    :param dem: xDEM.DEM
    :param z_name: string, 'h_te_interp','h_te_best_fit','h_te_mean'...
    :param gdf: return GeoDataFrame

    return a subseted geodataframe after (1)(2)(3)(4).

    e.g.
    sf_gdf_subset = load_gdf(sf_gdf,dem,z_name='h_te_best_fit')
    '''
    if not isinstance(dem,DEM):
        dem = xdem.DEM(dem)

    # step 1: CRS
    # if there is E,N the DEM must be 32633 as well
    if 'E' in df.columns and 'N' in df.columns:
        if dem.crs not in [pyproj.CRS(32633),pyproj.CRS(25833)]:
            dem = dem.reproject(dst_crs=pyproj.CRS(32633))
    # else, DEM not need be 32633, but turn gdf into DEM.crs
    else:
        transformer = Transformer.from_crs(pyproj.CRS(4326),dem.crs)
        df['E'],df['N'] = transformer.transform(df['latitude'],df['longitude'])
  

    # step 2: subset by DEM square
    [L, B, R, T] = dem.bounds
    df_subset = df.query(f'{B} < N < {T} and {L} < E <{R}').copy()

    # step 3: if there are points, and there is anyone mask out
    if len(df_subset) > 100 and dem.data.mask.any():
        # if there is false (the mask out area)

        final_mask = ~dem.data.mask
        pts = np.array((df_subset['E'].values,df_subset['N'].values)).T
        mask_raster = dem.copy(new_array=final_mask.astype(np.float32))
        ref_inlier = mask_raster.interp_points(pts, input_latlon=False, order=0)
        ref_inlier = ref_inlier.astype(bool)
        df_subset = df_subset[ref_inlier]
        
    # step 4 : turn into gdf if need
    if gdf:
        if not isinstance(df_subset,GeoDataFrame):
            geometry = [Point(xy) for xy in zip(df_subset.E, df_subset.N)]
            df_subset = gpd.GeoDataFrame(df_subset, geometry=geometry,crs=dem.crs)
        if df_subset.crs.equals(dem.crs):
            df_subset = df_subset.to_crs(dem.crs)
        if df_subset.geometry.is_valid.sum() != len(df_subset):
            print('Warning: Invalid geometry detected')
    if z_name:
        df_subset['z'] = df_subset[z_name] # h_te_interp

    return df_subset

def get_value_point(dem: DEM, 
                    gdf:GeoDataFrame, 
                    z_name= False, attribute=None, order=1, footprint=False, window_size = 3,
                    stats = None or list,area_or_point= 'Area'):
    '''
    get value(h or other attributes) at the point (E,N) from DEM. Method 
    (1) interpolation by scipy map_coordinates. Used for 10m DEM or below.
    (2) zonal statistics by rasterstats. Used for higher resolution DEMs.

    :param dem: xDEM.DEM
    :param gdf: GeoDataFrame after load_gdf, has 'E','N', 'region'(if there is a footprint zonal statistics)
    :parameter z_name: string, i.e 'h_te_interp','h_te_best_fit','h_te_mean'...
    :param attribute: i.e ['slope','aspect','planform_curvature', 'profile_curvature']
    :param order: algorithem of interpolation 0 nearest, 1 linear, 2 cubic, works if no footprint...
    :param footprint: tuple, (width,length), pts locate at the middle of footprint...
    :param stats: string, 'mean','median','min','std','sum','nodata'...works when there is footprint
    :param area_or_point: = 'Area' by default, which means the offset = 'ul', or 'Point', offset = 'center'

    https://pythonhosted.org/rasterstats/manual.html#vector-data-sources

    return a dict of values with keys: 'h' -> np.ndarray
                                       'dh'(if give z_name) -> np.ndarray
                                       'attribute' if not None.
                                       'stats' if not None.
    '''

    if stats is None:
        stats = ["mean",'median','std','count']

    if (area_or_point is None) and ('AREA_OR_POINT' in dem.tags):
        try:
            area_or_point = dem.tags['AREA_OR_POINT']
        except (KeyError, AttributeError):
            area_or_point = 'Area'

    # prepared pts:
    x_coords, y_coords = (gdf['E'].values,gdf['N'].values)
    pts = np.array((x_coords, y_coords)).T
    pts_return = {}
    
    # Terrain parameters
    if attribute:
        dem_attr = xdem.terrain.get_terrain_attribute(dem, attribute=attribute, window_size=window_size)
        attr_value = [interp_points(i,pts,input_latlon=False, order=order) for i in dem_attr]
        # pass the results to return_dict
        for i,j in zip(attribute,attr_value):
            pts_return[i] = j

    # if there is footprint, do method (2) zonal stats and return statistic values about 'h'
    if footprint and stats:
        # extend a list of point into a list of polygon. Requires gdf.region_ang
        poly_list = points_to_footprint(gdf,footprint)
        # do zonal statistics inside the polygon
        pts_stats = poly_zonal_stats(poly_list,dem,stats=stats)
        # pass the results (array) to return_dict
        pts_return['h'] = pts_stats[stats[0]].values
        for i in stats:
            pts_return[i] = pts_stats[i]
    else:
        # else retrieving results (array) by method(1) interpolation.
        ## here the 'mode' is not interpolation algorithem but the edge control
        ## instead the 'order' control the interpolation algorithem 0 nearest, 1 linear, 2 cubic,
        pts_return['h'] = interp_points(dem,pts,order=order,mode='nearest',area_or_point=area_or_point)

    if z_name:
        pts_return['dh']  = gdf[z_name].values - pts_return['h']
    return pts_return

def get_dh_dem(tba_dem: DEM,sf_gdf:GeoDataFrame, 
               inlier_mask=None, verbose=True, 
               d_lim=100, std_t=3, perc_t =99, range=(-20,20), downsampling=8000,
               pp=None, attribute= None, stats = None, mask_highcurv:bool = True, weight = None, shift_median = True, geosegment=False,
               order=1, footprint:bool = False, moving_avrage:bool=False, coreg = 'GradientDescending',shift_dem:bool=False,**kwargs):
    '''
    get dh from DEM before and after co-registration.
    
    Different methods to get values from DEMs: 
    (1) interpolation by scipy map_coordinates, and then co-registration. e.g.
        result = get_dh_dem(dem,sf_gdf,coreg='GradientDescending')

    (2) zonal statistics by rasterstats, no co-registration. Used for higher resolution DEMs. e.g.
        result = get_dh_dem(dem,sf_gdf,footprint=(13,39),coreg=False)

    (3) Do moving average on DEM and then (1). Used to degrade high resolution DEMs by a window. e.g.
        result = get_dh_dem(dem,sf_gdf, moving_avrage=(13,39),coreg='NuthKaab')

    :compulsory:
    :param dem: xDEM.DEM
    :param gdf: GeoDataFrame, after load_gdf, has 'E','N'. 'z', and 'z' are used for coregistration. dh calculation.

    :statistics: Does not affect on results, just for plot.
    :param d_lim: = 100 by default, means remove dh when it > 100 m. (1)
    :param perc_t: = 99 by default, means remove the last 1% quantities. (2)
    :param std_t: = None by default, means remove dh when it > 3 std. Only suggest using (1+2) or (1+3)
    :param range: =(-20,20) by default, means the xlim of the axis
    :param pp: string. If pp, save the plot as a jpg to root directory.

    :others:
    :param inlier_mask (valide for NuthKaab): use to mask out moving terrain, such as glaciers, during coreg.
    :param verbose (valide for NuthKaab): the detailed info for coreg ...
    :param attribute: list like ['slope','aspect','planform_curvature', 'profile_curvature'], append these values to 'gdf'.
    :param stats (valide for footprint): string, 'mean','median','min','std','sum','nodata'...works when there is footprint
    :param mask_highcurv (valide for NuthKaab): use to mask out highcurv=5 during coreg, need extra computation (curvature). If on, more robost coreg.
    :param order: algorithem of interpolation: 0 nearest, 1 linear, 2 cubic, works if no footprint...
    :param footprint: tuple, (width,length), pts locate at the middle of footprint...
    :param moving_avrage (valide for NuthKaab): False or (width,length). If not False, there is a moving average on DEM before coregistration.

    return a dict of values with keys: 'gdf': check gdf['dh_before'] and gdf['dh_after'],
                                       'dem': aligned dem,
                                       'sum': statistics metrics,
                                       'shift_matrix': the parameters used in coreg.
    '''
    if stats is None:
        stats = ['mean']

    # (1) original dh
    sf_gdf['dh_before']  = get_value_point(tba_dem,sf_gdf,order=order,footprint=footprint,stats=stats,z_name='z')['dh']

    # (2) Co-registration and get shift_matrix
    if not coreg:
        shift_matrix = (0,0,0)
    elif coreg == 'NuthKaab':
        # fit dem with snow free measurements sf_gdf['z'] to get aligned dem
        func = xdem.coreg.NuthKaab()
        func.fit_pts(sf_gdf, tba_dem, inlier_mask,verbose=verbose,mask_highcurv=mask_highcurv,order=order,moving_avrage=moving_avrage)
        shift_matrix = (func._meta["offset_east_px"],func._meta["offset_north_px"],func._meta["bias"])
    elif coreg == 'GradientDescending':
        shift_matrix = best_shift_px(tba_dem,sf_gdf,x0=(0,0),footprint=False,bounds=(-3,3),z_name='z',disp=False,stat='nmad',perc_t=perc_t,std_t=std_t,geosegment=geosegment,downsampling=downsampling,weight=weight)
    else:
        print('Coreg needs to be \'NuthKaab\',\'GradientDescending\' or False')

    res = tba_dem.res[0]
    sf_gdf['coreg_bias'] = shift_matrix[2]

    # (3) applying shift matrix (to DEM or to gdf) and get dh.
    shift_median = shift_matrix[2] if shift_median is not False else 0

    if shift_dem:
        aligned_dem = dem_shift(tba_dem,shift_matrix[0],shift_matrix[1],shift_median)
        pts = get_value_point(aligned_dem,sf_gdf,attribute=attribute,order=order,footprint=footprint,stats=stats,z_name='z')
    else:
        gdf_shifted = gdf_shift(sf_gdf,shift_matrix[0]*res, shift_matrix[1]*res, shift_median, z_name='z')
        pts = get_value_point(tba_dem, gdf_shifted, attribute=attribute,order=order,footprint=footprint,stats=stats,z_name='z')
        aligned_dem = tba_dem

    sf_gdf['dh_after'] = pts['dh']

    if attribute is not None:
        for i in attribute:
            sf_gdf[i] = pts[i]
    if pp is not None:
        sf_gdf['fid'] = pp

    # (4) return and export the statistics into a dict
    N = len(sf_gdf['z'])
    stats_0,stats_1 = final_histogram(sf_gdf['dh_before'],sf_gdf['dh_after'],d_lim=d_lim,pp=pp,std_t=std_t,perc_t=perc_t,range=range,offset=shift_matrix[0:2]);
    [L,B,R,T] = tba_dem.bounds

    sum_ = {'mean_before':stats_0[0],
            'mean_after':stats_1[0],
            'median_before':stats_0[1],
            'median_after':stats_1[1],
            'std_before':stats_0[2],
            'std_after':stats_1[2],
            'rmse_before':stats_0[3],
            'rmse_after':stats_1[3],
            'nmad_before':stats_0[5],
            'nmad_after':stats_1[5],
            'n_before':stats_0[4],
            'n_after':stats_1[4],
            'n<05_before':len(sf_gdf.query('-0.5< dh_before < 0.5'))/N*100,
            'n<05_after':len(sf_gdf.query('-0.5< dh_after < 0.5'))/N*100,
            'n<1_before':len(sf_gdf.query('-1< dh_before < 1'))/N*100,
            'n<1_after':len(sf_gdf.query('-1< dh_after < 1'))/N*100,
            'n<2_before':len(sf_gdf.query('-2< dh_before < 2'))/N*100,
            'n<2_after':len(sf_gdf.query('-2< dh_after < 2'))/N*100,
            'N':N,
            'bond_L':L,
            'bond_B':B,
            'bond_R':R,
            'bond_T':T,
            'geometry':Polygon(((L,B),
                                (R,B),
                                (R,T),
                                (L,T))),
            'fid':pp,
            'mask_highcurv':mask_highcurv,
            'order':order,
            'footprint':footprint,
            'bias':shift_matrix[2],
            'resolution':res,
            'offset_east_px':shift_matrix[0],
            'offset_north_px':shift_matrix[1],
            'coreg':coreg,
            'shift_median':shift_median}

    # return the result
    results = {'gdf':sf_gdf,'dem':aligned_dem,'sum':sum_,'shift_matrix':shift_matrix}

    if verbose:
        pp = pprint.PrettyPrinter(depth=3)
        pp.pprint(sum_,compact=True)
    return results

def pipeline_compare(*para_list):
    '''
    For comparing different settings for corege, it yield results from get_dh_dem(para_dict) e.g.:

    results_generator = pipeline_compare(para_dict_a,para_dict_b...)
    results = list(results_generator)

    where, para_dict should be define first, e.g.:

    para_dict_a ={'tba_dem':dem,'sf_gdf':sf_gdf,'inlier_mask':None,'d_lim':100, 'std_t':None, 'perc_t':99, 'range':(-20,20), 
               'pp':None, 'attribute':None, 'stats':None, 'mask_highcurv':True, 'order':1, 'footprint':False, 'moving_avrage':False, 'coreg':'GradientDescending','shift_dem':False}
    para_dict_b ={'tba_dem':dem,'sf_gdf':sf_gdf,'inlier_mask':None,'d_lim':100, 'std_t':None, 'perc_t':99, 'range':(-20,20), 
               'pp':None, 'attribute':None, 'stats':None, 'mask_highcurv':True, 'order':1, 'footprint':False, 'moving_avrage':False, 'coreg':'GradientDescending','shift_dem':False}

    :param sf_gdf: DataFrame or GeoDataFrame 
    :param tba_dem: xdem.DEM after vertical reference correction.
    :param z_name: the column used as the elevation of ICESat-2

    :statistics part:
    :param d_lim: (1) d_lim = 100, filter dh when it > d_lim
    :param perc_t: (2) perc_t = 99 by default, filter dh locates out of 1%-99% after (1). set perc_t = 100 to disable this function.
    :param std_t: (3) std = None by default, filter dh when it > 3 std after (1) and (2). Only suggest one of (2) or (3).
    :param range: the xlim of the axis of the histgram

    :others:
    :param attribute: like ['slope','aspect','planform_curvature', 'profile_curvature'], append these values to 'gdf'.
    :param stats: string, 'mean','median','min','std','sum','nodata'...works when there is footprint
    :param mask_highcurv: use to mask out highcurv=5 during coreg, need extra computation (curvature). If on, more robost coreg.
    :param footprint: tuple, (width,length), pts locate at the middle of footprint...
    
    '''

    for para_dict in para_list:

        if {'sf_gdf', 'tba_dem'} > set(para_dict):
            raise KeyError("Invalid parameters for coregistration and dh calculation")

        gdf_subset = load_gdf(para_dict['sf_gdf'],para_dict['tba_dem'])

        if len(gdf_subset) > 100:
            para_dict['sf_gdf'] = gdf_subset
            yield get_dh_dem(**para_dict)
        else:
            print('The points less than 100. Check DEM or dataframe.')


def get_dh_by_footprint(dem:DEM,sf_gdf_subset:DataFrame,
                        footprint,z_name='h_te_best_fit',s_name='mean',stat='nmad',perc_t=99.75):
    '''
    get the dh/NMAD value of the footprint, acting like a lost function, e.g.:

    get_dh_by_footprint(dem_1,sf_gdf_subset,(11,31))

    :param dem_1: xdem.DEM after vertical reference correction.
    :param sf_gdf_subset: DataFrame or GeoDataFrame.
    :param footprint: (width, length)
    :param s_name: the value from zonal statistics: 'mean','median','std','count','min','max'
    :param z_name: the value from Icesat-2: 'h_te_best_fit', 'h_te_mean', 'h_te_min', 'h_te_max', 'h_te_mode', 
                  'h_te_median', 'h_te_interp', 'h_te_skew', 'h_te_std', 'h_te_uncertainty'.
    return NMAD or pts
    '''
    mm,nn = footprint
    pts = get_value_point(dem=dem,gdf=sf_gdf_subset,z_name=z_name,
                        attribute=None,order=1,stats=["mean",'median','std','count','min','max'],footprint=(mm,nn))
    
    res = {}
    res['mean'],res['median'],res['std'],res['rmse'],res['num'],res['nmad'] = normal_statistics(sf_gdf_subset[z_name].values -pts[s_name],perc_t=perc_t) # best_fit - mean
    
    if stat =='mix':
        res['mix'] = (2 * res['nmad'] + res['rmse'])/3
        return res['mix']
    elif stat == 'all':
        return res
    elif stat in ['nmad','rmse','median','std']:
        return res[stat]
    else:
        return pts

def get_dh_by_shift_px_dem(dem:DEM,sf_gdf_subset:DataFrame,
                           shift_px,shift_bias=0,z_name='h_te_best_fit',footprint=False,stat='nmad',perc_t=99.75):
    '''
    get the dh/NMAD value of the shift matrix (one of the coreg concept) on DEM, acting like a lost function:
    
    e.g.:
    nmad_by_shift_px_gdf(dem_1,sf_gdf_subset,(0,0))

    :param dem: xdem.DEM after vertical reference correction.
    :param sf_gdf_subset: DataFrame or GeoDataFrame.
    :param footprint: (width, length).

    return statistics: one of the NMAD, median, std, rmse, mean. OR
    return pts: h and dh at the points.
    '''
    e_px,n_px = shift_px

    dem_shifted = dem_shift(dem,e_px,n_px,shift_bias)
    pts = get_value_point(dem=dem_shifted,gdf=sf_gdf_subset,z_name=z_name,
                        attribute=None,order=1,footprint=footprint)

    res = {}
    res['mean'],res['median'],res['std'],res['rmse'],res['num'],res['nmad'] = normal_statistics(pts['dh'],perc_t=perc_t)
        
    if stat =='mix':
        res['mix'] = (2 * res['nmad'] + res['rmse'])/3
        return res['mix']
    elif stat == 'all':
        return res
    elif stat in ['nmad','rmse','median','std']:
        return res[stat]
    else:
        return pts


def get_dh_by_shift_px_gdf(dem:DEM or xr.DataArray or xarray.Dataset,sf_gdf_subset:DataFrame or xr.DataArray,
                           shift_px,shift_bias=0,z_name='h_te_best_fit',downsampling=False,
                           weight=False, footprint=False, stat:str='nmad' or bool, perc_t=99.75, std_t=None):
    '''
    get the dh/NMAD value of the shift matrix (one of the coreg concept) on gdf, acting like a lost function:
    
    e.g.:
    nmad_by_shift_px_gdf(dem,sf_gdf_subset,(0,0))

    :param dem: xdem.DEM after vertical reference correction.
    :param sf_gdf_subset: DataFrame or GeoDataFrame. To be shift.
    :param footprint: (width, length).
    :param downsampling: False by default. Or can set it into a int.
    :param weight: a column, or you can set a mask.

    return statistics: one of the NMAD, median, std, rmse, mean.  OR
    return pts: h and dh at the points.
    '''

    # shift ee,nn
    if isinstance(dem,DEM):
        ee,nn = [i * dem.res[0] for i in shift_px]
    else:
        ee,nn = shift_px

    if downsampling and (len(sf_gdf_subset) > downsampling):
        sf_gdf_subset = sf_gdf_subset.sample(frac=downsampling/len(sf_gdf_subset),random_state=42).copy()
        print('Running get_dh_by_shift_px_gdf on downsampling. The length of the gdf:',len(sf_gdf_subset))
        #print('Set downsampling = other value or False to make a change.')

    gdf_shifted = gdf_shift(sf_gdf_subset,ee,nn,shift_bias,z_name=z_name)
    
    if isinstance(dem,DEM) and isinstance(sf_gdf_subset,DataFrame):
        pts = get_value_point(dem=dem,gdf=gdf_shifted,z_name=z_name,
                            attribute=None,order=1,footprint=footprint)

        weight_ = sf_gdf_subset[weight] if weight else 1

        res = {}
        res['mean'],res['median'],res['std'],res['rmse'],res['num'],res['nmad'] = normal_statistics(pts['dh'] * weight_, perc_t = perc_t, std_t = std_t)
    elif isinstance(sf_gdf_subset,xr.DataArray):
        
        weight_ = sf_gdf_subset[weight] if weight in sf_gdf_subset else 1

        # difference between dem and dem_shifted, two xr.DataArray
        dh = dem - gdf_shifted.rio.reproject_match(dem,resampling = Resampling.bilinear)
        res = {}
        res['mean'],res['median'],res['std'],res['rmse'],res['num'],res['nmad'] = normal_statistics(dh, perc_t = perc_t, std_t = std_t)

    if stat =='mix':
        res['mix'] = (2 * res['nmad'] + res['rmse'])/3
        return res['mix']
    elif stat == 'all':
        return res
    elif stat in {'nmad', 'rmse', 'median', 'std'}:
        return res[stat]
    else:
        return pts

def best_footprint(dem:DEM,sf_gdf_subset:DataFrame,
                   x0=(13,26),bonds=([8,16],[8,100]),z_name='h_te_best_fit',s_name='mean',stat='nmad',perc_t=99.75,disp=False)-> tuple:
    '''
    get the best foorprint by gradient decending algorithm from noisyopt.
    
    e.g.:
    best_footprint(dem,sf_gdf_subset)

    :param dem: xdem.DEM after vertical reference correction.
    :param sf_gdf_subset: DataFrame or GeoDataFrame.
    :param x0: = (width, length), the initial footprint.
    :param bounds: = ([8,16],[8,100]), the max px of searching the best fit.
    :param perc_t: = 99 means remove points at the 1% quntities, this will help gradient descending. Set = 100 to disable it.
    
    :return the best fit footprint and NMAD: (m,n,NMAD)
    '''
    func_x = lambda x: get_dh_by_footprint(dem,sf_gdf_subset,x,z_name=z_name,s_name=s_name,stat=stat,perc_t=perc_t)
    res = minimizeCompass(func_x, x0=x0, deltainit=2,deltatol=0.006,feps=0.0001,errorcontrol=False,bounds=bonds,disp=disp)
    print(f'the best fit footprint(width,length): ({res.x[0]:.4f},{res.x[1]:.4f})')
    return res.x[0],res.x[1],res.fun

def best_shift_px(dem:DEM or xr.array,sf_gdf_subset:DataFrame or xr.array,
                  x0=(0,0),footprint=False,bounds=(-3,3),z_name='h_te_best_fit',weight=False,
                  disp=False, mask_out=True, downsampling=6000,geosegment=False,stat='nmad',perc_t=99.75,std_t=3,
                  deltainit=2,deltatol=0.006,feps=0.0002)-> tuple:
    '''
    get the best shift matrix by gradient decending algorithm from noisyopt.

    - Bias becasue the median may not robust if sampling points are few.
    
    e.g.:
    best_shift_px(dem,sf_gdf_subset)

    :param dem: xdem.DEM after vertical reference correction.
    :param sf_gdf_subset: DataFrame or GeoDataFrame.
    :param x0: (width, length), the initial footprint. Speeding up by give the good initial x0.
    :param downsampling: True by default.
    :param weight: a column.
    :param bounds: the max px of searching the best fit.
    :param stat: one of the NMAD, median, std, rmse, mean as a lost function
    :param mask_out: = True by default. If using datafram after load_df set it False. If the points are
                     sampled from ref_dem, set it True to mask out the invalide value from dem_tba

    return the best fit: (e_px,n_px,bias,stat)

    The algorithm terminates when the current iterate is locally optimally at the target pattern size 'deltatol' 
    or when the function value differs by less than the tolerance 'feps' along all directions.

    '''
    sf_gdf_subset_all = sf_gdf_subset

    # to speed up by downsampling. Here I am not sure is 25000 okay, but it still fast!    
    
    if downsampling and len(sf_gdf_subset) > downsampling:
        sf_gdf_subset = sf_gdf_subset.sample(frac=downsampling/len(sf_gdf_subset),random_state=42).copy()
        print('Running best_shift_px on downsampling. The length of the gdf:',len(sf_gdf_subset))
        #print('Set downsampling = other value or False to make a change.')
    elif geosegment and len(sf_gdf_subset) < downsampling:
        # need extend
        sf_gdf_subset = extend_geosegment(sf_gdf_subset,dem)

    # start iteration, find the best shifting px
    func_x = lambda x: get_dh_by_shift_px_gdf(dem,sf_gdf_subset,x,footprint=footprint,z_name=z_name,stat=stat,perc_t=perc_t,std_t=std_t,weight=weight)
    res = minimizeCompass(func_x, x0=x0, deltainit=deltainit,deltatol=deltatol,feps=feps,bounds=(bounds,bounds),disp=disp,errorcontrol=False)
    
    # Send the best solution to find all results
    result = get_dh_by_shift_px_gdf(dem,sf_gdf_subset_all,(res.x[0],res.x[1]),footprint=footprint,z_name=z_name,stat='all',perc_t=perc_t,std_t=std_t)
    print('Gradient Descending Coreg fit matrix(e_px,n_px,bias),nmad:({:.4f},{:.4f},{:.4f}),{:.4f}'.format(res.x[0],res.x[1],result['median'],result['nmad']))

    return res.x[0],res.x[1],result['median'],result['nmad'],result['rmse'],result['std']

def best_shift_px_scipy(dem:DEM or xr.array,sf_gdf_subset:DataFrame or xr.array,
                        x0=(0,0),footprint=False,bounds=(-3,3),z_name='h_te_best_fit',
                        disp=False, downsampling=6000,geosegment=False,stat='nmad',perc_t=99.75,std_t=None)-> tuple:
    '''
    get the best shift matrix by gradient decending algorithm from scipy.

    - Bias becasue the median may not robust if sampling points are few.
    
    e.g.:
    best_shift_px(dem,sf_gdf_subset)

    :param dem: xdem.DEM after vertical reference correction.
    :param sf_gdf_subset: DataFrame or GeoDataFrame.
    :param x0: (width, length), the initial footprint. Speeding up by give the good initial x0.
    :param downsampling: True by default.
    :param bounds: the max px of searching the best fit.
    :param stat: one of the NMAD, median, std, rmse, mean as a lost function

    return the best fit: (e_px,n_px,bias,stat)

    The algorithm terminates when the current iterate is locally optimally at the target pattern size 'deltatol' 
    or when the function value differs by less than the tolerance 'feps' along all directions.

    '''
    from scipy.optimize import minimize

    sf_gdf_subset_all = sf_gdf_subset

    # to speed up by downsampling. Here I am not sure is 25000 okay, but it still fast!    
    if downsampling and len(sf_gdf_subset) > downsampling:
        sf_gdf_subset = sf_gdf_subset.sample(frac=downsampling/len(sf_gdf_subset),random_state=42).copy()
        print('Running best_shift_px on downsampling. The length of the gdf:',len(sf_gdf_subset))
        #print('Set downsampling = other value or False to make a change.')
    elif geosegment and len(sf_gdf_subset) < downsampling:
        # need extend
        sf_gdf_subset = extend_geosegment(sf_gdf_subset,dem)
        
    # start iteration, find the best shifting px
    func_x = lambda x: get_dh_by_shift_px_gdf(dem,sf_gdf_subset,x,footprint=footprint,z_name=z_name,stat=stat,perc_t=perc_t,std_t=std_t)
    res = minimize(func_x, x0=x0, method='L-BFGS-B',bounds=(bounds,bounds))
    
    # Send the best solution to find all results
    result = get_dh_by_shift_px_gdf(dem,sf_gdf_subset_all,(res.x[0],res.x[1]),footprint=footprint,z_name=z_name,stat='all',perc_t=perc_t,std_t=std_t)
    print('Gradient Descending Coreg (L-BFGS-B) fit matrix(e_px,n_px,bias),nmad:({:.4f},{:.4f},{:.4f}),{:.4f}'.format(res.x[0],res.x[1],result['median'],result['nmad']))

    return res.x[0],res.x[1],result['median'],result['nmad'],result['rmse'],result['std']


from shapely import wkt

def dem_profiler(dem_dict,df,date:int,pair,beam,sample_distance,order=1,footprint=False,fillhole=True,stats='mean') -> pd.DataFrame:

    '''
    get a profile over DEMs by interpolate a dataframe that contains ICESat-2 ground tracks.

    :param dem_dict: a dict = {'name_key':xDEM.DEM,'name_key':xDEM.DEM}
    :param df: DataFrame after load_gdf (subset to the bonds of the DEM)
    
    :param date: int.
    :param pair: int. Could be 1,2,3. Pair of beam 1,2,3 from right to left.
    :param beam: int. Coule be 0,1. Strong and Weak beam.
    :param sample_distance: int = 1.
    
    :param order: algorithem of interpolation 0 nearest, 1 linear, 2 cubic, works if no footprint...
    :param footprint_size: int, pts locate at the middle of footprint. Use it if you want downsampling DEM.
    :parameter stats: string, 'mean','median','min','std','sum','nodata'...works when there is footprint
    :fillhole: bool = Ture by default. If DataFrame has been processed by load_gdf set it False to speed up.
    return a datafram after interpolation and dem_h profiling.
    '''
    if not isinstance(df,GeoDataFrame) and isinstance(df['geometry'][0],str):
        df['geometry'] = df['geometry'].apply(wkt.loads)
        # Create a GeoDataFrame from the DataFrame
        df = gpd.GeoDataFrame(df, geometry='geometry')

    # generate a line. Decending from North to south.
    df_line = df[(df.date == date) & (df.pair == pair) & (df.beam == beam)].reset_index().sort_values('N',ascending=False)
    assert len(df_line) > 1,f'The number of ICESat-2 observations is {len(df_line)}'
    line = LineString(df_line.geometry.values)

    # the distance from ICESat-2 points
    df_line['Distance'] = [line.project(Point(point)) for point in line.coords]

    # interpolate the line, and also append the points from ICESat-2
    distances = np.append(np.arange(0, line.length, sample_distance), df_line['Distance'].values)
    sample_points = [line.interpolate(d) for d in distances]

    # put the results into a dataframe
    df_profile = pd.DataFrame({'Distance':distances,
                               'E':[i.coords[0][0] for i in sample_points],
                               'N':[i.coords[0][1] for i in sample_points]})

    # do some calculation for DEM height profile
    if footprint:
        f_width = footprint[0]
        f_length = footprint[1]
        # poly_list = [point.buffer(footprint_size/2) for point in sample_points]
        poly_list = [LineString([line.interpolate(d-f_length/2),line.interpolate(d+f_length/2)]).buffer(f_width/2) for d in distances]

        # interp_points
    x_coords, y_coords = (df_profile['E'].values,df_profile['N'].values)
    pts = np.array((x_coords, y_coords)).T

    for name, dem in dem_dict.items():
        if footprint: # Could be other condition e.g. acoording to the name
            # priority to extract value from DEM with footprint
            pts_stats = poly_zonal_stats(poly_list,dem,stats=stats)
            df_profile[f'dem_h_{name}'] = pts_stats[stats].values
        else:
            # else do interp_points
            df_profile[f'dem_h_{name}'] = interp_points(dem,pts,order=order,fillhole=fillhole,mode='constant',cval=np.nan)

    return pd.merge(df_profile, df_line.drop(columns=['E','N']), on='Distance', how='left').sort_values('Distance')

def dem_profiler_snowdepth(dem_dict,df,sample_distance,order=1,footprint=False,fillhole=True,stats='mean') -> pd.DataFrame:

    '''
    get a profile over DEMs by interpolate a dataframe that contains ICESat-2 ground tracks.

    :param dem_dict: a dict = {'name_key':xDEM.DEM,'name_key':xDEM.DEM}
    :param df: DataFrame with N and E
    
    :param sample_distance: int = 1.
    
    :param order: algorithem of interpolation 0 nearest, 1 linear, 2 cubic, works if no footprint...
    :param footprint_size: int, pts locate at the middle of footprint. Use it if you want downsampling DEM.
    :parameter stats: string, 'mean','median','min','std','sum','nodata'...works when there is footprint
    :fillhole: bool = Ture by default. If DataFrame has been processed by load_gdf set it False to speed up.
    return a datafram after interpolation and dem_h profiling.
    '''
    if not isinstance(df,GeoDataFrame):
        df['geometry'] = df.apply(lambda row: Point(row['E'], row['N']), axis=1)
        # Create a GeoDataFrame from the DataFrame
        df = gpd.GeoDataFrame(df, geometry='geometry')

    # generate a line. Decending from West to East.
    df_line = df.reset_index().sort_values('E',ascending=False)

    line = LineString(df_line.geometry.values)

    # the distance from ICESat-2 points
    df_line['Distance'] = [line.project(Point(point)) for point in line.coords]

    # interpolate the line, and also append the points from ICESat-2
    distances = np.append(np.arange(0, line.length, sample_distance), df_line['Distance'].values)
    sample_points = [line.interpolate(d) for d in distances]

    # put the results into a dataframe
    df_profile = pd.DataFrame({'Distance':distances,
                               'E':[i.coords[0][0] for i in sample_points],
                               'N':[i.coords[0][1] for i in sample_points]})

    # do some calculation for DEM height profile
    if footprint:
        f_width = footprint[0]
        f_length = footprint[1]
        # poly_list = [point.buffer(footprint_size/2) for point in sample_points]
        poly_list = [LineString([line.interpolate(d-f_length/2),line.interpolate(d+f_length/2)]).buffer(f_width/2) for d in distances]

        # interp_points
    x_coords, y_coords = (df_profile['E'].values,df_profile['N'].values)
    pts = np.array((x_coords, y_coords)).T

    for name, dem in dem_dict.items():
        if footprint: # Could be other condition e.g. acoording to the name
            # priority to extract value from DEM with footprint
            pts_stats = poly_zonal_stats(poly_list,dem,stats=stats)
            df_profile[f'dem_h_{name}'] = pts_stats[stats].values
        else:
            # else do interp_points
            df_profile[f'dem_h_{name}'] = interp_points(dem,pts,order=order,fillhole=fillhole,mode='constant',cval=np.nan)

    return pd.merge(df_profile, df_line.drop(columns=['E','N']), on='Distance', how='left').sort_values('Distance')


def dem_difference(dem_ref:DEM,dem_obs:DEM, 
                   coreg=False, shift_px=False, resampling='bilinear',samples:int = 5000,bounds=(-5,5),shift_median=False):
    '''
    return ddem = dem_obs - dem_ref. 

    :param dem_ref: xdem.DEM after vertical reference correction.
    :param dem_obs: xdem.DEM after vertical reference correction. To be shift.

    :param coreg: = False by defualt, or 'NuthKaab','GradientDescending'
    :param resampling: = 'bilinear'by defualt, or 'nearest','bicubic'...
    :param samples: = 5000 by default. Too low may results unstable GradientDescending Coreg.

    How to use it, e.g:

    ddem = dem_defference(dem_ref,dem_obs)
    '''

    # project dem_ref to dem_obs if needs.
    if dem_ref.res != dem_obs.res:
        dem_ref = dem_ref.reproject(dem_obs,resampling=resampling)

    # crop def_ref into dem_obs 
    if not coreg:
        if not shift_px:
            return dem_obs - dem_ref
        else:
            aligned_dem = dem_shift(dem_obs,shift_px[0],shift_px[1],0)
            return aligned_dem - dem_ref
    elif coreg == 'NuthKaab':
        func = xdem.coreg.NuthKaab()
        func.fit(dem_ref,dem_obs)
        print(f'Coreg maxtrix east_px, north_px, bias:{func._meta["offset_east_px"]:.4f},{func._meta["offset_north_px"]:.4f},{func._meta["bias"]:.3f}')
        aligned_dem = func.apply(dem_obs)
    elif coreg == 'GradientDescending':

        df = df_sampling_from_dem(dem_ref,samples=samples)
        # get results from best_shift_px
        res = best_shift_px(dem_obs,df,x0=(0,0),footprint=False,bounds=bounds,z_name='z',disp=False,stat='mix',downsampling=False,geosegment=False)
        
        shift_median = res[2] if shift_median is not False else 0
        aligned_dem = dem_shift(dem_obs,res[0],res[1],shift_median)
    else:
        print('coreg needs to be \'NuthKaab\',\'GradientDescending\' or False')
    return  aligned_dem - dem_ref

def dem_difference_plot(dem_ref:DEM,dem_obs:DEM,grid:bool = None or DEM,
                        range=(-5,5),vmax=2,vmin=-2,bounds=(-5,5),std_t=3,
                        resampling='bilinear',coreg='GradientDescending',samples=5000,shift_median=False):
    '''
    plot the difference map before and after NuthKaab coreg between dem_ref and dem_obs. Be in mind:

    - Always project dem_ref to dem_obs.
    - ddem = dem_obs - dem_ref

    :param dem_ref: xdem.DEM after vertical reference correction.
    :param dem_obs: xdem.DEM. Needs to be shift if providing grid.
    :param grid: xdem.DEM or Bool = None

    :param coreg: = False by defualt, or 'NuthKaab','GradientDescending'
    :param resampling: = 'bilinear'by defualt, or 'nearest','bicubic'...
    :param samples: = 5000 by default. Too low may results unstable GradientDescending Coreg.

    # resampling method from rasterio: nearest by default, bilinear, cubic, cubic_soline,lanczos
    # https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling

    How to use it, e.g:

    ddem_1_arctic = dem_difference_plot(dtm_1_ref,dtm_arctic,grid=None,range=(-5,5),resampling='cubic')

    return ddem_raw,ddem_coreg
    '''

    # datum shift
    if grid:
        dem_obs = dem_obs + grid.reproject(dem_obs,resampling='bilinear')

    # reproject DTM1 into low resolution

    if dem_ref.res != dem_obs.res and dem_ref.bounds != dem_obs.bounds: 
        dem_ref = dem_ref.reproject(dem_obs,resampling=resampling)

    # coreg
    ddem_coreg = dem_difference(dem_ref,dem_obs, coreg=coreg,samples=samples,shift_median=shift_median,bounds=bounds)
    
    # no-coreg
    ddem_raw = dem_obs - dem_ref
    
    # plot 
    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(18, 6))
    
    ddem_raw.show(ax=ax1, vmax=vmax,vmin=vmin)
    ddem_coreg.show(ax=ax2, vmax=vmax,vmin=vmin)

    ax1.set_title('dH (raw)')
    ax2.set_title('dH (co-registration)')
    final_histogram(ddem_raw,ddem_coreg,std_t=std_t,range=range,ax=ax3)

    return ddem_raw,ddem_coreg

def produc_to_csv(fid,sc_gdf_subset,sf_gdf_subset,grid,folder,dst_res=None,
                  order=1,range=(-10,10),attribute=None,footprint=None,weight=None,shift_median=True,downsampling=8000, geosegment=True,
                  min_points=1500,suffix='_dem',coreg='GradientDescending',shift_dem=False):
    # sourcery skip: extract-method
    '''
    Production function for parallel processing over dems, append results (sf,sc,_sum) into the folder.

    Need to use with partial function. How to use it parallelly please check workflows-production-coregistration.

    :param fid: the directory of the DEM to be processed.
    :param sc: Dataframe of ICESat-2 snow cover measurements.
    :param sf: Dataframe of ICESat-2 snow free measurements.
    :param folder: the directory of exporting the results.

    :param order: 1-using linear interpolation, 2- using cubic interpolation.
    :param range: the range of histgram output.
    :param attribute: the list of attributes to be produced and recorded during processing.
    :param footprint: the footprint used in zonal statistics when it needs (high resolution DEM).
    :param min_points: if the number of the measurements is less than min_points, the DEM is not going to be processed.
    :param suffix: to distinguish the elevation defference
    :param geosegment: True by default. Extent ICESat-08 geosegment when the samples are less than downsamplings(6000).

    :return retrun_info: gives the information: done or fail (with reasons).

    example:

    from functools import partial
    from p_tqdm import p_uimap
    from xsnow.godh import yield_gdf,sc_sf_subset_dem
    folder = r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\output_xdem_pts'
    
    # Warening ! Clean the folder or creat a new folder before running ! 
    # Oterwise it is going to append the result to old results by default !

    # three iteratable objects have the same length
    dem_list = dem_list_[75:]
    sc_list = yield_gdf(sc,dem_list,z_name='h_te_best_fit')
    sf_list = yield_gdf(sf,dem_list,z_name='h_te_best_fit')

    # start production
    production_func = partial(sc_sf_subset_dem,
                              grid='no_kv_HREF2018B_NN2000_EUREF89.tif',folder=folder,
                              order=1,range=(-10,10),attribute=['slope','aspect','planform_curvature', 'profile_curvature','curvature','topographic_position_index'],
                              footprint=None,min_points=900,suffix='_dtm10',coreg='GradientDescending',shift_dem=False)

    for result in p_uimap(production_func, dem_list,sc_list,sf_list,**{"num_cpus": 10}):
        print(result)
    '''

    import os
    import xdem
    import pandas as pd
    import gc
    from xsnow.godh import get_dh_dem,get_value_point

    # todo list
    name,extension = os.path.splitext(os.path.split(fid)[1])

    # done list from sum
    done_name_list = pd.read_csv(folder + '\\sum_.csv')['fid'].values if os.path.exists(folder + '\\sum_.csv') else [None]
    msg = {}

    # exclude done list from todo list
    if name not in done_name_list:
        dem = xdem.DEM(fid)

        if grid:
            try:
                dem = dem + xdem.DEM(grid).reproject(dem,resampling='bilinear') 
            except ValueError as err:
                msg['flag'] = f'Error info for {name}:{err}'

        # turn into 32633
        if dem.crs not in [pyproj.CRS(32633),pyproj.CRS(25833)]:
            dem = dem.reproject(dst_crs=pyproj.CRS(32633),dst_res=dst_res)

        if len(sf_gdf_subset) > min_points and len(sc_gdf_subset) > min_points/10 and 'flag' not in msg:

            try:# snow free coreg results, attribute = None if no need. 'z' is used to coregistration
                results = get_dh_dem(dem,sf_gdf_subset,
                                    pp=name,verbose=False,order=order,range=range,weight=weight,geosegment=geosegment,downsampling=downsampling,shift_median=shift_median,
                                    mask_highcurv=False,attribute=attribute,coreg=coreg,shift_dem=shift_dem)

                # shift median or not
                shift_median = results['shift_matrix'][2] if shift_median is not False else 0
                # snow cover dataframe. If do not need topograph attribute, set attribute = None to save time!
                gdf_shifted = gdf_shift(sc_gdf_subset,results['shift_matrix'][0]*dem.res[0], results['shift_matrix'][1]*dem.res[0], shift_median,z_name='z')
                pts = get_value_point(dem,gdf_shifted,attribute=attribute,order=order,footprint=footprint,z_name='z')
                
                sc_gdf_subset['snowdepth'+suffix] = pts['dh']
                sc_gdf_subset['coreg_bias'+suffix] = results['shift_matrix'][2]
                sc_gdf_subset['fid'+suffix] = name
                sc_gdf_subset['coreg_offset_east_px'+suffix] = results['shift_matrix'][0]
                sc_gdf_subset['coreg_offset_north_px'+suffix] = results['shift_matrix'][1]

                # rename snowfreee dataframe
                results['gdf']['fid'+suffix] = name
                results['gdf'] = results['gdf'].rename(columns={'dh_before': 'dh_before'+suffix,
                                                                'dh_after': 'dh_after'+suffix,
                                                                'coreg_bias':'coreg_bias'+suffix});
                if attribute is not None:
                    with contextlib.suppress(KeyError):
                        sc_gdf_subset['slope']= pts['slope']
                        sc_gdf_subset['aspect']= pts['aspect']
                        sc_gdf_subset['planc']= pts['planform_curvature']
                        sc_gdf_subset['profc'] = pts['profile_curvature']
                        sc_gdf_subset['curvature'] = pts['curvature']
                        sc_gdf_subset['tpi'] = pts['topographic_position_index']
                        results['gdf'] = results['gdf'].rename(columns={'planform_curvature': 'planc',
                                                                        'profile_curvature': 'profc',                                  
                                                                         'topographic_position_index':'tpi'});
                # append sum into csv
                results['sum']['N_snow'] = len(sc_gdf_subset)
                pd.DataFrame({name:results['sum']}).T.to_csv(folder+'\\sum_.csv', encoding='utf-8',mode='a',header=(not os.path.exists(folder+'\\sum_.csv')),index=False)
                
                msg['flag'] = 'Done'
                msg['snowcover'] = sc_gdf_subset
                msg['snowfree'] = results['gdf']
                
                del pts
                del sc_gdf_subset
                del results

            except (OSError, ValueError, KeyError, NotImplementedError, NameError) as err:
                msg['flag'] = f'Error info for {name}:{err}'
        else:
            msg['flag'] = f'Ignoring {name}: Snow_free={len(sf_gdf_subset)},snow_on={len(sc_gdf_subset)}'
    
        del dem
    else:
        msg['flag'] = f'Ignoring {name}: already in _sum'
    # Free memory
    gc.collect()
    # multiprocessing requires return for every processing. 
    return msg

def produc_tpi(fid, sc_gdf_subset, sf_gdf_subset, grid, folder, dst_res=None, order=1):

    '''
    Production function for parallel processing over dems, append results (sf,sc,_sum) into the folder:
        1. get shift px from sum
        2. apply shift to df
        3. calculate tpi by xdem
        4. interpolate values and return it.

    Need to use with partial function. How to use it parallelly please check workflows-production-coregistration.

    :param fid: the directory of the DEM to be processed.
    :param sc: Dataframe of ICESat-2 snow cover measurements.
    :param sf: Dataframe of ICESat-2 snow free measurements.
    :param folder: the directory of exporting the results.
    :param order: 1-using linear interpolation, 2- using cubic interpolation.

    :return retrun_info: gives the information: done or fail (with reasons).

    '''

    import os
    import xdem
    import pandas as pd
    import gc

    # todo list
    name,extension = os.path.splitext(os.path.split(fid)[1])

    # done list from sum
    sum_list = pd.read_csv(folder + '\\sum_.csv') if os.path.exists(folder + '\\sum_.csv') else [None]
    msg = {}

    # todo list in done list
    if name in sum_list.fid.values:
        dem = xdem.DEM(fid)

        if grid:
            try:
                dem = dem + xdem.DEM(grid).reproject(dem,resampling='bilinear') 
            except ValueError as err:
                msg['flag'] = f'Error info for {name}:{err}'

        # turn into 32633
        if dem.crs not in [pyproj.CRS(32633),pyproj.CRS(25833)]:
            dem = dem.reproject(dst_crs=pyproj.CRS(32633),dst_res=dst_res)

        if 'flag' not in msg:

            try:# snow free coreg results, attribute = None if no need. 'z' is used to coregistration
                # get shift matrix from sum files.
                e_px = sum_list[sum_list.fid.isin([name])].offset_east_px.values
                n_px = sum_list[sum_list.fid.isin([name])].offset_north_px.values
                
                # apply shift to DEM or gdf, and get dh
                sc_shifted = gdf_shift(sc_gdf_subset,e_px*dem.res[0], n_px*dem.res[0], 0)
                sf_shifted = gdf_shift(sf_gdf_subset,e_px*dem.res[0], n_px*dem.res[0], 0)

                # tpi
                tpi_9 = xdem.terrain.get_terrain_attribute(dem, attribute=['topographic_position_index'], window_size=9)
                tpi_27 = xdem.terrain.get_terrain_attribute(dem, attribute=['topographic_position_index'], window_size=27)

                # interpolate to get value
                for gdf,df in zip([sc_shifted,sf_shifted],[sc_gdf_subset,sf_gdf_subset]):
                    x_coords, y_coords = (gdf['E'].values,gdf['N'].values)
                    pts = np.array((x_coords, y_coords)).T

                    attr_value = [interp_points(i,pts,input_latlon=False, order=order) for i in [tpi_9,tpi_27]]
                    # pass the results to return_dict
                    for i,j in zip(['tpi_9','tpi_27'],attr_value):
                        df[i] = j

                # return
                msg['flag'] = 'Done'
                msg['snowcover'] = sc_gdf_subset
                msg['snowfree'] = sf_gdf_subset
                
                del pts
                del sc_gdf_subset
                del sf_gdf_subset

            except (OSError, ValueError, KeyError, NotImplementedError, NameError) as err:
                msg['flag'] = f'Error info for {name}:{err}'
        else:
            msg['flag'] = f'Ignoring {name}: ValueError when applying grid'
    
        del dem
    else:
        msg['flag'] = f'Ignoring {name}: Not in _sum'
    # Free memory
    gc.collect()
    # multiprocessing requires return for every processing. 
    return msg

def produc_to_csv_sc(fid,sc_gdf_subset,grid,folder,
                     order=1,attribute=None,footprint=None,
                     min_points=1500,suffix='_dem',z_name='h_te_best_fit',shift_dem=False):
    '''
    Update snow depth results acording to privious co-registration func.

    This is a function for parallel processing over dems, append snow depth results into the folder.

    Need to use with partial function. How to use it parallelly please check workflows-production-coregistration.

    :param fid: the directory of the DEM to be processed.
    :param sc: Dataframe of ICESat-2 snow cover measurements.
    :param folder: the directory of exporting the results where '_sum' exist!

    :param order: 1-using linear interpolation, 2- using cubic interpolation.
    :param range: the range of histgram output.
    :param attribute: the list of attributes to be produced and recorded during processing.
    :param footprint: the footprint used in zonal statistics when it needs (high resolution DEM).
    :param min_points: if the number of the measurements is less than min_points, the DEM is not going to be processed.
    :param suffix: to distinguish the elevation defference

    :return retrun_info: gives the information: done or fail (with reasons).
    '''

    import os
    import xdem
    import gc
    import pandas as pd
    from xsnow.godh import get_value_point

    # todo list
    name_fid,extension = os.path.splitext(os.path.split(fid)[1])

    # done list from sum
    sum_list = pd.read_csv(folder + '\\sum_.csv') if os.path.exists(folder + '\\sum_.csv') else [None]
    msg = {}

    # todo list in done list
    if name_fid in sum_list['fid'].values:
        dem = xdem.DEM(fid)
        
        if grid:
            dem = dem + xdem.DEM(grid).reproject(dem,resampling='bilinear') 

        # Snow free/snow cover measurements of entire Norway from ICESat-2

        if len(sc_gdf_subset) > min_points/10:

            try:
                # get shift matrix from sum files.
                e_px = sum_list[sum_list.fid.isin([name_fid])].offset_east_px.values
                n_px = sum_list[sum_list.fid.isin([name_fid])].offset_north_px.values
                bias = sum_list[sum_list.fid.isin([name_fid])].bias.values
                
                # apply shift to DEM or gdf, and get dh
                if shift_dem:
                    aligned_dem = dem_shift(dem,e_px, n_px, bias)
                    pts = get_value_point(aligned_dem,sc_gdf_subset,attribute=attribute,order=order,footprint=footprint,z_name=z_name)
                else:
                    gdf_shifted = gdf_shift(sc_gdf_subset,e_px*dem.res[0], n_px*dem.res[0], bias,z_name=z_name)
                    pts = get_value_point(dem,gdf_shifted,attribute=attribute,order=order,footprint=footprint,z_name=z_name)

                sc_gdf_subset['snowdepth'+suffix] = pts['dh']
                sc_gdf_subset['fid'+suffix] = name_fid
                sc_gdf_subset['coreg_offset_east_px'+suffix] = e_px
                sc_gdf_subset['coreg_offset_north_px'+suffix] = n_px
                sc_gdf_subset['coreg_bias'+suffix] = bias

                # rename snowfree dataframe
                if attribute is not None:
                    with contextlib.suppress(KeyError):
                        sc_gdf_subset['slope']= pts['slope']
                        sc_gdf_subset['aspect']= pts['aspect']
                        sc_gdf_subset['planc']= pts['planform_curvature']
                        sc_gdf_subset['profc'] = pts['profile_curvature']
                        sc_gdf_subset['curvature'] = pts['curvature']
                        sc_gdf_subset['tpi'] = pts['topographic_position_index']

                # append results into csv
                fname_sc = folder+'\\snow_cover_all_gdf'+suffix+'.csv'
                sc_gdf_subset.to_csv(fname_sc, sep='\t', encoding='utf-8',mode='a', header=(not os.path.exists(fname_sc)),index=False)  # index=False
                

                msg['flag'] = 'Done'
                msg['snowcover'] = sc_gdf_subset
                #return results['gdf'],sc_gdf_subset

            except (OSError, ValueError, KeyError, NotImplementedError, NameError) as err:
                msg['flag'] = f'Error info for {name_fid}:{err}'
        else:
            msg['flag'] = f'Ignoring {name_fid}: ,snow_on={len(sc_gdf_subset)}'
        
        del dem
    # multiprocessing requires return for every processing task. 
    else:
        msg['flag'] = f'Ignoring {name_fid}: already in _sum'
    gc.collect()
    return msg
