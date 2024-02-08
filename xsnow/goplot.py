'''
Plot and statistics funcition
'''

from cProfile import label
from cmath import tan
import zipapp
import geoutils as gu
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from geoutils.spatial_tools import subsample_raster, get_array_and_mask
import numpy as np
import xdem
from xdem.dem import DEM
from geopandas.geodataframe import GeoDataFrame
from typing import IO, Any
import holoviews as hv, datashader as ds, geoviews as gv, geoviews.tile_sources as gvts
from holoviews.operation.datashader import rasterize
hv.extension('bokeh')
from bokeh.models import Title
from holoviews import opts
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def RMSE(indata, perc_t=None,d_lim=100):
    """
    Calculate the root mean square error (RMSE) of an input array.

    :param indata: Input array to calculate RMSE of.
    :type indata: array-like

    :param perc_t: Optional percentile threshold for outlier removal.
    :type perc_t: float or None

    :return: RMSE of input array.
    :rtype: float
    """
    # Check input data type and convert to numpy array if necessary
    if not isinstance(indata, np.ndarray):
        indata = np.asarray(indata)

    # Check input data shape
    if indata.size == 0:
        raise ValueError("Input array is empty.")

    if perc_t is not None:
        # Check percentile threshold value
        if not isinstance(perc_t, (float, int)) or perc_t < 0 or perc_t > 100:
            raise ValueError("Invalid percentile threshold value.")

        # Remove outliers based on percentile threshold
        lower_percentile = np.nanpercentile(indata, 100-perc_t)
        upper_percentile = np.nanpercentile(indata, perc_t)
        valids = np.where((indata > lower_percentile) & (indata < upper_percentile) & (np.abs(indata) < d_lim))
        indata = indata[valids]

    # Calculate RMSE and return
    return np.sqrt(np.nanmean(indata ** 2))

def threshold(df, perc_t=99.75, dlim=100, std_t=None, window=(-10,10)):
    if window is None:
        if perc_t or dlim:
        # Remove outliers
            lower_percentile = np.nanpercentile(df, 100-perc_t)
            upper_percentile = np.nanpercentile(df, perc_t)
            valids = ((df > lower_percentile) & (df < upper_percentile) & (np.abs(df) < dlim))
            df= df[valids]
        if std_t:
            df = np.squeeze(np.asarray(df[np.logical_and.reduce((np.isfinite(df), (np.abs(df < np.nanstd(df) * std_t))))]))
    else:
        valids = ((df > window[0]) & (df < window[1]) & (np.abs(df) < dlim))
        df= df[valids]
    return df


def final_histogram(
    dH0, dHfinal, 
    pp=None,nmad=True,
    d_lim=100,std_t=None,perc_t=99.5, window=None,
    range=(-20, 20),legend=['Original', 'Coregistered'],
    offset=None,
    ax=None,
    dH_ref=None,
    bins=100,
    title='Elevation difference histograms',
    density=False,
    quantile=False):

    '''
    :data
    :param dH0: np.array
    :param dHfinal: np.array
    :param dH_ref: np.array

    :statistics part:
    :param d_lim: (1) d_lim = 100, filter dh when it > d_lim
    :param perc_t: (2) perc_t = 99 by default, filter dh locates out of 1%-99% after (1). set perc_t = 100 to disable this function.
    :param std_t: (3) std = None by default, filter dh when it > 3 std after (1) and (2). Only suggest one of (2) or (3).
    :param range: the xlim of the axis of the histgram
    :param bins: the bins of the axis.
    :param density: False by default. If True, the result is the value of the probability density function at the bin, normalized such that the integral over the range is 1. 
     
    :others:
    :param pp: bool. Ture = export results as a png.
    :param title: string.
    :param ax: matplotlib ax. default is None
    :param nmad: bool.  Ture = using NMAD.
    '''
    
    if isinstance(dH0, np.ma.masked_array):
        dH0 = get_array_and_mask(dH0, check_shape=False)[0]
    else:
        dH0 = np.asarray(dH0)
    if isinstance(dHfinal, np.ma.masked_array):
        dHfinal = get_array_and_mask(dHfinal, check_shape=False)[0]
    else:
        dHfinal = np.asarray(dHfinal)

    if ax is None:
        fig,ax = plt.subplots(figsize=(7, 5), dpi=200)
    ax.set_title(title, fontsize=10)


    dH0 = dH0[np.isfinite(dH0)]
    dHfinal = dHfinal[np.isfinite(dHfinal)]
    number_raw = len(dH0)

    nmad = [xdem.spatialstats.nmad(dH0),xdem.spatialstats.nmad(dHfinal)]

    # using 3 std by default as a threshold
    if window:
        lower_percentile =  window[0]
        upper_percentile = window[1]
        valids = np.where((dH0 > lower_percentile) & (dH0 < upper_percentile))
        dH0 = dH0[valids]
        
        valids = np.where((dHfinal > lower_percentile) & (dHfinal < upper_percentile))
        dHfinal = dHfinal[valids]
    else:
        if perc_t or d_lim:
            # Remove outliers
            lower_percentile = min(np.nanpercentile(dH0, 100-perc_t),np.nanpercentile(dHfinal, 100-perc_t))
            upper_percentile = max(np.nanpercentile(dH0, perc_t),np.nanpercentile(dHfinal, perc_t))
            valids = np.where((dH0 > lower_percentile) & (dH0 < upper_percentile) & (np.abs(dH0) < d_lim))
            dH0 = dH0[valids]
            
            valids = np.where((dHfinal > lower_percentile) & (dHfinal < upper_percentile) & (np.abs(dHfinal) < d_lim))
            dHfinal = dHfinal[valids]
        if std_t:
            dH0 = np.squeeze(np.asarray(dH0[np.logical_and.reduce((np.isfinite(dH0), (np.abs(dH0) < np.nanstd(dH0) * std_t)))]))
            dHfinal = np.squeeze(np.asarray(dHfinal[np.logical_and.reduce((np.isfinite(dHfinal),(np.abs(dHfinal) < np.nanstd(dHfinal) * std_t)))]))


    j1, j2 = np.histogram(dH0, bins=bins, range=range,density=density)
    k1, k2 = np.histogram(dHfinal, bins=bins, range=range,density=density)

    stats0 = [np.mean(dH0), np.median(dH0), 
              np.std(dH0), RMSE(dH0), 
              np.sum(np.isfinite(dH0)),nmad[0]
              ]
    stats_fin = [np.mean(dHfinal), np.median(dHfinal), 
                 np.std(dHfinal), RMSE(dHfinal), 
                 np.sum(np.isfinite(dHfinal)),nmad[1]]
    
    ax.plot(j2[1:], j1, 'k-', linewidth=1.2,alpha=0.7) # legend = 'Original'
    ax.plot(k2[1:], k1, 'r-', linewidth=1.2,alpha=0.7) # legend = 'Coregistered'

    if dH_ref is not None:
        dH_ref = np.asarray(dH_ref)
        valids = np.where((dH_ref > lower_percentile) & (dH_ref < upper_percentile) & (np.abs(dH_ref) < d_lim))
        dH_ref = dH_ref[valids]
        m1, m2 = np.histogram(dH_ref, bins=bins, range=range,density=density)
        ax.plot(m2[1:], m1, 'b--', linewidth=1,alpha=0.5)
        stats_ref = [np.mean(dH_ref), np.median(dH_ref),np.std(dH_ref), RMSE(dH_ref),np.sum(np.isfinite(dH_ref))]
        ax.text(0.70, 0.50, 'Mean: ' + ('{:.2f} m'.format(stats_ref[0])),
             fontsize=9, fontweight='bold', color='blue', family='monospace', transform=ax.transAxes)
        ax.text(0.70, 0.45, 'Median: ' + ('{:.2f} m'.format(stats_ref[1])),
             fontsize=9, fontweight='bold', color='blue', family='monospace', transform=ax.transAxes)
        ax.text(0.70, 0.40, 'Std dev.: ' + ('{:.2f} m'.format(stats_ref[2])),
             fontsize=9, fontweight='bold', color='blue', family='monospace', transform=ax.transAxes)
        ax.text(0.70, 0.35, 'RMSE: ' + ('{:.2f} m'.format(stats_ref[2])),
             fontsize=9, fontweight='bold', color='blue', family='monospace', transform=ax.transAxes)
        if nmad:
            ax.text(0.70, 0.30, 'NMAD: ' + ('{:.2f} m'.format(xdem.spatialstats.nmad(dH_ref))),
                fontsize=9, fontweight='bold', color='blue', family='monospace', transform=ax.transAxes)
    
    ax.legend(legend)
    ax.set_xlabel('Elevation difference [m]')
    ax.set_xlim(range)

    if quantile:
        quantile_0 = np.quantile(dH0, [0.25, 0.75], axis=0)
        quantile_f = np.quantile(dHfinal, [0.25, 0.75], axis=0)

        for quantile in quantile_0:
            ax.axvline(quantile, color='k', linestyle='-',alpha=0.25)
        for quantile in quantile_f:
            ax.axvline(quantile, color='r', linestyle='-',alpha=0.25)


    if density:
        ax.set_ylabel('Density')
    else:
        ax.set_ylabel('Number of samples')

    # numwidth = max([len('{:.2f} m'.format(xadj)), len('{:.2f} m'.format(yadj)), len('{:.2f} m'.format(zadj))])
    ax.text(0.05, 0.80, 'Mean: ' + ('{:.2f} m'.format(stats0[0])),
             fontsize=9, fontweight='bold', color='black', family='monospace', transform=ax.transAxes)
    ax.text(0.05, 0.75, 'Median: ' + ('{:.2f} m'.format(stats0[1])),
             fontsize=9, fontweight='bold', color='black', family='monospace', transform=ax.transAxes)
    ax.text(0.05, 0.70, 'Std dev.: ' + ('{:.2f} m'.format(stats0[2])),
             fontsize=9, fontweight='bold', color='black', family='monospace', transform=ax.transAxes)
    ax.text(0.05, 0.65, 'RMSE: ' + ('{:.2f} m'.format(stats0[3])),
             fontsize=9, fontweight='bold', color='black', family='monospace', transform=ax.transAxes)
    if nmad:
        ax.text(0.05, 0.60, 'NMAD: ' + ('{:.2f} m'.format(nmad[0])),
             fontsize=9, fontweight='bold', color='black', family='monospace', transform=ax.transAxes)
        ax.text(0.05, 0.30, 'NMAD: ' + ('{:.2f} m'.format(nmad[1])),
             fontsize=9, fontweight='bold', color='red', family='monospace', transform=ax.transAxes)
    
    ax.text(0.05, 0.50, 'Mean: ' + ('{:.2f} m'.format(stats_fin[0])),
             fontsize=9, fontweight='bold', color='red', family='monospace', transform=ax.transAxes)
    ax.text(0.05, 0.45, 'Median: ' + ('{:.2f} m'.format(stats_fin[1])),
             fontsize=9, fontweight='bold', color='red', family='monospace', transform=ax.transAxes)
    ax.text(0.05, 0.40, 'Std dev.: ' + ('{:.2f} m'.format(stats_fin[2])),
             fontsize=9, fontweight='bold', color='red', family='monospace', transform=ax.transAxes)
    ax.text(0.05, 0.35, 'RMSE: ' + ('{:.2f} m'.format(stats_fin[3])),
             fontsize=9, fontweight='bold', color='red', family='monospace', transform=ax.transAxes)
    ax.text(0.05, 0.20, 'n: ' + ('{}'.format(int(stats0[4]))),
             fontsize=8, fontweight='normal', color='black', family='monospace', transform=ax.transAxes)
    ax.text(0.05, 0.15, 'n: ' + ('{}'.format(int(stats_fin[4]))),
             fontsize=8, fontweight='normal', color='red', family='monospace', transform=ax.transAxes)    
    #ax.text(0.05, 0.10, 'N: ' + ('{}'.format(number_raw)),
    #         fontsize=8, fontweight='normal', color='black', family='monospace', transform=ax.transAxes)                

    if offset is not None:
        ax.text(0.80, 0.15, 'Offset px E,N: \n' + ('{:.3f},{:.3f}'.format(offset[0],offset[1])),
             fontsize=7, fontweight='light', color='black', family='monospace', transform=ax.transAxes)  

    if pp is not None:
        fig.savefig(pp, bbox_inches='tight', dpi=300)
    return stats0,stats_fin

def normal_statistics(dH0: np.ndarray,
                      perc_t = 99.875,
                      d_lim = 100,
                      std_t = None,
                     ) -> float:
    """
    :param data: input data
    :param nfact: normalization factor for the data

    :returns nmad: (normalized) median absolute deviation of data.
    """
    if isinstance(dH0, np.ma.masked_array):
        dH0 = get_array_and_mask(dH0, check_shape=False)[0]
    else:
        dH0 = np.asarray(dH0)

    nmad = xdem.spatialstats.nmad(dH0)

    if perc_t or d_lim:
        # Remove outliers
        lower_percentile = np.nanpercentile(dH0, 100-perc_t)
        upper_percentile = np.nanpercentile(dH0, perc_t)
        valids = np.where((dH0 > lower_percentile) & (dH0 < upper_percentile) & (np.abs(dH0) < d_lim))
        dH0 = dH0[valids]
    if std_t:
        dH0 = np.squeeze(np.asarray(dH0[np.logical_and.reduce((np.isfinite(dH0), (np.abs(dH0) < np.nanstd(dH0) * std_t)))]))

    #print('normaly distribution stats: mean,median,std,rmse:',stats_0)
    return [np.mean(dH0), np.median(dH0), np.std(dH0), RMSE(dH0), np.sum(np.isfinite(dH0)), nmad]

def plot_over_dem(gdf:GeoDataFrame or DEM,dem: DEM,attribute=['hillshade'],col='z',
                  cmap='inferno',cmap_b=None,vmin=-10,vmax=10,
                  figsize = (10,10),markersize=3,
                  **kwargs: Any):
    '''
    Plot DEM as background and plot gdf on it.
    '''

    [L,B,R,T] = dem.bounds
    
    z = []

    if cmap_b is None:
        cmap_b = [] 

    if attribute == False:
        z.append(dem)
        cmap_b.append('gist_earth')
    else:
        dem_attr = xdem.terrain.get_terrain_attribute(dem, attribute=attribute)
        z.extend([dem_attr] if dem_attr is not list else dem_attr)
        dict = {'slope':'Reds','hillshade':'Greys_r','aspect':'twilight','Curvature':'RdGy_r' }
        cmap_b.extend([dict[i] for i in attribute])

    for i,c in zip(z,cmap_b):
        fig,ax = plt.subplots(figsize = figsize)
        i.show(ax=ax,cmap=c,extent=(L,R,B,T),**kwargs)
        gdf.plot(ax=ax,column=col, 
                 cmap=cmap,vmin=vmin,vmax=vmax, markersize=markersize, 
                 legend=True,legend_kwds={'label': "dh [m]"})

def gdf_over_map(sf_cop,v):
    opts_ = dict(width=500, height=500, tools=['hover'], 
            colorbar=True,symmetric=True,clim=(-5,5),
            cmap='Spectral')
    tiles = gvts.OSM.options(alpha=0.6)

    points = gv.Points(sf_cop, kdims=['longitude','latitude'],vdims=[v])
    map = rasterize(points, x_sampling=0.01, y_sampling=0.01,aggregator='mean').options(**opts_,title=v)* tiles
    return map.opts(width = 500,height=500)

# binning analysis
import hvplot.pandas 

def bin_analysis(sf_cop,dems = ['dh_after_dtm10','dh_after_cop30','dh_after_dtm1']):
    # prepared data
    #sf_cop.loc[:,'maxc_arr'] = np.maximum(np.abs(sf_cop['planc']), np.abs(sf_cop['profc']))

    ## data prepare - binning by elevation, landcover, slope, curvature
    sf_cop.loc[:,'slope_bin'] = sf_cop['slope'].apply(lambda x: np.floor(x) if np.floor(x) %2 ==0 else np.floor(x)-1)
    
    #sf_cop.loc[:,'curvature_bin'] = sf_cop['curvature'].apply(lambda x: np.floor(x))
    sf_cop.loc[:,'profc_bin'] = sf_cop['profc'].apply(lambda x: np.floor(x))
    sf_cop.loc[:,'planc_bin'] = sf_cop['planc'].apply(lambda x: np.floor(x))
    sf_cop.loc[:,'elev_bin'] = sf_cop['h_te_best_fit'].apply(lambda x: np.floor(x/100))
    sf_cop.loc[:,'aspect_bin'] = sf_cop['aspect'].apply(lambda x: np.floor(x/10))
    sf_cop.loc[:,'tpi_bin'] = sf_cop['tpi'].apply(lambda x: np.floor(x*10))
    sf_cop.loc[:,'tpi_9_bin'] = sf_cop['tpi_9'].apply(lambda x: np.floor(x))
    sf_cop.loc[:,'tpi_27_bin'] = sf_cop['tpi_27'].apply(lambda x: np.floor(x))
    sf_cop.loc[:,'h_mean_canopy_bin'] = sf_cop['h_mean_canopy'].apply(lambda x: np.floor(x))
    sf_cop.loc[:,'segment_cover_bin'] = sf_cop['segment_cover'].apply(lambda x: np.floor(x/5))
    sf_cop.loc[:,'canopy_openness_bin'] = sf_cop['canopy_openness'].apply(lambda x: np.floor(x*10))
    sf_cop.loc[:,'n_te_photons_bin'] = sf_cop['n_te_photons'].apply(lambda x: np.floor(x/100))

    ## QC: hardcut by 50m.
    sf_cop = sf_cop.query('abs(%s) < 50 & abs(%s) < 50 & abs(%s) < 50 & h_te_best_fit > 0' % (dems[0],dems[1],dems[2]))

    return sf_cop


def plot_binning(sf_cop,dem,color,ylim=(-25, 20)):

    ## filter and sort for an easier plot.
    violin_elev = hv.Violin(sf_cop.sort_values(by='elev_bin'),['elev_bin'], dem,label="Elevation").opts(xlabel='WGS84 Height [100 m]',ylabel='dH [m]')
    violin_aspect = hv.Violin(sf_cop.sort_values(by='aspect_bin'),['aspect_bin'], dem,label="Aspect").opts(xlabel='Azimuith [10 degree]',ylabel='dH [m]')
    violin_slope = hv.Violin(sf_cop[sf_cop['slope_bin']<45].sort_values(by='slope_bin'),['slope_bin'], dem,label="Slope").opts(xlabel='Slope [degree]',ylabel='dH [m]')
    violin_curvature_plan = hv.Violin(sf_cop[abs(sf_cop['planc_bin'])<14].sort_values(by='planc_bin'),['planc_bin'], dem,label="Plan Curvature").opts(xlabel='Plan Curvature [100 / m]',ylabel='dH [m]')
    violin_curvature_prof = hv.Violin(sf_cop[abs(sf_cop['profc_bin'])<14].sort_values(by='profc_bin'),['profc_bin'], dem,label="Profile Curvature").opts(xlabel='Profile Curvature [100 / m]',ylabel='dH [m]')

    #violin_curvature = hv.Violin(sf_cop[abs(sf_cop['curvature_bin'])<14].sort_values(by='curvature_bin'),['curvature_bin'], dem,label="Curvature").opts(xlabel='Curvature [100 / m]',ylabel='dH [m]')
    #violin_photons = hv.Violin(sf_cop[sf_cop['n_te_photons_bin']<23].sort_values(by='n_te_photons_bin'),['n_te_photons_bin'], dem,label="N_photons").opts(xlabel='N [100]',ylabel='dH [m]')
    violin_tpi = hv.Violin(sf_cop[abs(sf_cop['tpi_bin'])<15].sort_values(by='tpi_bin'),['tpi_bin'], dem,label="TPI (30 m)").opts(xlabel='TPI [0.1]',ylabel='dH [m]')
    #violin_tpi_9 = hv.Violin(sf_cop[abs(sf_cop['tpi_9_bin'])<15].sort_values(by='tpi_9_bin'),['tpi_9_bin'], dem,label="TPI (90 m)").opts(xlabel='TPI [1]',ylabel='dH [m]')
    #violin_tpi_27 = hv.Violin(sf_cop[abs(sf_cop['tpi_27_bin'])<12].sort_values(by='tpi_27_bin'),['tpi_27_bin'], dem,label="TPI (270 m)").opts(xlabel='TPI [1]',ylabel='dH [m]')

    violin_openness = hv.Violin(sf_cop[sf_cop['canopy_openness_bin']<30].sort_values(by='canopy_openness_bin'),['canopy_openness_bin'], dem,label="Canopy openness").opts(xlabel='Canopy Openness [0.1]',ylabel='dH [m]')
    violin_cover = hv.Violin(sf_cop.sort_values(by='segment_cover_bin'),['segment_cover_bin'], dem,label="Canopy cover").opts(xlabel='Canopy coverage [5%]',ylabel='dH [m]')
    violin_canopy = hv.Violin(sf_cop[sf_cop['h_mean_canopy_bin']<20].sort_values(by='h_mean_canopy_bin'),['h_mean_canopy_bin'], dem,label="Canopy height").opts(xlabel='Canopy height [m]',ylabel='dH [m]')

    layout = violin_elev + violin_aspect + violin_slope + violin_curvature_plan + violin_curvature_prof + violin_tpi + violin_openness + violin_cover + violin_canopy
    map = layout.opts(
        opts.Violin(height=300,violin_fill_color=color, width=410, ylim=ylim,violin_line_color='grey',violin_line_alpha=0.1,xrotation=30,fontscale=0.8)).cols(3)
    hv.save(map,dem,fmt='png',dpi=300)

    return map

def plot_counting(sf_cop,dems = ['dh_after_dtm10','dh_after_cop30','dh_int_dem']):
    ## filter and sort for an easier plot.
    df_slope = sf_cop[sf_cop['slope_bin']<45].sort_values(by='slope_bin')
    df_curvature = sf_cop[sf_cop['maxc_arr_bin']<16].sort_values(by='maxc_arr_bin')
    
    # counting the sampling, all and dh below 1m.
    def count_below_1(x,name):
        return len(x[np.abs(x[name]) < 1])
    def count_all(df,groupby):
        for name in dems:
            df['N_1m_'+name.split('_')[-1]] = groupby.apply(count_below_1,name).cumsum()
        df.reset_index(inplace=True)
    slope_groupby =  df_slope.groupby('slope_bin')
    curvature_groupby = df_curvature.groupby('maxc_arr_bin')
    landcover_groupby = sf_cop.sort_values(by='class').groupby('class')
    elev_groupby = sf_cop.groupby('elev_bin')
    df_slope_count = slope_groupby.aggregate({'beam':'count'}).rename(columns={'beam': 'count'}).cumsum()
    df_curvature_count = curvature_groupby.aggregate({'beam':'count'}).rename(columns={'beam': 'count'}).cumsum()
    df_landcover_count = landcover_groupby.aggregate({'beam':'count'}).rename(columns={'beam': 'count'}).cumsum()
    df_elev_count = elev_groupby.aggregate({'beam':'count'}).rename(columns={'beam': 'count'}).cumsum()
    count_all(df_slope_count,slope_groupby)
    count_all(df_curvature_count,curvature_groupby)
    count_all(df_landcover_count,landcover_groupby)
    count_all(df_elev_count,elev_groupby)

    count_elev = df_elev_count.hvplot.line(x='elev_bin',y=['count','N_1m_dtm10','N_1m_cop30','N_1m_dem'],
                                            legend=False,width=400,height=400,xlabel='elevation [100m]',ylabel='N of measurements')
    count_slope = df_slope_count.hvplot.line(x='slope_bin',y=['count','N_1m_dtm10','N_1m_cop30','N_1m_dem'],
                                                xlabel='Slope [degree]',attr_labels=False,legend=False, width=400,height=400)
    count_curvature = df_curvature_count.hvplot.line(x='maxc_arr_bin',y=['count','N_1m_dtm10','N_1m_cop30','N_1m_dem'],
                                                xlabel='Curvature',legend=False,width=400,height=400)
    count_landcover = df_landcover_count.hvplot.line(x='class',y=['count','N_1m_dtm10','N_1m_cop30','N_1m_dem'],
                                                        rot=45,legend=False,width=400,height=400)
    layout = count_elev + count_slope + count_curvature + count_landcover
    return layout.cols(4)

def plot_scheme(dh_0,dh_1,aspect,slope,figsize=(6,6),ylim=(-20,20)):
    '''
    Accoding to Nuth&Kaab, the relationship betwwen dh and slope,aspect
    could be plot in 2D scheme. Thus we can use it to check the improvement'
    of the coregistration.
    '''
    
    fig,ax = plt.subplots(figsize = figsize)
    ax.scatter(aspect,dh_0/np.tan(slope),c='cyan',marker='.',alpha=.8,label='original')
    ax.scatter(aspect,dh_1/np.tan(slope),c='black',marker='.',alpha=.4,label='coregistration')
    ax.legend()
    ax.set_xlim(0,360)
    ax.set_ylim(ylim)

    ax.set_xlabel('Terrain Aspect [degrees]')
    ax.set_ylabel('dh/tan(α) [m]')

def plot_by_datashader(sf):
    import datashader as ds, pandas as pd, colorcet
    cvs = ds.Canvas(plot_width=850, plot_height=500)
    agg = cvs.points(sf, 'longitude', 'latitude')
    img = ds.tf.shade(agg, cmap=colorcet.fire, how='log')
    return img

def sum_to_json(done_csv,crs='EPSG:32633'):

    '''
    input a dataframe, which contains 'geometry' columns under crs=32633:

    folder = r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\output_xdems_dem10_SRTM'
    done_csv = pd.read_csv(folder+'\\sum_.csv')

    output a json, which could be plot by plotly
    '''
    import geopandas as gpd
    import json
    gdf_2 = gpd.GeoDataFrame(done_csv, geometry=gpd.GeoSeries.from_wkt(done_csv['geometry']),crs=crs)
    gdf_2 = gdf_2.to_crs('EPSG:4326')
    # geodataframe into geojson
    return json.loads(gdf_2.to_json())

def dem_profiler_plot(df,fig,row,col,error_y_dict=None,lgroup=None,ref=None):

    '''
    Used to plot dataframe genarated from xsnow.godh.dem_profiler.
    
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1,shared_xaxes='all',shared_yaxes='all',vertical_spacing=0.03, subplot_titles=("a", 'b'))
    df_plot(df_profile_2011,fig, row=1, col=1,lgroup=None)
    df_plot(df_profile_2111,fig, row=2, col=1,lgroup=None)
    fig.update_layout(height=900, width=1800)
    fig.show()

    Change rows and cols accoding to subplots
    '''

    # color map
    colorMap = px.colors.qualitative.Bold

    if ref:
        ref = df[ref] 
    else:
        ref = 0
    # create a trace for col1
    trace1 = go.Scatter(x=df['Distance'], y=df['dem_h_dtm1']-ref, name='dtm1', mode='lines',legendgroup=lgroup,line_color=colorMap[1])
    trace2 = go.Scatter(x=df['Distance'], y=df['dem_h_dtm10']-ref, name='dtm10', mode='lines',legendgroup=lgroup,line_color=colorMap[2])
    trace3 = go.Scatter(x=df['Distance'], y=df['dem_h_lidar_n']-ref, name='lidar_n', mode='lines',legendgroup=lgroup,line_color=colorMap[3])
    trace4 = go.Scatter(x=df['Distance'], y=df['dem_h_lidar_s']-ref, name='lidar_s', mode='lines',legendgroup=lgroup,line_color=colorMap[4])
    trace5 = go.Scatter(x=df['Distance'], y=df['dem_h_dem_arctic']-ref, name='arctic', mode='lines',legendgroup=lgroup,line_color=colorMap[5])
    trace6 = go.Scatter(x=df['Distance'], y=df['dem_h_dem_cop30']-ref, name='cop30', mode='lines',legendgroup=lgroup,line_color=colorMap[6])
    trace7 = go.Scatter(x=df['Distance'], y=df['dem_h_dem_fab']-ref, name='fab', mode='lines',legendgroup=lgroup,line_color=colorMap[7])

    # create a scatter plot using plotly.express
    scatter1 = go.Scatter(x=df['Distance'], y=df['h_te_best_fit']-ref, marker=dict(color='black'),mode='markers',name='ATL08_best_fit')
    scatter2 = go.Scatter(x=df['Distance'], y=df['sd_correct_dtm1']+df['dem_h_dtm1']-ref, marker=dict(color=colorMap[1]),mode='markers', name='sd_correct_dtm1')
    scatter3 = go.Scatter(x=df['Distance'], y=df['sd_correct_dtm10']+df['dem_h_dtm1']-ref, marker=dict(color=colorMap[2]),mode='markers', name='sd_correct_dtm10')
    scatter4 = go.Scatter(x=df['Distance'], y=df['sd_correct_cop30']+df['dem_h_dtm1']-ref, marker=dict(color=colorMap[6]),mode='markers', name='sd_correct_cop30')
    scatter5 = go.Scatter(x=df['Distance'], y=df['sd_correct_fab']+df['dem_h_dtm1']-ref, marker=dict(color=colorMap[7]),mode='markers', name='sd_correct_fab')
    scatter_tree = go.Scatter(x=df['Distance'], y=df['h_mean_canopy']+df['dem_h_dtm1']-ref, marker=dict(color=colorMap[7]),mode='markers', name='h_mean_canopy')

    # add the scatter plot trace to the figure
    fig.add_trace(scatter1,row=row, col=col)
    fig.add_trace(scatter2,row=row, col=col)
    fig.add_trace(scatter3,row=row, col=col)
    fig.add_trace(scatter4,row=row, col=col)
    fig.add_trace(scatter5,row=row, col=col)
    fig.add_trace(scatter_tree,row=row, col=col)

    # add the line chart trace to the figure
    fig.add_trace(trace1,row=row, col=col)
    fig.add_trace(trace2,row=row, col=col)
    fig.add_trace(trace3,row=row, col=col)
    fig.add_trace(trace4,row=row, col=col)
    fig.add_trace(trace5,row=row, col=col)
    fig.add_trace(trace6,row=row, col=col)
    fig.add_trace(trace7,row=row, col=col)

    # add a color bar to the figure
    fig.update_layout(coloraxis=dict(colorbar=dict(title='snow depth',
                                                    tickfont=dict(size=10, color='rgb(107, 107, 107)'),
                                                    tickcolor='rgb(107, 107, 107)',
                                                    len=0.5,
                                                    thickness=20,
                                                    y=0.2),
                                    colorscale='matter'))

def dem_profiler_sf_plot(df,fig,row,col,error_y_dict=None,lgroup=None,ref=None):

    '''
    Used to plot dataframe genarated from xsnow.godh.dem_profiler.
    
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1,shared_xaxes='all',shared_yaxes='all',vertical_spacing=0.03, subplot_titles=("a", 'b'))
    df_plot(df_profile_2011,fig, row=1, col=1,lgroup=None)
    df_plot(df_profile_2111,fig, row=2, col=1,lgroup=None)
    fig.update_layout(height=900, width=1800)
    fig.show()

    Change rows and cols accoding to subplots
    '''

    # color map
    colorMap = px.colors.qualitative.Bold

    if ref:
        ref = df[ref] 
    else:
        ref = 0
    # create a trace for col1
    trace1 = go.Scatter(x=df['Distance'], y=df['dem_h_dtm1']-ref, name='dtm1', mode='lines',legendgroup=lgroup,line_color=colorMap[1])
    trace2 = go.Scatter(x=df['Distance'], y=df['dem_h_dtm10']-ref, name='dtm10', mode='lines',legendgroup=lgroup,line_color=colorMap[2])
    trace3 = go.Scatter(x=df['Distance'], y=df['dem_h_lidar_n']-ref, name='lidar_n', mode='lines',legendgroup=lgroup,line_color=colorMap[3])
    trace4 = go.Scatter(x=df['Distance'], y=df['dem_h_lidar_s']-ref, name='lidar_s', mode='lines',legendgroup=lgroup,line_color=colorMap[4])
    trace5 = go.Scatter(x=df['Distance'], y=df['dem_h_dem_arctic']-ref, name='arctic', mode='lines',legendgroup=lgroup,line_color=colorMap[5])
    trace6 = go.Scatter(x=df['Distance'], y=df['dem_h_dem_cop30']-ref, name='cop30', mode='lines',legendgroup=lgroup,line_color=colorMap[6])
    trace7 = go.Scatter(x=df['Distance'], y=df['dem_h_dem_fab']-ref, name='fab', mode='lines',legendgroup=lgroup,line_color=colorMap[7])

    # create a scatter plot using plotly.express
    scatter1 = go.Scatter(
    x=df['Distance'],
    y=df['h_te_best_fit']-ref,
    marker=dict(color='black'),
    mode='markers',
    name='ATL08_best_fit',
    hoverinfo='text',
    text=[
        f"dh_after_dtm10: {val1}<br>dh_after_dtm1: {val2}<br>dh_after_cop30: {val3}<br>dh_after_fab: {val4}"
        for val1, val2, val3, val4 in zip(df['dh_after_dtm10'], df['dh_after_dtm1'], df['dh_after_cop30'], df['dh_after_fab'])
    ])

    scatter2 = go.Scatter(x=df['Distance'], y=df['dh_reg_dtm1']+df['dem_h_dtm1']-ref, 
                          marker=dict(color=colorMap[1]), mode='markers',
                          name='DTM1_reg',hoverinfo='text',
                          text=[f"dh_reg_dtm1: {val1}<br>dh_after_dtm1: {val2}<br>h_mean_canopy: {val3}<br>n_te_photons: {val4}<br>h_te_skew: {val5}<br>segment_cover: {val6}<br>canopy_openness: {val7}" 
                          for val1, val2, val3, val4, val5, val6, val7 in zip(df['dh_reg_dtm1'], df['dh_after_dtm1'], df['h_mean_canopy'], df['n_te_photons'], df['h_te_skew'], df['segment_cover'], df['canopy_openness'])
        ])

    scatter3 = go.Scatter(x=df['Distance'], y=df['dh_reg_dtm10']+df['dem_h_dtm1']-ref, marker=dict(color=colorMap[2]), mode='markers',
        name='DTM10_reg',hoverinfo='text',
        text=[f"dh_reg_dtm10: {val1}<br>dh_after_dtm10: {val2}" for val1, val2 in zip(df['dh_reg_dtm10'], df['dh_after_dtm10'])
        ])

    tree = go.Scatter(x=df['Distance'], y=df['h_mean_canopy']+df['dem_h_dtm1']-ref, name='h_mean_canopy', mode='markers', 
                        marker=dict(color='rgba(135, 206, 250, 0.5)', size=10, symbol='arrow'))

    # add the scatter plot trace to the figure
    fig.add_trace(scatter1,row=row, col=col)
    fig.add_trace(scatter2,row=row, col=col)
    fig.add_trace(scatter3,row=row, col=col)
    fig.add_trace(tree,row=row, col=col)

    # add the line chart trace to the figure
    fig.add_trace(trace1,row=row, col=col)
    fig.add_trace(trace2,row=row, col=col)
    fig.add_trace(trace3,row=row, col=col)
    fig.add_trace(trace4,row=row, col=col)
    fig.add_trace(trace5,row=row, col=col)
    fig.add_trace(trace6,row=row, col=col)
    fig.add_trace(trace7,row=row, col=col)



def plot_point_map(col,
                   df,
                   clim=(-2,2),
                   title='snowdepth',
                   cmap='Spectral',
                   sampling=0.01,
                   aggregate='mean',
                   colorbar=True,
                   tiles=gvts.CartoEco,
                   alpha=1):
    
    '''
    Plot point map with datashader
    
    col is a columns name in df
    df is a dataframe
    '''


    opts_ = dict(width=600, height=500, tools=['hover'], 
            colorbar=colorbar,symmetric=True,clim=clim,
            cmap=cmap,alpha=alpha) # ,clabel='[m]', colorbar_opts={'title_standoff':-150, 'padding':15}

    points = gv.Points(df, kdims=['longitude','latitude'],vdims=[col])
    map = rasterize(points, x_sampling=sampling, y_sampling=sampling,aggregator=aggregate).options(**opts_,title=title)
    if tiles:
        return map*tiles.opts(width = 400, height=400)
    else:
        return map.opts(width = 400, height=400)

def plot_bias_and_pred_bias_b(df_):
    '''
    use this if applying vertical bias correction in the first step
    '''

    fig,axs = plt.subplots(2,2,figsize=(12,12))
    final_histogram(df_['dh_after_dtm10_b'],df_['dh_reg_dtm10'],dH_ref=df_['dh_before_dtm10'],ax=axs[0,0],legend=['After coreg','After bias-correction','Raw'],range=(-10,10),perc_t=100)
    final_histogram(df_['dh_after_dtm1_b'],df_['dh_reg_dtm1'],dH_ref=df_['dh_before_dtm1'],ax=axs[0,1],legend=['After coreg','After bias-correction','Raw'],range=(-10,10),perc_t=100)
    final_histogram(df_['dh_after_cop30_b'],df_['dh_reg_cop30'],dH_ref=df_['dh_before_cop30'],ax=axs[1,0],legend=['After coreg','After bias-correction','Raw'],range=(-10,10),perc_t=100)
    final_histogram(df_['dh_after_fab_b'],df_['dh_reg_fab'],dH_ref=df_['dh_before_fab'],ax=axs[1,1],legend=['After coreg','After bias-correction','Raw'],range=(-10,10),perc_t=100)

    snow_dtm1_b = plot_point_map('dh_after_dtm1_b',df_,title='DTM1 -  diff [m]',clim=(-1.5,1.5),cmap='bwr')
    snow_dtm10_b = plot_point_map('dh_after_dtm10_b',df_,title='DTM10 - diff [m]',clim=(-1.5,1.5),cmap='bwr')
    snow_cop30_b = plot_point_map('dh_after_cop30_b',df_,title='COP30 - diff [m]',clim=(-1.5,1.5),cmap='bwr')
    snow_fab_b = plot_point_map('dh_after_fab_b',df_,title='FAB - diff [m]',clim=(-1.5,1.5),cmap='bwr')

    snow_dtm1 = plot_point_map('pred_correct_dtm1',df_,title='DTM1 -  bias [m]',clim=(-1.5,1.5),cmap='bwr')
    snow_dtm10 = plot_point_map('pred_correct_dtm10',df_,title='DTM10 - bias [m]',clim=(-1.5,1.5),cmap='bwr')
    snow_cop30 = plot_point_map('pred_correct_cop30',df_,title='COP30 - bias [m]',clim=(-1.5,1.5),cmap='bwr')
    snow_fab = plot_point_map('pred_correct_fab',df_,title='FAB - bias [m]',clim=(-1.5,1.5),cmap='bwr')

    return (snow_dtm10_b + snow_dtm1_b + snow_cop30_b + snow_fab_b + snow_dtm10 + snow_dtm1 + snow_cop30 + snow_fab).cols(4)

def plot_all_hist_sf(df_,std=3,perc_t=99,window=(-10,10),range=(-10,10)):
    fig,axs = plt.subplots(2,2,figsize=(15,10))

    if 'dh_before_dtm10' in df_.keys():
        '''
        dh_after is after coregistration
        dh_reg is after bias-correction regression
        '''

        final_histogram(df_['dh_after_dtm10'],df_['dh_reg_dtm10'],dH_ref=df_['dh_before_dtm10'],ax=axs[0,1],legend=['After co-registration','After bias-correction','Raw'],range=range,std_t=std,perc_t=perc_t,window=window,quantile=True,title='DTM1')
        final_histogram(df_['dh_after_dtm1'],df_['dh_reg_dtm1'],dH_ref=df_['dh_before_dtm1'],ax=axs[0,0],legend=['After co-registration','After bias correction','Raw'],range=range,std_t=std,perc_t=perc_t,window=window,quantile=True,title='DTM10')
        final_histogram(df_['dh_after_cop30'],df_['dh_reg_cop30'],dH_ref=df_['dh_before_cop30'],ax=axs[1,0],legend=['After co-registration','After bias correction','Raw'],range=range,std_t=std,perc_t=perc_t,window=window,quantile=True,title='COP30')
        final_histogram(df_['dh_after_fab'],df_['dh_reg_fab'],dH_ref=df_['dh_before_fab'],ax=axs[1,1],legend=['After co-registration','After bias-correction','Raw'],range=range,std_t=std,perc_t=perc_t,window=window,quantile=True,title='FAB')
    else:
        final_histogram(df_['dh_after_dtm10'],df_['dh_reg_dtm10'],ax=axs[0,0],legend=['After coreg','After bias-correction','Raw'],range=range,std_t=std,perc_t=perc_t,window=window,quantile=True)
        final_histogram(df_['dh_after_dtm1'],df_['dh_reg_dtm1'],ax=axs[0,1],legend=['After coreg','After bias-correction','Raw'],range=range,std_t=std,perc_t=perc_t,window=window,quantile=True)
        final_histogram(df_['dh_after_cop30'],df_['dh_reg_cop30'],ax=axs[1,0],legend=['After coreg','After bias-correction','Raw'],range=range,std_t=std,perc_t=perc_t,window=window,quantile=True)
        final_histogram(df_['dh_after_fab'],df_['dh_reg_fab'],ax=axs[1,1],legend=['After coreg','After bias-correction','Raw'],range=range,std_t=std,perc_t=perc_t,window=window,quantile=True)
        
    for ax in axs.flatten():
        ax.grid(False)

def plot_hist_sd_vs_era(df,min=0):
    '''
    snow depth(>0 m) hist from different model
    '''

    fig,axs = plt.subplots(1,2,figsize=(14,6))
    df_era = df.query('0 < sd_era <10')
    df_dtm1 = df.query(f'{min} < sd_correct_dtm1 < 10')
    df_dtm10 = df.query(f'{min} < sd_correct_dtm10 < 10')

    final_histogram(df_dtm1['sd_correct_dtm1'],df_dtm10['sd_correct_dtm10'],dH_ref=df_era['sd_era'],ax=axs[0],legend=['ICESat-2 - DTM1','ICESat-2 - DTM10','ERA5 Land'],range=(-4,8),window=(-4,8));
    
    if 'sd_correct_cop30' in df.keys():
        df_cop30 = df.query(f'{min} < sd_correct_cop30 < 10')
        df_fab = df.query(f'{min} < sd_correct_fab <10')
        print('N (era,dtm1,dtm10,cop30,fab):',len(df_era),len(df_dtm1),len(df_dtm10),len(df_cop30),len(df_fab))
        print('% (dtm1,dtm10,cop30,fab):',1-len(df_dtm1)/len(df_era),1-len(df_dtm10)/len(df_era),1-len(df_cop30)/len(df_era),1-len(df_fab)/len(df_era))
        final_histogram(df_cop30['sd_correct_cop30'],df_fab['sd_correct_fab'],dH_ref=df_era['sd_era'],ax=axs[1],legend=['ICESat-2 - GLO30','ICESat-2 - FAB','ERA5 Land'],range=(-4,8),window=(-4,8));

def plot_quantile_grid(sf_df_1):

    # Define nmad as a lambda function
    nmad = lambda x: xdem.spatialstats.nmad(x)

    heatmap_dic_nmad = {}
    heatmap_dic_median = {}
    heatmap_dic_mean = {}

    for _f in ['E', 'N','h_te_best_fit','slope', 'aspect', 'planc','profc','tpi','tpi_9','tpi_27','curvature','sd_era']:
        try:
            sf_df_1[f'q_{_f}'] = pd.qcut(sf_df_1[_f], q=10,precision=3,duplicates='drop')
        except ValueError as err:
            print(err,_f)
        heatmap_dic_nmad[f'q_{_f}'] = sf_df_1.groupby(f'q_{_f}').agg({'df_dtm1_era5': nmad, 'df_dtm10_era5': nmad, 'df_cop30_era5': nmad, 
                                                                                'df_fab_era5': nmad})

        heatmap_dic_median[f'q_{_f}'] = sf_df_1.groupby(f'q_{_f}').agg({'df_dtm1_era5': 'median', 'df_dtm10_era5': 'median', 'df_cop30_era5': 'median', 
                                                                                'df_fab_era5': 'median'})
        
        
        heatmap_dic_mean[f'q_{_f}'] = sf_df_1.groupby(f'q_{_f}').agg({'df_dtm1_era5': 'mean', 'df_dtm10_era5': 'mean', 'df_cop30_era5': 'mean', 
                                                                                'df_fab_era5': 'mean'})

    nmad_dem_dic = {}
    median_dem_dic = {}
    mean_dem_dic = {}
    index_dic = {}

    for dem in ['df_dtm1_era5', 'df_dtm10_era5','df_cop30_era5','df_fab_era5']:
        nmad_dict = {}
        median_dict = {}
        mean_dict = {}

        for _f in ['E', 'N','h_te_best_fit','slope', 'aspect', 'planc','profc','tpi','tpi_9','tpi_27','curvature','sd_era']:
            nmad_dict[_f] = heatmap_dic_nmad[f'q_{_f}'][dem].values
            median_dict[_f] = heatmap_dic_median[f'q_{_f}'][dem].values
            mean_dict[_f] = heatmap_dic_mean[f'q_{_f}'][dem].values

            if _f not in index_dic:
                index_dic[_f] = heatmap_dic_nmad[f'q_{_f}'].index
        nmad_dem_dic[dem] = nmad_dict
        median_dem_dic[dem] = median_dict
        mean_dem_dic[dem] = mean_dict

    sdh_dtm1_nmad = pd.DataFrame(nmad_dem_dic['df_dtm1_era5'])
    sdh_dtm10_nmad = pd.DataFrame(nmad_dem_dic['df_dtm10_era5'])
    sdh_cop30_nmad = pd.DataFrame(nmad_dem_dic['df_cop30_era5'])
    sdh_fab_nmad = pd.DataFrame(nmad_dem_dic['df_fab_era5'])

    sdh_dtm1_median = pd.DataFrame(median_dem_dic['df_dtm1_era5'])
    sdh_dtm10_median = pd.DataFrame(median_dem_dic['df_dtm10_era5'])
    sdh_cop30_median = pd.DataFrame(median_dem_dic['df_cop30_era5'])
    sdh_fab_median = pd.DataFrame(median_dem_dic['df_fab_era5'])

    sdh_dtm1_mean = pd.DataFrame(mean_dem_dic['df_dtm1_era5'])
    sdh_dtm10_mean = pd.DataFrame(mean_dem_dic['df_dtm10_era5'])
    sdh_cop30_mean = pd.DataFrame(mean_dem_dic['df_cop30_era5'])
    sdh_fab_mean = pd.DataFrame(mean_dem_dic['df_fab_era5'])

    p_median = plot_heatmap(sdh_dtm10_median,sdh_dtm1_median,sdh_cop30_median,sdh_fab_median,pp='sd_median_heatmap_vs_era5',title=['DTM10','DTM1','GLO30','FAB'],clim=(-2,2),cmap='coolwarm_r')
    p_mu = plot_heatmap(sdh_dtm10_mean,sdh_dtm1_mean,sdh_cop30_mean,sdh_fab_mean,pp='sd_median_heatmap_vs_era5',title=['DTM10','DTM1','GLO30','FAB'],clim=(-2,2),cmap='coolwarm_r')
    p_nmad = plot_heatmap(sdh_dtm10_nmad,sdh_dtm1_nmad,sdh_cop30_nmad,sdh_fab_nmad,pp='sd_nmad_heatmap_vs_era5',title=['DTM10','DTM1','GLO30','FAB'],clim=(0,2))

    return p_median,p_mu,p_nmad

def plot_quantile_heatmaps(sf_df_1, dem_cols,v_t=None):

    if v_t is None:
        v_t = ['E', 'N', 'h_te_best_fit', 'slope', 'aspect', 'planc', 'profc', 'tpi', 'tpi_9', 'tpi_27', 'curvature', 'sd_era','h_mean_canopy','canopy_openness']
    # Define nmad as a lambda function
    nmad = lambda x: xdem.spatialstats.nmad(x)

    heatmap_dic_nmad = {}
    heatmap_dic_median = {}
    heatmap_dic_mean = {}

    for _f in v_t:
        try:
            sf_df_1.loc[:,f'q_{_f}'] = pd.qcut(sf_df_1[_f], q=10, precision=3, duplicates='drop')
        except ValueError as err:
            print(err, _f)
        
        heatmap_dic_nmad[f'q_{_f}'] = sf_df_1.groupby(f'q_{_f}').agg({col: nmad for col in dem_cols})
        heatmap_dic_median[f'q_{_f}'] = sf_df_1.groupby(f'q_{_f}').agg({col: 'median' for col in dem_cols})
        heatmap_dic_mean[f'q_{_f}'] = sf_df_1.groupby(f'q_{_f}').agg({col: 'mean' for col in dem_cols})

    nmad_dem_dic = {}
    median_dem_dic = {}
    mean_dem_dic = {}
    index_dic = {}

    for dem in dem_cols:
        nmad_dict = {}
        median_dict = {}
        mean_dict = {}

        for _f in v_t:
            nmad_dict[_f] = heatmap_dic_nmad[f'q_{_f}'][dem].values
            median_dict[_f] = heatmap_dic_median[f'q_{_f}'][dem].values
            mean_dict[_f] = heatmap_dic_mean[f'q_{_f}'][dem].values

            if _f not in index_dic:
                index_dic[_f] = heatmap_dic_nmad[f'q_{_f}'].index

        nmad_dem_dic[dem] = nmad_dict
        median_dem_dic[dem] = median_dict
        mean_dem_dic[dem] = mean_dict

    try:
        sdh_dem_cols_nmad = {col: pd.DataFrame(nmad_dem_dic[col]) for col in dem_cols}
        sdh_dem_cols_median = {col: pd.DataFrame(median_dem_dic[col]) for col in dem_cols}
        sdh_dem_cols_mean = {col: pd.DataFrame(mean_dem_dic[col]) for col in dem_cols}
    except ValueError as err:
        print(err, nmad_dem_dic)

    p_median = plot_heatmap(*sdh_dem_cols_median.values(), pp='sd_median_heatmap_vs_era5',
                            title=list(dem_cols), clim=(-2, 2), cmap='coolwarm_r')
    p_mu = plot_heatmap(*sdh_dem_cols_mean.values(), pp='sd_median_heatmap_vs_era5',
                        title=list(dem_cols), clim=(-2, 2), cmap='coolwarm_r')
    p_nmad = plot_heatmap(*sdh_dem_cols_nmad.values(), pp='sd_nmad_heatmap_vs_era5',
                        title=list(dem_cols), clim=(0, 2))

    return p_median,p_mu,p_nmad

def plot_heatmap(df_0,df_1,df_2,df_3,pp='save',title='Title',cmap=None,clim=(0,4)):
    # Convert any Series objects to DataFrame objects
    if isinstance(df_0, pd.Series):
        df_0 = df_0.to_frame()
    if isinstance(df_1, pd.Series):
        df_1 = df_1.to_frame()
    if isinstance(df_2, pd.Series):
        df_2 = df_2.to_frame()
    if isinstance(df_3, pd.Series):
        df_3 = df_3.to_frame()
    
    if cmap is None:
        cmap = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    map_0 = df_0.T.hvplot.heatmap(x='columns', y='index',title=title[0],xlabel='Quantiles',colorbar=False,width=410) * hv.Text(6, 'h_te_best_fit', f'Average: {df_0.mean().mean():.2f} m',fontsize=9)
    map_1 = df_1.T.hvplot.heatmap(x='columns', y='index',title=title[1],xlabel='Quantiles',colorbar=False,yaxis=None,width=250) * hv.Text(6, 'h_te_best_fit',  f'Average: {df_1.mean().mean():.2f} m',fontsize=9)
    map_2 = df_2.T.hvplot.heatmap(x='columns', y='index',title=title[2],xlabel='Quantiles',colorbar=False,yaxis=None,width=250) * hv.Text(6, 'h_te_best_fit',  f'Average: {df_2.mean().mean():.2f} m',fontsize=9)
    map_3 = df_3.T.hvplot.heatmap(x='columns', y='index',title=title[3],xlabel='Quantiles',yaxis=None,width=320) * hv.Text(6, 'h_te_best_fit', f'Average: {df_3.mean().mean():.2f} m',fontsize=9)

    layout = (map_0 + map_1 + map_2 + map_3).cols(4)
    map = layout.opts(opts.HeatMap(
        cmap=cmap, 
        clim=clim,
        invert_yaxis=True,
        height=300, 
        toolbar=None, 
        fontsize={'title': 10, 'xticks': 5, 'yticks': 11}
    ))
    hv.save(map,pp,fmt='png',dpi=300)
    return map


def plot_sc_parameter(df,tiles=True):
    '''
    produce a map, demonstrating the slope....
    '''

    slope = plot_point_map('slope',df,title='slope',clim=(0,30),sampling=0.01,cmap='viridis',tiles=tiles,colorbar=False).opts(width = 370,height=350,xaxis=None)
    photons = plot_point_map('n_te_photons',df,title='n_te_photons',clim=(50,500),sampling=0.01,cmap='RdBu_r',tiles=tiles).opts(yaxis=None,xaxis=None,width = 400,height=350)
    canopy = plot_point_map('canopy_openness',df,title='canopy openness',clim=(0,4),sampling=0.01,cmap='YlGn',tiles=tiles,colorbar=False).opts(width = 370,height=380)
    elev = plot_point_map('h_te_best_fit',df,title='elevation',clim=(0,2000),sampling=0.01,cmap='terrain',tiles=tiles).opts(yaxis=None,width = 400,height=380)
    
    pp = (slope + photons + canopy + elev).cols(2)
    return pp

def plot_sf_dh(df,min=-15,max=15):
    '''
    produce a map, demonstrating predict bias..
    '''
    corr_dtm10_era5 = plot_point_map('dh_after_dtm10',df.query(f'{min} < dh_after_dtm10 < {max}'),title='icesat - dem10',clim=(-2,2),sampling=0.01,cmap='RdBu',tiles=True,colorbar=False).opts(width = 370,height=350,xaxis=None)
    corr_dtm1_era5 = plot_point_map('dh_after_dtm1',df.query(f'{min} < dh_after_dtm1 < {max}'),title='icesat - dem1',clim=(-2,2),sampling=0.01,cmap='RdBu',tiles=True).opts(yaxis=None,xaxis=None,width = 400,height=350)
    corr_cop30_era5 = plot_point_map('dh_after_cop30',df.query(f'{min} < dh_after_cop30 < {max}'),title='icesat - cop30',clim=(-2,2),sampling=0.01,cmap='RdBu',tiles=True,colorbar=False).opts(width = 370,height=380)
    corr_fab_era5 = plot_point_map('dh_after_fab',df.query(f'{min} < dh_after_fab < {max}'),title='icesat - fab',clim=(-2,2),sampling=0.01,cmap='RdBu',tiles=True).opts(yaxis=None,width = 400,height=380)

    pp = (corr_dtm10_era5 + corr_dtm1_era5 + corr_cop30_era5 + corr_fab_era5).cols(2)
    return pp


def plot_sc_bias_pred(df,min=0,max=15):

    '''
    produce a map, demonstrating predict bias..'pred_correct_dtm1'
    '''

    corr_dtm10_era5 = plot_point_map('pred_correct_dtm10',df.query(f'{min} < sd_correct_dtm10 < {max}'),title='pred_correct_dtm10',clim=(-2,2),sampling=0.01,cmap='RdBu',tiles=True,colorbar=False).opts(width = 370,height=350,xaxis=None)
    corr_dtm1_era5 = plot_point_map('pred_correct_dtm1',df.query(f'{min} < sd_correct_dtm1 < {max}'),title='pred_correct_dtm1',clim=(-2,2),sampling=0.01,cmap='RdBu',tiles=True).opts(yaxis=None,xaxis=None,width = 400,height=350)
    corr_cop30_era5 = plot_point_map('pred_correct_cop30',df.query(f'{min} < sd_correct_cop30 < {max}'),title='pred_correct_cop30',clim=(-2,2),sampling=0.01,cmap='RdBu',tiles=True,colorbar=False).opts(width = 370,height=380)
    corr_fab_era5 = plot_point_map('pred_correct_fab',df.query(f'{min} < sd_correct_fab < {max}'),title='pred_correct_fab',clim=(-2,2),sampling=0.01,cmap='RdBu',tiles=True).opts(yaxis=None,width = 400,height=380)

    pp = (corr_dtm10_era5 + corr_dtm1_era5 + corr_cop30_era5 + corr_fab_era5).cols(2)
    return pp


def plot_map_sd_vs_era(df,condition='0 < sde_era < 10',min=0,max=15,tiles=gvts.CartoEco):
    '''
    4 model outpput snow depth vs era, plot the difference.
    '''
    df_ = df.query(condition)
    diff_dtm10_era5 = plot_point_map('df_dtm10_era5',df_.query(f'{min} < sd_correct_dtm10 < {max}'),title='diff dtm10-era',clim=(-2,2),sampling=0.01,cmap='RdBu',tiles=tiles,colorbar=False).opts(width = 370,height=350,xaxis=None)
    diff_dtm1_era5 = plot_point_map('df_dtm1_era5',df_.query(f'{min} < sd_correct_dtm1 < {max}'),title='diff dtm1-era',clim=(-2,2),sampling=0.01,cmap='RdBu',tiles=tiles).opts(yaxis=None,xaxis=None,width = 400,height=350)
    diff_cop30_era5 = plot_point_map('df_cop30_era5',df_.query(f'{min} < sd_correct_cop30 < {max}'),title='diff cop30-era',clim=(-2,2),sampling=0.01,cmap='RdBu',colorbar=False,tiles=tiles).opts(width = 370,height=380)
    diff_fab_era5 = plot_point_map('df_fab_era5',df_.query(f'{min} < sd_correct_fab < {max}'),title='diff fab-era',clim=(-2,2),sampling=0.01,cmap='RdBu',tiles=tiles).opts(yaxis=None,width = 400,height=380)

    return (diff_dtm10_era5 + diff_dtm1_era5 + diff_cop30_era5 + diff_fab_era5).cols(2)

def plot_map_sd_final(df,condition='0 < sde_era < 10',min=0,max=15,tiles=gvts.CartoEco):
    '''
    4 model outpput snow depth vs era, plot the difference.
    '''
    df_ = df.query(condition)
    diff_dtm10_era5 = plot_point_map('sd_predict_dtm10',df_.query(f'{min} < sd_predict_dtm10 < {max}'),title='diff dtm10-era',clim=(0,2),sampling=0.01,cmap='PuOr',tiles=tiles,colorbar=False).opts(width = 370,height=350,xaxis=None)
    diff_dtm1_era5 = plot_point_map('sd_predict_dtm1',df_.query(f'{min} < sd_predict_dtm1 < {max}'),title='diff dtm1-era',clim=(0,2),sampling=0.01,cmap='PuOr',tiles=tiles).opts(yaxis=None,xaxis=None,width = 400,height=350)
    diff_cop30_era5 = plot_point_map('sd_predict_cop30',df_.query(f'{min} < sd_predict_cop30 < {max}'),title='diff cop30-era',clim=(0,2),sampling=0.01,cmap='PuOr',colorbar=False,tiles=tiles).opts(width = 370,height=380)
    diff_fab_era5 = plot_point_map('sd_predict_fab',df_.query(f'{min} < sd_predict_fab < {max}'),title='diff fab-era',clim=(0,2),sampling=0.01,cmap='PuOr',tiles=tiles).opts(yaxis=None,width = 400,height=380)

    return (diff_dtm10_era5 + diff_dtm1_era5 + diff_cop30_era5 + diff_fab_era5).cols(2)


def plot_map_sd_and_era(df,condition='0 < sde_era < 10',min=0,max=10,tiles=gvts.StamenToner):
    '''
    4 model outpput snow depth vs era, plot the difference.
    '''
    df_ = df.query(condition)
    diff_dtm10_era5 = plot_point_map('sd_correct_dtm10',df_.query(f'{min} < sd_correct_dtm10 < {max}'),title='corr dtm10',clim=(0,2),sampling=0.01,cmap='PuOr',colorbar=False,tiles=tiles).opts(width = 370,height=350,xaxis=None)
    diff_dtm1_era5 = plot_point_map('sd_correct_dtm1',df_.query(f'{min} < sd_correct_dtm1 < {max}'),title='corr dtm1',clim=(0,2),sampling=0.01,cmap='PuOr',tiles=tiles).opts(yaxis=None,xaxis=None,width = 400,height=350)
    diff_cop30_era5 = plot_point_map('sd_correct_cop30',df_.query(f'{min} < sd_correct_cop30 < {max}'),title='corr cop30',clim=(0,2),sampling=0.01,cmap='PuOr',colorbar=False,tiles=tiles).opts(width = 370,height=380)
    diff_fab_era5 = plot_point_map('sd_correct_fab',df_.query(f'{min} < sd_correct_fab < {max}'),title='corr fab',clim=(0,2),sampling=0.01,cmap='PuOr',tiles=tiles).opts(yaxis=None,width = 400,height=380)
    diff_era5 = plot_point_map('sde_era',df_.query('0 < sde_era '),title='sde_era',clim=(0,2),sampling=0.01,cmap='PuOr',colorbar=False,tiles=tiles).opts(yaxis=None,width = 370,height=380)

    return (diff_dtm10_era5 + diff_dtm1_era5 + diff_cop30_era5 + diff_fab_era5 + diff_era5).cols(2)

def plot_map_raw_sd_and_era(df,condition='0 < sde_era < 10',min=0,max=10,tiles=gvts.CartoEco,cmap='PuOr'):
    '''
    4 model outpput snow depth vs era, plot the difference.
    '''
    df_ = df.query(condition)
    diff_dtm10 = plot_point_map('snowdepth_dtm10',df_.query(f'{min} < snowdepth_dtm10 < {max}'),title='raw snowdepth dtm10',clim=(0,2),sampling=0.01,cmap='PuOr',colorbar=False,tiles=tiles).opts(width = 370,height=350,xaxis=None)
    diff_dtm1 = plot_point_map('snowdepth_dtm1',df_.query(f'{min} < snowdepth_dtm1 < {max}'),title='raw snowdepth dtm1',clim=(0,2),sampling=0.01,cmap='PuOr',colorbar=False,tiles=tiles).opts(yaxis=None,xaxis=None,width = 350,height=350)
    diff_era5 = plot_point_map('sde_era',df_.query('0 < sde_era '),title='sde_era',clim=(0,2),sampling=0.01,cmap='PuOr',colorbar=True,tiles=tiles).opts(yaxis=None,xaxis=None,width = 400,height=350)
    diff_dtm10_corr = plot_point_map('sd_correct_dtm10',df_.query(f'{min} < sd_correct_dtm10 < {max}'),title='corr dtm10',clim=(0,2),sampling=0.01,cmap='PuOr',colorbar=False,tiles=tiles).opts(width = 370,height=380)
    diff_dtm1_corr = plot_point_map('sd_correct_dtm1',df_.query(f'{min} < sd_correct_dtm1 < {max}'),title='corr dtm1',clim=(0,2),sampling=0.01,cmap='PuOr',colorbar=False,tiles=tiles).opts(yaxis=None,width = 350,height=380)
    diff_dtm1_era5 = plot_point_map('df_dtm1_era5',df_.query(f'{min} < sd_correct_dtm1 < {max}'),title='diff dtm1-era',clim=(-2,2),sampling=0.01,cmap='RdBu',colorbar=True,tiles=tiles).opts(yaxis=None, width = 400,height=380)

    return (diff_dtm10 + diff_dtm1 + diff_era5 + diff_dtm10_corr + diff_dtm1_corr+ diff_dtm1_era5).cols(3)

def plot_map_raw_sd_and_era_cop(df,condition='0 < sde_era < 10',min=0,max=10,tiles=gvts.CartoEco,cmap='PuOr'):
    '''
    4 model outpput snow depth vs era, plot the difference.
    '''
    df_ = df.query(condition)
    diff_dtm10 = plot_point_map('snowdepth_cop30',df_.query(f'{min} < snowdepth_cop30 < {max}'),title='raw snowdepth cop30',clim=(0,2),sampling=0.01,cmap=cmap,colorbar=False,tiles=tiles).opts(width = 370,height=350,xaxis=None)
    diff_dtm1 = plot_point_map('snowdepth_fab',df_.query(f'{min} < snowdepth_fab < {max}'),title='raw snowdepth fab',clim=(0,2),sampling=0.01,cmap=cmap,colorbar=False,tiles=tiles).opts(yaxis=None,xaxis=None,width = 350,height=350)
    diff_era5 = plot_point_map('sde_era',df_.query('0 < sde_era '),title='sde_era',clim=(0,2),sampling=0.01,cmap=cmap,colorbar=True,tiles=tiles).opts(yaxis=None,xaxis=None,width = 400,height=350)
    diff_dtm10_corr = plot_point_map('sd_correct_cop30',df_.query(f'{min} < sd_correct_cop30 < {max}'),title='corr cop30',clim=(0,2),sampling=0.01,cmap=cmap,colorbar=False,tiles=tiles).opts(width = 370,height=380)
    diff_dtm1_corr = plot_point_map('sd_correct_fab',df_.query(f'{min} < sd_correct_fab < {max}'),title='corr fab',clim=(0,2),sampling=0.01,cmap=cmap,colorbar=False,tiles=tiles).opts(yaxis=None,width = 350,height=380)
    diff_dtm1_era5 = plot_point_map('df_cop30_era5',df_.query(f'{min} < snowdepth_cop30 < {max}'),title='diff cop30-era',clim=(-2,2),sampling=0.01,cmap=cmap,colorbar=True,tiles=tiles).opts(yaxis=None, width = 400,height=380)

    return (diff_dtm10 + diff_dtm1 + diff_era5 + diff_dtm10_corr + diff_dtm1_corr+ diff_dtm1_era5).cols(3)

def plot_map_tree(df_,condition='-10 < dh_after_dtm1 <10',ref='sde_era',clim=(0,2)):
    '''
    snow on vs snow free (tree parameters)
    '''
    df_
    diff_dtm10_era5 = plot_point_map('h_mean_canopy',df_.query(condition),title='h_mean_canopy',clim=(0,5),sampling=0.01,cmap='GnBu',colorbar=True).opts(width = 420,height=380)
    diff_dtm1_era5 = plot_point_map('segment_cover',df_.query(condition),title='segment_cover',clim=(0,100),sampling=0.01,cmap='Reds').opts(yaxis=None,width = 400,height=380)
    diff_fab_era5 = plot_point_map('canopy_openness',df_.query(condition),title='canopy_openness',clim=(0,4),sampling=0.01,cmap='YlGn').opts(yaxis=None,width = 400,height=380)
    diff_era5 = plot_point_map(ref,df_.query(condition),title=ref,clim=clim,sampling=0.01,cmap='PuOr',colorbar=True).opts(yaxis=None,width = 420,height=380)

    return (diff_dtm10_era5 + diff_dtm1_era5  +diff_fab_era5 + diff_era5).cols(4)


import seaborn as sns
import matplotlib.pyplot as plt

def plot_correction_comparision(new_df_era_sub,cor = 'sd_correct_dtm1', raw = 'snowdepth_dtm1'):
    '''
    does the bais correction works or not?
    '''
    
    # Add columns for over correction and non-correction
    new_df_era_sub['over_correction'] = new_df_era_sub[cor] - new_df_era_sub['sd_lidar']
    new_df_era_sub['non_correction'] = new_df_era_sub[raw] - new_df_era_sub['sd_lidar']

    fig, axs = plt.subplots(ncols=3,figsize=(15,4),sharey=False)

    # Plot the scatterplot and regression lines
    sns.set(style='ticks', font_scale=1.0)

    sns.regplot(x='slope', y='over_correction', data=new_df_era_sub, color='red', label='Bias corrected',ax=axs[0], scatter_kws={'alpha':0.3})
    sns.regplot(x='slope', y='non_correction', data=new_df_era_sub, color='green', label='Raw' ,ax=axs[0], scatter_kws={'alpha':0.3})
    axs[0].legend()

    axs[0].set_ylabel('Difference to validation [m]')
    axs[0].set_xlabel('Slope [°]')
    axs[0].set_ylim((-3, 1))

    sns.regplot(x='sd_lidar', y='over_correction', data=new_df_era_sub, color='red', label='Bias corrected',ax=axs[1], scatter_kws={'alpha':0.3})
    sns.regplot(x='sd_lidar', y='non_correction', data=new_df_era_sub, color='green', label='Raw',ax=axs[1], scatter_kws={'alpha':0.3})
    axs[1].legend()
    axs[1].set_ylabel('Difference to validation [m]')
    axs[1].set_xlabel('Snow depth [m]')

    sns.regplot(x='h_mean_canopy', y='over_correction', data=new_df_era_sub, color='red', label='Bias corrected',ax=axs[2], scatter_kws={'alpha':0.3})
    sns.regplot(x='h_mean_canopy', y='non_correction', data=new_df_era_sub, color='green', label='Raw',ax=axs[2], scatter_kws={'alpha':0.3})
    axs[2].legend()
    axs[2].set_ylabel('Difference to validation [m]')
    axs[2].set_xlabel('h_mean_canopy [m]')

    plt.show()

import numpy as np
from scipy.stats import linregress
import holoviews as hv
from sklearn.metrics import mean_squared_error

# Calculate R-squared
def plot_scater_nve(new_df_era_sub,
                    x_sd,
                    y_sd,
                    x_lim=(0,5),
                    y_lim=(0,5),
                    y_offset=0,
                    title='Plot'):
    '''
    plot scatter plot and calculate the R2
    '''

    x = new_df_era_sub[x_sd]
    y = new_df_era_sub[y_sd] - y_offset

    # Apply mask to x and y arrays to select non-missing values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    rmse = np.sqrt(mean_squared_error(x,y))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value ** 2
    cov_x = np.nanstd(x) / np.nanmean(x)
    cov_y = np.nanstd(y) / np.nanmean(y)
    #r_squ = np.corrcoef(x,y)[0, 1] **2
    corr_coef, p_value = spearmanr(x,y)

    # line
    xs = np.linspace(x.min(), x.max(), 100)
    ys = slope * xs + intercept
    base_line = hv.Curve((np.linspace(0, 10, 50), np.linspace(0, 10, 50))).opts(line_width=2, color='red',alpha=0.3)

    # metrics
    fit_line = hv.Curve((xs, ys)).opts(line_width=1, color='black',line_dash='dashed')
    equation = hv.Text(4, 0.75, f"y = {intercept:.2f} + {slope:.2f}x").opts(color='black', text_font_size='9pt')
    r_2 = hv.Text(4, 0.55, f"R\u00B2: {r_squared:.2f} RMSE:{rmse:.2f}").opts(color='black', text_font_size='9pt') 
    number = hv.Text(4, 0.35, f"N: {len(x)}").opts(color='black', text_font_size='9pt') 
    sprc = hv.Text(4, 0.15, f"Spearman Coeff: {corr_coef:.2f}").opts(color='black', text_font_size='9pt') 

    # Create scatter plot with R-squared annotation
    sc = hv.Points((x, y)).opts(width=400, height=400,xlim=x_lim,ylim=y_lim,xlabel='Lidar survey [m]',ylabel='ICESat-2 [m]').opts(title=title) 
    kde_x = hv.Distribution(x).opts(height=50,xlim=x_lim,yaxis=None,xaxis=None) * hv.Text(3.8, 0.2, f"CoV: {cov_x:.2f}  Mean: {np.mean(x):.2f}").opts(text_font_size='8pt')
    kde_y = hv.Distribution(y).opts(width=50,xlim=x_lim,yaxis=None,xaxis=None) * hv.Text(3.8, 0.3, f"CoV: {cov_y:.2f}  Mean: {np.mean(y):.2f}").opts(text_font_size='8pt',angle=-90)

    # Show the plot
    return sc* fit_line*base_line *equation*r_2*number*sprc  << kde_y << kde_x

import pandas as pd
import numpy as np
import datashader as ds
import datashader.transfer_functions as tf
from datashader import reductions
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews.operation.datashader import datashade, dynspread

def plot_scater_nve_new(new_df_era_sub,x_sd,y_sd,title='Plot',log=False):
    x = new_df_era_sub[x_sd]
    y = new_df_era_sub[y_sd]

    # Apply mask to x and y arrays to select non-missing values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if log:
        y = np.log(y)

    rmse = np.sqrt(mean_squared_error(x,y))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    r_squared = r_value ** 2
    cov_x = np.nanstd(x) / np.nanmean(x)
    cov_y = np.nanstd(y) / np.nanmean(y)

    xs = np.linspace(x.min(), x.max(), 100)
    ys = slope * xs + intercept
    base_line = hv.Curve((np.linspace(0, 10, 50), np.linspace(0, 10, 50))).opts(line_width=2, color='red',alpha=0.3)

    fit_line = hv.Curve((xs, ys)).opts(line_width=1, color='black',line_dash='dashed')
    equation = hv.Text(4, 0.75, f"y = {intercept:.2f} + {slope:.2f}x").opts(color='black', text_font_size='9pt')
    r_2 = hv.Text(4, 0.55, f"R\u00B2: {r_squared:.2f} RMSE:{rmse:.2f}").opts(color='black', text_font_size='9pt') 
    number = hv.Text(4, 0.35, f"N: {len(x)}").opts(color='black', text_font_size='9pt') 

    # Create scatter plot with R-squared annotation using datashader
    cvs = ds.Canvas(plot_width=400, plot_height=400, x_range=(0, 5), y_range=(0, 5))
    agg = cvs.points(pd.DataFrame({'x': x, 'y': y}), 'x', 'y', ds.count())
    img = tf.shade(agg, cmap=['white', 'black'], how='eq_hist')
    ds_scatter = dynspread(img, threshold=0.5, max_px=4)
    ds_scatter = hv.Image(ds_scatter).opts(xlabel='Lidar survey [m]',ylabel='ICESat-2 [m]',title=title)

    kde_x = hv.Distribution(x).opts(height=50,xlim=(0,5),yaxis=None,xaxis=None) * hv.Text(3.8, 0.2, f"CoV: {cov_x:.2f}  Mean: {np.mean(x):.2f}").opts(text_font_size='8pt')
    kde_y = hv.Distribution(y).opts(width=50,xlim=(0,5),yaxis=None,xaxis=None) * hv.Text(3.8, 0.3, f"CoV: {cov_y:.2f}  Mean: {np.mean(y):.2f}").opts(text_font_size='8pt',angle=-90)

    # Show the plot
    return ds_scatter* fit_line*base_line *equation*r_2*number  << kde_y << kde_x


def plot_all_nve_validation(new_df_era_sub,v='sd_nve',title='snowdepth_nve_scatter_all_qc_all'):
    #new_df_era_sub = regression_sd_single(new_df_era_sub, sd_reg_dtm10,sd_reg_dtm1,sd_reg_cop30,sd_reg_fab)
    era_p =plot_scater_nve(new_df_era_sub.query(f'{v}>0'),v,'sd_era')
    dtm_10_p =plot_scater_nve(new_df_era_sub.query(f'{v}>0'),v,'sd_predict_dtm10')
    dtm_1_p =plot_scater_nve(new_df_era_sub.query(f'{v}>0'),v,'sd_predict_dtm1')
    dtm_cop_p =plot_scater_nve(new_df_era_sub.query(f'{v}>0'),v,'sd_predict_cop30')
    dtm_fab_p =plot_scater_nve(new_df_era_sub.query(f'{v}>0'),v,'sd_predict_fab')

    p1 = (era_p + dtm_10_p+dtm_1_p+dtm_cop_p+dtm_fab_p).cols(5)
    #p1 = (era_p[0] + dtm_10_p[0]+dtm_1_p[0]+dtm_cop_p[0]+dtm_fab_p[0]).cols(5)
    #p2 = (era_p[1] + dtm_10_p[1]+dtm_1_p[1]+dtm_cop_p[1]+dtm_fab_p[1]).cols(5)
    hv.save(p1,title,fmt='png',dpi=300)
    #hv.save(p2,'snowdepth_nve_kde_all_qc',fmt='png',dpi=300)
    #(p2 + p1).cols(5)
    return p1

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr,spearmanr
import datashader as ds
from datashader.mpl_ext import dsshow
import pandas as pd
from sklearn.metrics import r2_score


def using_datashader(ax, x, y,vmax,vmin=0,cmap=plt.cm.PuRd):

    df = pd.DataFrame(dict(x=x, y=y))
    dsartist = dsshow(
        df,
        ds.Point("x", "y"),
        ds.count(),
        vmin=vmin,
        vmax=vmax,
        norm="linear",
        aspect="auto",
        ax=ax,
        cmap=cmap,
    )

    return dsartist

from scipy import stats
def plot_df_era_scatter(df,x='sd_era',y = 'sd_correct_dtm1',title='Correlation',
                        ax=None, x_lims=None,y_lims=None,y_offset=0,
                        vmax=1000,vmin=0,text=True,n_quantiles=30,fit_line=True,cmap=plt.cm.PuRd,bar=True):
    
    t_critical= 0.95
    if ax is None:
        fig, ax = plt.subplots()

    # Identify rows with NaN values in x or y
    nan_rows = np.isnan(df[x]) | np.isnan(df[y])

    # Remove rows with NaN values
    x_clean = df[x][~nan_rows].copy()
    y_clean = (df[y][~nan_rows] - y_offset).copy()

    number_of_df = len(df)
    ratio_of_x = df[x].count() / number_of_df
    ratio_of_y = df[y].count() / number_of_df
    number_of_xy = y_clean.count()

    if vmax is None:
        vmax = number_of_xy/10000

    plt.text(.90, .75, f'N = {number_of_df}', fontsize=10, c='r', ha='right', va='top', transform=ax.transAxes)


    if n_quantiles:
        df_quantiles = pd.qcut(df[x], q=n_quantiles, precision=2, labels=False, retbins=False, duplicates='drop')

        q = df.groupby(df_quantiles)
        n = q.size().values
        df_group_mu = q.mean()
        df_group_me = q.median()
        ax.plot(df_group_me[x], df_group_me[y]- y_offset, 'b.-', lw=1, markersize=6, alpha=0.5,label='Median')
        ax.plot(df_group_mu[x], df_group_mu[y]- y_offset, 'r.-', lw=1, markersize=6, alpha=0.5,label='Mean')
        

        # add C.I for mean and median 
        y_err_mu = t_critical * (q[y].std() / np.sqrt(n[0]))
        y_err_median_ = [stats.median_abs_deviation(group[1][y]) for group in q]
        y_err_median = 1.57 * np.array(y_err_median_) / np.sqrt(n)
        #ax.fill_between(df_group_me[x], df_group_me[y] - y_err_median, df_group_me[y] + y_err_median, alpha=0.15,facecolor='b')
        #ax.fill_between(df_group_mu[x], df_group_mu[y] - y_err_mu, df_group_mu[y] + y_err_mu, alpha=0.15,facecolor='r')
        ax.legend()

    dsartist = using_datashader(ax,  x_clean, y_clean,vmax=vmax,vmin=vmin,cmap=cmap)
    
    if bar:
        plt.colorbar(dsartist,ax=ax)

    if x_lims:
        ax.set_xlim(x_lims[0], x_lims[1])
    if y_lims:
        ax.set_ylim(y_lims[0], y_lims[1])

    if text:
        # Add Pearson's correlation coefficient to the plot
        corr_coef, p_value = spearmanr(x_clean,y_clean)
        plt.text(.90, .80, f'Spearman correlation = {corr_coef:.2f}', fontsize=10, c='r',ha='right', va='top', transform=ax.transAxes)
    
    # Add a regression line to the plot
    if fit_line:
        # Perform linear regression on the cleaned data
        #z = np.polyfit(x_clean, y_clean, 1)
        #p = np.poly1d(z)
        #ax.plot(df[x], p(df[x]), 'r--', alpha=0.5)
        
        slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
        r2 = r_value ** 2

        xs = np.linspace(x_clean.min(), x_clean.max(), 50)
        ys = slope * xs + intercept
        ax.plot(xs, ys, 'r--', alpha=0.5)
        print(f'y = {slope} * x + {intercept}')
        # Calculate R-squared value
        #r2 = r2_score(df[y], p(df[x]))
        plt.text(.90, .70, f'R\u00B2 = {r2:.2f}', fontsize=10, c='r',ha='right', va='top', transform=ax.transAxes)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    return ratio_of_x, ratio_of_y

import matplotlib.pyplot as plt
from datashader.mpl_ext import dsshow
import datashader as ds

def plot_datashaded_scatter(sf, x_col, y_col, value_col, ax=None, cmap=plt.cm.bwr, vmin=-5, vmax=5, xlim=(-10, 10), ylim=(0, 25)):
    """
    Plot a datashaded scatter plot with a colorbar
    :param sf: geopandas GeoDataFrame
    :param x_col: str, name of column with x values
    :param y_col: str, name of column with y values
    :param value_col: str, name of column with values to color by
    :param cmap: matplotlib colormap, default plt.cm.bwr
    :param vmin: float, minimum value of colormap, default -5
    :param vmax: float, maximum value of colormap, default 5
    :return: matplotlib Axes object
    """
    if ax is None:
        fig, ax = plt.subplots()

    dsartist = dsshow(
        sf,
        ds.Point(x_col, y_col),
        ds.mean(value_col),
        vmin=vmin,
        vmax=vmax,
        norm="linear",
        aspect="auto",
        ax=ax,
        cmap=cmap,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel(y_col)
    ax.set_xlabel(x_col)

    plt.colorbar(dsartist, ax=ax)

    return ax

import pandas as pd
import hvplot.pandas

def plot_segment_cover(sf, ax=None, title='Segment Cover vs Tree Presence'):


    # Create bins for the segment_cover column
    bins = pd.cut(sf['segment_cover'], bins=20)

    # Group the data by the new bins and tree_presence
    grouped = sf.groupby([bins, 'tree_presence']).size().unstack()

    if ax is None:
        fig,ax = plt.subplots()
       # Create the stacked bar chart using matplotlib
    ax = grouped.plot(kind='barh', stacked=True)
    ax.set_xlabel('Count')
    ax.set_ylabel('Segment Cover')
    ax.set_title(title)

def plot_metrics(data_m, data_s, 
                 show_legend=True,
                 radius_limits=[0, 3],
                 alpha=0.3,
                 tt=['a','b'],
                 title='metrics.png',
                 cmap='Tab20'):
    # Define the number of vertices (number of metrics)
    num_metrics = 4
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()

    # Create a figure with two subplots for Metric M and Metric S
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), subplot_kw={'projection': 'polar'})

    # Define colors for each group
    # Get a colormap
    colormap = plt.get_cmap(cmap)
    # Create a list of colors based on the number of groups
    colors = [colormap(i) for i in np.linspace(0, 1, len(data_m)+1)]

    # Plot Metric M
    for i, group in enumerate(data_m.keys()):
        values_m = data_m[group]
        ax1.fill(angles, values_m, color=colors[i], alpha=alpha, label=group)

    ax1.set_xticks(angles)
    ax1.set_xticklabels(['$R^2$', 'KSD', 'ρ', 'RMSE'])
    #ax1.set_yticklabels([0.25,0.5,0.75])
    ax1.yaxis.grid(True)
    ax1.set_title(tt[0])
    ax1.tick_params(pad=-20)
    ax1.set_rlabel_position(300)

    # Plot Metric S
    for i, group in enumerate(data_s.keys()):
        values_s = data_s[group]
        ax2.fill(angles, values_s, color=colors[i], alpha=alpha, label=group)

    ax2.set_xticks(angles)
    ax2.set_xticklabels(['$R^2$', 'KSD', 'ρ', 'RMSE'])
    ax2.tick_params(pad=-20)  # Adjust the distance of the labels from the circle
    #ax2.set_yticklabels([0.25,0.5,0.75])
    ax2.yaxis.grid(True)
    ax2.set_title(tt[1])
    ax2.set_rlabel_position(300)

    # Set radius limits
    if radius_limits is not None:
        ax1.set_ylim(radius_limits)
        ax2.set_ylim(radius_limits)

    # Show or hide legends based on the show_legend parameter
    if show_legend:
        #ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the subplots  
    plt.savefig(title, dpi=300)
    plt.show()  