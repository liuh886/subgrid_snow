from sympy import E, N
import xdem
from xdem.dem import DEM
import numpy as np
import geopandas as gpd
from collections import abc
from geopandas.geodataframe import GeoDataFrame
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_image
from scipy import ndimage
from geoutils import spatial_tools
import pyproj
from typing import IO, Any,Optional,Callable
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from rasterstats import zonal_stats
from shapely.geometry import LineString,Point
from shapely.wkt import loads
import copy
import rasterio as rio
from shapely.geometry.polygon import Polygon
import shapely.wkt
import pyproj
from pyproj import Transformer


def fill_by_nearest(data, invalid=None):
    """
    https://stackoverflow.com/questions/3662361/fill-in-missing-values-with-nearest-neighbour-in-python-numpy-masked-arrays

    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data', 'mask'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """

    if invalid is None: invalid = np.isnan(data)
    ind = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]

def xy2ij(
    dem:DEM,
    x: np.ndarray,
    y: np.ndarray,
    op: type = np.float32,
    area_or_point: str or None = None,        
    precision: float or None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Return row, column indices for a given x,y coordinate pair.
    :param x: x coordinates
    :param y: y coordinates
    :param op: operator to calculate index
    :param precision: precision for rio.Dataset.index
    :param area_or_point: shift index according to GDAL AREA_OR_POINT attribute (None) or \
        force position ('Point' or 'Area') of the interpretation of where the raster value \
        corresponds to in the pixel ('Area' = lower left or 'Point' = center)

    :returns i, j: indices of x,y in the image.

    edit from geoutils.georaster.raster.py

    # How to interpolate offset/area_or_point:
    # set ij2xy offset = center when xy2ij area_or_point = Point
    # set ij2xy offset = ul when xy2ij area_or_point = Area

    """
    if op not in [np.float32, np.float64, float]:
        raise UserWarning(
            "Operator is not of type float: rio.Dataset.index might "
            "return unreliable indexes due to rounding issues.")
    if area_or_point not in [None, "Area", "Point"]:
        raise ValueError(
            'Argument "area_or_point" must be either None (falls back to GDAL metadata), "Point" or "Area".'
            )

    # By defualt, this area_or_point == Area (interpret x,y as upper left corner of the cell)
    i, j = rio.transform.rowcol(dem.transform, x, y, op=op, precision=precision)

    #necessary because rio.Dataset.index does not return abc.Iterable for a single point
    i, j = (np.asarray(i), np.asarray(j)) if isinstance(i, abc.Iterable) else (np.asarray([i,]), np.asarray([j,]))

    # AREA_OR_POINT GDAL attribute, i.e. does the value refer to the upper left corner (AREA) or
    # the center of pixel (POINT)
    # This has no influence on georeferencing, it's only about the interpretation of the raster values,
    # and thus only affects sub-pixel interpolation

    # if input is None, default to GDAL METADATA
    if area_or_point is None:
        try:
            area_or_point = dem.tags["AREA_OR_POINT"]
        except (AttributeError,KeyError):
            area_or_point = 'Area'

    if area_or_point == "Area":
        i -= 0
        j -= 0

    elif area_or_point == "Point":
        if not isinstance(i.flat[0], np.floating):
            raise ValueError(
                "Operator must return np.floating values to perform AREA_OR_POINT subpixel index shifting"
            )

        # if point, shift index by half a pixel
        i -= 0.5
        j -= 0.5
        # otherwise, leave as is

    return i, j

def ij2xy(dem:DEM, i: np.ndarray, j: np.ndarray, offset: str = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Return x,y coordinates for a given row, column index pair.
    :param i: row (i) index of pixel.
    :param j: column (j) index of pixel.
    :param offset: return coordinates as "corner" or "center" of pixel. a corner can be returned by setting offset to one of ul, ur, ll, lr.

    :returns x, y: x,y coordinates of i,j in reference system.
    
    edit from geoutils.georaster.raster.py

    # How to interpolate offset/area_or_point:
    # set ij2xy offset = center (by defualt) when xy2ij area_or_point = Point
    # set ij2xy offset = ul (if dem.ds.tages == Area) when xy2ij area_or_point = Area (if dem.ds.tages == Area)

    """

    try:
        if offset is None and (dem.tags['AREA_OR_POINT'] in ('Point','point')):
            offset = 'center'
    except (KeyError, AttributeError):
        offset = 'ul'

    x, y = dem.ij2xy(i, j, offset=offset)
    return x, y

def outside_image(dem:DEM, xi: np.ndarray, yj: np.ndarray, index: bool = True) -> bool:
    """
    Check whether a given point falls outside of the raster.
    :param xi: Indices (or coordinates) of x direction to check.
    :param yj: Indices (or coordinates) of y direction to check.
    :param index: Interpret ij as raster indices (default is True). If False, assumes ij is coordinates.
    :returns is_outside: True if ij is outside of the image.
    
    edit from geoutils.georaster.raster.py

    """
    if not index:
        xi, xj = dem.xy2ij(xi, yj)
    if np.any(np.array((xi, yj)) < 0):
        return True
    elif xi > dem.width or yj > dem.height:
        return True
    else:
        return False

def interp_points(
    dem: DEM,
    pts: np.ndarray,
    input_latlon: bool = False,
    mode: str = "nearest",
    area_or_point: str or None = None,
    order=1,
    fillhole = False,
    **kwargs: Any,
    ) -> np.ndarray:

    x, y = list(zip(*pts))
    # if those are in latlon, convert to Raster crs
    if input_latlon and isinstance(dem,DEM):
        init_crs = pyproj.CRS(4326)
        dest_crs = pyproj.CRS(dem.crs)
        transformer = pyproj.Transformer.from_crs(init_crs, dest_crs)
        x, y = transformer.transform(x, y)
        
    i, j = xy2ij(dem,x, y, op=np.float32, area_or_point=area_or_point)

    # prepare DEM
    arr_ = dem.data[0, :, :]
    if fillhole and (not np.all(~arr_.mask)):   
        # if there is nodata, fill by nearest for interpolation(map_cordinate)
        arr_ = fill_by_nearest(arr_,arr_.mask)
    
    return ndimage.map_coordinates(arr_, [i, j],order=order,mode=mode,**kwargs)

def poly_zonal_stats(poly_list, dem: DEM, stats = None, all_touched=False):
    '''
    Using zonal_stats to extract zonal result from DEM

    poly_list should be the list of vector data: 
    https://pythonhosted.org/rasterstats/manual.html#vector-data-sources

    dem shoule be:
    https://pythonhosted.org/rasterstats/manual.html#raster-data-sources

    all_touched=True
    https://pythonhosted.org/rasterstats/manual.html#rasterization-strategy
    '''         
    if stats is None:
        stats = ["mean"]

    dem_arr, dem_mask = spatial_tools.get_array_and_mask(dem)
    # assign dem.nodata to np.nan
    # TODO :need test the statistics on nodata

    #dem_arr[dem_arr == dem.nodata] = np.nan
    stats_ = zonal_stats(poly_list,dem_arr,affine=dem.transform,nodata=dem.nodata,stats=stats,all_touched=all_touched)
    return pd.DataFrame(stats_)

def points_to_footprint(
    gdf:pd.DataFrame,
    footprint):

    poly_list = []

    # TODO :the direciton should be corrected, not just typical value +-5.8
    if 'region_ang' not in gdf:
        gdf['region_ang'] = gdf.region.map({3:-5.8,5:4.4,
                                        2:-5.6,6:5.4,
                                        1:-5.6,7:5.4,
                                        14:-5.6,8:5.4,
                                        13:-5.6,9:5.4,
                                        12:-5.6,10:5.4})

    for index, row in gdf.iterrows():
        # Turn ture north to grid north.
        # UTM33N center meridian is E15
        # https://gis.stackexchange.com/questions/115531/calculating-grid-convergence-true-north-to-grid-north/393677
        # convergence calculation is verified by https://sgss.ca/tools/coordcalc.html
        to_grid_north = row['region_ang'] - (np.arctan(np.tan((row['longitude'] - 15)/180*np.pi) *np.sin(row['latitude']/180*np.pi)) * 180 / np.pi)
        # to_grid_north = row['region_ang'] - (32.39 * (row['E']/1000-500) * np.tan(row['latitude']/180*np.pi) / 3600) 
        radians = to_grid_north * np.pi / 180

        Point_N = Point(row['E'] + footprint[1]/2*np.sin(radians), row['N'] + (footprint[1]/2-footprint[0]/2)*np.cos(radians))
        Point_S = Point(row['E'] - footprint[1]/2*np.sin(radians), row['N'] - (footprint[1]/2-footprint[0]/2)*np.cos(radians))
        poly_list.append(LineString([Point_S, Point_N]).buffer(footprint[0]/2,resolution=2))

    return poly_list

def gdf_to_dem(gdf:GeoDataFrame,dem:DEM,col='z'):
    # step 1 rasterize and repoject: turnning gdf into xarray -> DEM
    geo_grid = make_geocube(
    vector_data=gdf,
    measurements=[col],
    resolution=dem.res,
    align = tuple(i/2 for i in dem.res), # align to the middle of the pixel
    rasterize_function=rasterize_image,  # rasterize_image, NO interpolation
    )

    # step 2 create dem from array
    a ,b = geo_grid.rio.resolution()
    L,B,R,T = geo_grid.rio.bounds()
    return xdem.DEM.from_array(geo_grid[col].data, (a, 0.0, L, 0.0, b, B), geo_grid.rio.crs).reproject(dem)


def extend_geosegment(df,dem):
    
    '''
    extend ATL-08 with geosegment
    '''
    dst_crs = dem.crs if dem else pyproj.CRS(32633)
    
    assert 'latitude_20m_0' in df.columns

    df = df.query('subset_te_flag == 5').copy()
    df_new_0 = df[['latitude_20m_0','longitude_20m_0','h_te_best_fit_20m_0','subset_te_flag']].copy().rename(columns={'latitude_20m_0': 'latitude', 'longitude_20m_0': 'longitude','h_te_best_fit_20m_0':'h_te_best_fit'}).dropna()
    df_new_1 = df[['latitude_20m_1','longitude_20m_1','h_te_best_fit_20m_1','subset_te_flag']].copy().rename(columns={'latitude_20m_1': 'latitude', 'longitude_20m_1': 'longitude','h_te_best_fit_20m_1':'h_te_best_fit'}).dropna()
    df_new_3 = df[['latitude_20m_3','longitude_20m_3','h_te_best_fit_20m_3','subset_te_flag']].copy().rename(columns={'latitude_20m_3': 'latitude', 'longitude_20m_3': 'longitude','h_te_best_fit_20m_3':'h_te_best_fit'}).dropna()
    df_new_4 = df[['latitude_20m_4','longitude_20m_4','h_te_best_fit_20m_4','subset_te_flag']].copy().rename(columns={'latitude_20m_4': 'latitude', 'longitude_20m_4': 'longitude','h_te_best_fit_20m_4':'h_te_best_fit'}).dropna()
    for df_ in [df_new_0,df_new_1,df_new_3,df_new_4]:
        df_['subset_te_flag'] = df_['subset_te_flag'] / 2
    df_new = df.drop(columns=['latitude_20m_0','longitude_20m_0','h_te_best_fit_20m_0','latitude_20m_1','longitude_20m_1','h_te_best_fit_20m_1','latitude_20m_3','longitude_20m_3','h_te_best_fit_20m_3','latitude_20m_4','longitude_20m_4','h_te_best_fit_20m_4']).copy()
    df_concat = pd.concat([df_new, df_new_0, df_new_1, df_new_3, df_new_4], axis=0).reset_index(drop=True)
    transformer = Transformer.from_crs(pyproj.CRS(4326),dst_crs)
    df_concat['E'],df_concat['N'] = transformer.transform(df_concat['latitude'],df_concat['longitude'])

    return df_concat

def plot_variogram(df: pd.DataFrame, list_fit_fun: Optional[list[Callable[[np.ndarray], np.ndarray]]] = None,
                   list_fit_fun_label: Optional[list[str]] = None, ax: matplotlib.axes.Axes or None = None,
                   xscale='linear', xscale_range_split: Optional[list] = None,
                   xlabel = None, ylabel = None, xlim = None, ylim = None):
    """
    edit from xdem
    
    Plot empirical variogram, and optionally also plot one or several model fits.
    Input dataframe is expected to be the output of xdem.spatialstats.sample_empirical_variogram.
    Input function model is expected to be the output of xdem.spatialstats.fit_sum_model_variogram.

    :param df: Empirical variogram, formatted as a dataframe with count (pairwise sample count), lags
        (upper bound of spatial lag bin), exp (experimental variance), and err_exp (error on experimental variance).
    :param list_fit_fun: List of model function fits
    :param list_fit_fun_label: List of model function fits labels
    :param ax: Plotting ax to use, creates a new one by default
    :param xscale: Scale of X-axis
    :param xscale_range_split: List of ranges at which to split the figure
    :param xlabel: Label of X-axis
    :param ylabel: Label of Y-axis
    :param xlim: Limits of X-axis
    :param ylim: Limits of Y-axis
    :return:
    """

    # Create axes if they are not passed
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
    elif isinstance(ax, matplotlib.axes.Axes):
        fig = ax.figure
    else:
        raise ValueError("ax must be a matplotlib.axes.Axes instance or None")

    # Check format of input dataframe
    expected_values = ['exp', 'lags', 'count']
    for val in expected_values:
        if val not in df.columns.values:
            raise ValueError(f'The expected variable "{val}" is not part of the provided dataframe column names.')


    # Hide axes for the main subplot (which will be subdivded)
    ax.axis("off")

    if ylabel is None:
        ylabel = r'Variance [$\mu$ $\pm \sigma$]'
    if xlabel is None:
        xlabel = 'Spatial lag (m)'

    init_gridsize = [10, 10]
    # Create parameters to split x axis into different linear scales
    # If there is no split, get parameters for a single subplot
    if xscale_range_split is None:
        nb_subpanels=1
        xmin = [np.min(df.lags)/2] if xscale == 'log' else [0]
        xmax = [np.max(df.lags)]
        xgridmin = [0]
        xgridmax = [init_gridsize[0]]
        gridsize = init_gridsize
    else:
        # Add initial zero if not in input
        if xscale_range_split[0] != 0:
            first_xmin = np.min(df.lags)/2 if xscale == 'log' else 0
            xscale_range_split = [first_xmin] + xscale_range_split
        # Add maximum distance if not in input
        if xscale_range_split[-1] != np.max(df.lags):
            xscale_range_split.append(np.max(df.lags))

        # Scale grid size by the number of subpanels
        nb_subpanels = len(xscale_range_split)-1
        gridsize = init_gridsize.copy()
        gridsize[0] *= nb_subpanels
        # Create list of parameters to pass to ax/grid objects of subpanels
        xmin, xmax, xgridmin, xgridmax = ([] for _ in range(4))
        for i in range(nb_subpanels):
            xmin.append(xscale_range_split[i])
            xmax.append(xscale_range_split[i+1])
            xgridmin.append(init_gridsize[0]*i)
            xgridmax.append(init_gridsize[0]*(i+1))

    # Need a grid plot to show the sample count and the statistic
    grid = plt.GridSpec(gridsize[1], gridsize[0], wspace=0.5, hspace=0.5)

    # Loop over each subpanel
    for k in range(nb_subpanels):
        # First, an axis to plot the sample histogram
        ax0 = ax.inset_axes(grid[:3, xgridmin[k]:xgridmax[k]].get_position(fig).bounds)
        ax0.set_xscale(xscale)
        ax0.set_xticks([])

        # Plot the histogram manually with fill_between
        interval_var = [0] + list(df.lags)
        for i in range(len(df)):
            count = df['count'].values[i]
            ax0.fill_between([interval_var[i], interval_var[i+1]], [0] * 2, [count] * 2,
                             facecolor=plt.cm.Greys(0.75), alpha=1,
                             edgecolor='white', linewidth=0.5)
        if k == 0:
            ax0.set_ylabel('Sample count')
        # Scientific format to avoid undesired additional space on the label side
            ax0.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        else:
            ax0.set_yticks([])
        # Ignore warnings for log scales
        ax0.set_xlim((xmin[k], xmax[k]))

        # Now, plot the statistic of the data
        ax1 = ax.inset_axes(grid[3:, xgridmin[k]:xgridmax[k]].get_position(fig).bounds)

        # Get the lags bin centers
        bins_center = np.subtract(df.lags, np.diff([0] + df.lags.tolist()) / 2)

        # If all the estimated errors are all NaN (single run), simply plot the empirical variogram
        if np.all(np.isnan(df.err_exp)):
            ax1.scatter(bins_center, df.exp, label='Empirical variogram', color='blue', marker='x')
        # Otherwise, plot the error estimates through multiple runs
        else:
            ax1.errorbar(bins_center, df.exp, yerr=df.err_exp, label='Empirical variogram (1-sigma s.d)', fmt='x')

        # If a list of functions is passed, plot the modelled variograms
        if list_fit_fun is not None:
            for i, fit_fun in enumerate(list_fit_fun):
                x = np.linspace(xmin[k], xmax[k], 1000)
                y = fit_fun(x)

                if list_fit_fun_label is not None:
                    ax1.plot(x, y, linestyle='dashed', label=list_fit_fun_label[i], zorder=30)
                else:
                    ax1.plot(x, y, linestyle='dashed', color='black', zorder=30)

            if list_fit_fun_label is None:
                ax1.plot([],[],linestyle='dashed',color='black',label='Model fit')

        ax1.set_xscale(xscale)
        if nb_subpanels>1:
            if k == nb_subpanels - 1:
                ax1.xaxis.set_ticks(np.linspace(xmin[k], xmax[k], 3))
            else:
                ax1.xaxis.set_ticks(np.linspace(xmin[k],xmax[k],3)[:-1])

        if xlim is None:
            ax1.set_xlim((xmin[k], xmax[k]))
        else:
            ax1.set_xlim(xlim)

        if ylim is not None:
            ax1.set_ylim(ylim)
        elif np.all(np.isnan(df.err_exp)):
            ax1.set_ylim((0, 1.05*np.nanmax(df.exp)))
        else:
            ax1.set_ylim((0, np.nanmax(df.exp)+np.nanmean(df.err_exp)))

        if k == int(nb_subpanels/2):
            ax1.set_xlabel(xlabel)
        if k == nb_subpanels - 1:
            ax1.legend(loc='lower right')
        if k == 0:
            ax1.set_ylabel(ylabel)
        else:
            ax1.set_yticks([])

def df_clip_by(sf,poly_fid,crs=4326):
    '''
    used to clip a dataframe by a polyon (in gpkg or other format supported by gpd.read_file)

    df and poly in the same crs: 4326.
    df have columns [longitude,latitude]

    '''

    # poly_fid = r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\data\Norway_shapefile\border_no_wgs84.gpkg'
    import geopandas as gpd
    from shapely.geometry import Point

    # turn df into gpd
    # df have columns [longitude,latitude]
    if not isinstance(sf,gpd.GeoDataFrame):
        sf = gpd.GeoDataFrame(sf, geometry=[Point(xy) for xy in zip(sf.longitude, sf.latitude)],crs=crs)
    # read polygon
    study_no = gpd.read_file(poly_fid)

    # clip and return new gpd
    return sf.clip(study_no)

def yield_croped_dem_(dem,bonds_list):
    '''
    not use
    '''
    for bond in bonds_list:
        yield dem_crop(dem,bond)

def yield_croped_dem(dem,bonds_list):
    from rasterio.errors import WindowError
    dem_ = dem.copy()

    for xyxy_or_wkt in bonds_list:
        if isinstance(xyxy_or_wkt,tuple):
            xmin, ymin, xmax, ymax = xyxy_or_wkt
            crop_bbox = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
        if isinstance(xyxy_or_wkt,str):
            crop_bbox = shapely.wkt.loads(xyxy_or_wkt)
        
        [L,B,R,T] = dem.bounds
        xmin, ymin, xmax, ymax = crop_bbox.exterior.coords[0] + crop_bbox.exterior.coords[2]
        bbox = rio.coords.BoundingBox(left=xmin, bottom=ymin, right=xmax, top=ymax)
        if xmax < L or xmin > R or ymax < B or ymin > T:
            newraster = None
        else:
            try: 
                out_rst = dem_.reproject(dst_bounds=bbox)  # should we instead raise an issue and point to reproject?
                crop_img = out_rst.data
                tfm = out_rst.transform
                newraster = dem_.from_array(crop_img, tfm, dem_.crs, dem_.nodata)
                newraster.tags["AREA_OR_POINT"] = "Area"
            except (WindowError,ValueError) as err:
                print('Warning:', err)
                newraster = None
        yield newraster

def dem_crop(dem, xyxy_or_wkt):
    '''
    not use
    '''

    # If cropGeom is a Vector, crop() will crop to the bounding geometry. If cropGeom is a
    #        list of coordinates, the order is assumed to be [xmin, ymin, xmax, ymax].
    
    if isinstance(xyxy_or_wkt,tuple):
        xmin, ymin, xmax, ymax = xyxy_or_wkt
        crop_bbox = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    if isinstance(xyxy_or_wkt,str):
        crop_bbox = shapely.wkt.loads(xyxy_or_wkt)

    assert type(crop_bbox) == Polygon

    try:
        crop_img, tfm = rio.mask.mask(dem.data, [crop_bbox], crop=True, all_touched=True)
    except ValueError as err:
        print('Warning:', err)
        return None

    return dem.from_array(crop_img, tfm, dem.crs, dem.nodata)


def dem_shift(dem:DEM,e_px:int,n_px:int,bias:int=0) -> DEM:
    '''
    appying a matrix shift to a dem.

    input in px!

    '''
    import xdem
    func = xdem.coreg.NuthKaab()
    dem_new = dem.copy()

    func._meta["offset_east_px"] = e_px
    func._meta["offset_north_px"] = n_px
    func._meta["bias"] = bias
    func._meta["resolution"] = dem.res[0]
    func._meta["matrix"] =func._to_matrix_func()
    return func.apply(dem_new)

import xarray as xr
def gdf_shift(gdf:pd.DataFrame or xr.DataArray,ee:int,nn:int,bias:int=0,z_name='h_te_best_fit') -> pd.DataFrame:
    '''
    appying a matrix shift to a gdf.

    input in meters!
    
    Check the function apply_matrix of xDEM_pt, line 1867 in coreg.py. It use:
    - x_coords -= centroid[0]
    - y_coords -= centroid[1]
    - point_cloud[:, 2] -= centroid[2]

    So here I use +=, becasue I shift gdf.

    '''
    if isinstance(gdf,xr.DataArray):
        new_gdf = gdf.assign_coords({"x": (gdf.x + ee),'y': (gdf.y + nn)})
        if z_name in new_gdf:
            new_gdf[z_name] -= bias
    else:
        new_gdf = gdf.copy()

        new_gdf['E'] += ee
        new_gdf['N'] += nn
        new_gdf[z_name] -= bias

    return new_gdf

def compare_shift_gdf_dem(dem_10,sf_subset,shift_px,bias):
    #verification by shifting dem
    
    from xsnow.goplot import final_histogram
    from xsnow.godh import get_dh_by_shift_px_gdf,get_dh_by_shift_px_dem
    pts_gdf = get_dh_by_shift_px_gdf(dem_10,sf_subset,shift_px,shift_bias=bias,z_name='h_te_best_fit',footprint=False,stat=False)
    pts_dem = get_dh_by_shift_px_dem(dem_10,sf_subset,shift_px,shift_bias=bias,z_name='h_te_best_fit',footprint=False,stat=False)
    final_histogram(pts_gdf['dh'],pts_dem['dh'],legend=['shifting gdf','shifting dem'],range=(-3,3))

def df_sampling_from_dem(dem:DEM, dem_tba:DEM = None, inlier_mask= None, samples=5000,order=1,offset='ul') -> pd.DataFrame:
    '''
    generate a datafram from a dem by random sampling.

    :param offset: The pixel’s center is returned by default, but a corner can be returned by setting offset to one of ul, ur, ll, lr.
    
    :returns dataframe: N,E coordinates and z of DEM at sampling points.
    '''

    ref_dem, ref_mask = spatial_tools.get_array_and_mask(dem)
    if inlier_mask is not None:
        inlier_mask = np.asarray(inlier_mask).squeeze()
        assert inlier_mask.dtype == bool, f"Invalid mask dtype: '{inlier_mask.dtype}'. Expected 'bool'"
        full_mask = (~ref_mask & (np.asarray(inlier_mask) if inlier_mask is not None else True)).squeeze()
        dem.set_mask(~full_mask)

    # Avoid edge, and mask-out-area in sampling
    width,length = dem.shape
    i,j = np.random.randint(10,width-10,samples),np.random.randint(10, length-10, samples)
    mask = dem.data.mask[0]

    # Get value # TODO : when i != j , may raise error
    x, y = ij2xy(dem,i[~mask[i,j]], j[~mask[i,j]], offset=offset)
    z = ndimage.map_coordinates(dem.data[0, :, :], [i[~mask[i,j]], j[~mask[i,j]]],order=order,mode="nearest")
    df = pd.DataFrame({'z': z,'N':y,'E':x})

    # Maks out from tba_dem
    if dem_tba is not None:
        pts = np.array((df['E'].values,df['N'].values)).T
        final_mask = ~dem.data.mask
        mask_raster = dem.copy(new_array=final_mask.astype(np.float32))
        ref_inlier = mask_raster.interp_points(pts, input_latlon=False, order=0)
        df = df[ref_inlier.astype(bool)].copy()

    return df

def test_ij_xy(dem,offset='ul',area_or_point='Area'):
    """ test ij to xy and xy to ij """
    i = [0,0.6,20.1]
    j = [0,0.7,10.1]
    print('(1) input i,j:',i,j)
    h_ij = ndimage.map_coordinates(dem.data[0, :, :], [i, j],order=1,mode="nearest")
    print('get h at i,j:',h_ij)
    xy = ij2xy(dem,i,j,offset=offset)
    print('x,y at i,j:',xy)
    print('---------verification:i,j---------')
    new_i,new_j = xy2ij(dem,xy[0], xy[1],area_or_point=area_or_point)
    print('(2) new i,j:',new_i,new_j)
    new_xy = ij2xy(dem, new_i,new_j, offset=offset)
    print('new xy:', new_xy)
    print('---------verification h at xy---------')
    print('new h at x,y:', interp_points(dem,zip(new_xy[0],new_xy[1]),area_or_point=area_or_point))

def test_ij_xy_dem(dem,offset='ul',area_or_point='Area'):
    """ test ij to xy and xy to ij """
    i = [0,0.6,20.1]
    j = [0,0.7,10.1]
    print('(1) input i,j:',i,j)
    h_ij = ndimage.map_coordinates(dem.data[0, :, :], [i, j], order=1,mode="nearest")
    print('get h at i,j:',h_ij)
    xy = dem.ij2xy(i,j,offset=offset)
    print('x,y at i,j:',xy)
    print('---------verification:i,j---------')
    new_i,new_j = dem.xy2ij(xy[0], xy[1], area_or_point=area_or_point)
    print('(2) new i,j:',new_i,new_j)
    new_xy = dem.ij2xy(new_i,new_j, offset=offset)
    print('new xy:', new_xy)
    print('---------verification h at xy---------')
    print('new h at x,y:', interp_points(dem,zip(new_xy[0],new_xy[1]),area_or_point=area_or_point))

def test_slow_fn(args):
    """ Simulated an optimisation problem with args coming in
    and function value being output, for testing multiprocessing """
    n = 10000
    y = 0
    for j in range(n):
        j = j / n
        for i, p in enumerate(args):
            y += j * (p ** (i + 1))
    return y / n

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr,ks_2samp

def calculate_weights(df, cell_size, window_size, month=False,w_name='weight'):
    '''
    a spatial density-based weighting system.

    The use of logarithmic transformation in weight calculation (1 / np.log10(density+1)) 
    is a non-linear transformation, to turn the density into more normally distributed values.
    '''
    # Discretize coordinates
    df['N_grid'] = (df['N'] // cell_size).astype(int)
    df['E_grid'] = (df['E'] // cell_size).astype(int)

    # Backup original coordinates
    df_original = df.copy()

    if 'weight' in df_original.columns:
        df_original.drop(columns=['weight'], inplace=True)

    if month:
        # Count occurrences for each combination of Month, N_grid, and E_grid
        count_df = df.groupby(['month', 'N_grid', 'E_grid']).size().reset_index(name='count')
        # Convert count DataFrame to xarray Dataset
        ds = xr.Dataset.from_dataframe(count_df.set_index(['month', 'N_grid', 'E_grid']))
        # fill nan
        ds['count'] = ds['count'].fillna(0)
        # Apply rolling window for density calculation
        density = ds['count'].rolling(month=1, N_grid=window_size, E_grid=window_size, min_periods=1,center=True).mean()

    else:
        count_df = df.groupby(['N_grid', 'E_grid']).size().reset_index(name='count')
        ds = xr.Dataset.from_dataframe(count_df.set_index(['N_grid', 'E_grid']))
        ds['count'] = ds['count'].fillna(0)
        density = ds['count'].rolling(N_grid=window_size, E_grid=window_size, min_periods=1, center=True).mean()

    # Convert density to z-score
    #density_z_score = (density - density.mean()) / density.std()
    # Convert z-scores to quantiles
    weights = 1 / (xr.apply_ufunc(np.log1p, density)+1)
    # Use quantiles as weights
    #weights = (density.max() - density)/(density.max()-0)

    # Map weights back to DataFrame
    weights_df = weights.to_dataframe(name=w_name).reset_index()
    merged_df = df_original.merge(weights_df, on=['N_grid', 'E_grid'] + (['month'] if month else []), how='left')

    return merged_df

def evaluate_difference(dataset1, dataset2):
    # Filter out NaN values
    valid_mask = ~np.isnan(dataset1) & ~np.isnan(dataset2)
    filtered_dataset1 = dataset1[valid_mask]
    filtered_dataset2 = dataset2[valid_mask]
    # Calculate the Spearman's correlation coefficient
    correlation, _ = spearmanr(filtered_dataset1, filtered_dataset2)
    ksd, _ = ks_2samp(filtered_dataset1, filtered_dataset2)

    # Calculate the R-squared
    r2 = r2_score(filtered_dataset1, filtered_dataset2)

    # Calculate the RMSE
    rmse = np.sqrt(mean_squared_error(filtered_dataset1, filtered_dataset2))


    return correlation, ksd, r2, rmse

def temporal_statistic(array1, array2, time_1, time_2):
    # Convert time arrays to a common compatible type (numpy datetime64)
    time_1 = np.asarray(time_1, dtype='datetime64')
    time_2 = np.asarray(time_2, dtype='datetime64')

    # Find the intersection of time arrays
    intersection_index = np.intersect1d(time_1, time_2)

    df1 = pd.DataFrame(array1,index=time_1)
    df2 = pd.DataFrame(array2,index=time_2)

    # Filter df1 and df2 based on the intersection of time arrays
    df1_aligned = df1[df1.index.isin(time_1) & df1.index.isin(time_2)]
    df2_aligned = df2[df2.index.isin(time_1) & df2.index.isin(time_2)]
    
    # Filter df1 and df2 based on the intersection of time arrays
    #df1_aligned = df1[np.isin(time_1, intersection_index, assume_unique=True)]
    #df2_aligned = df2[np.isin(time_2, intersection_index, assume_unique=True)]

    # Calculate the metrics
    mse = mean_squared_error(df1_aligned, df2_aligned)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(df1_aligned, df2_aligned)
    r2 = r2_score(df1_aligned, df2_aligned)

    nmad = xdem.spatialstats.nmad(df1_aligned - df2_aligned)

    # Calculate the Spearman's correlation coefficient
    spearman_r, spearman_p = spearmanr(df1_aligned, df2_aligned)

    # Calculate the Pearson's correlation coefficient
    pearson_r, pearson_p = pearsonr(df1_aligned.squeeze(), df2_aligned.squeeze())

    # Calculate the KS statistic
    ks_statistic, _ = ks_2samp(df1_aligned.squeeze(), df2_aligned.squeeze())

    dict_ = {'n':len(df1_aligned.squeeze()),'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'nmad': nmad, 'spearman_r': spearman_r,
            'spearman_p': spearman_p, 'pearson_r': pearson_r, 'pearson_p': pearson_p,'ks':ks_statistic,'df_1':df1_aligned,'df_2':df2_aligned}
    
    # Return the metrics
    return dict_

