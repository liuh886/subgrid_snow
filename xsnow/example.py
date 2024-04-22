from xsnow.goregression import regression_task, ERA5
from xsnow.goplot import plot_point_map
from xsnow.misc import ij2xy, interp_points
from scipy import ndimage
import pyproj
import numpy as np

# Regression
def snow_depth_regression_and_plot(new_df_era,list_regressor,range=(0,10),clim=(0,3)):

    regression_task(new_df_era, list_regressor)

    new_df_era['df_dtm1_era'] = new_df_era['sd_predict_dtm1'] - new_df_era['sd_era']

    snow_dtm1 = plot_point_map('sd_predict_dtm1',new_df_era.query(f'{range[0]} < sd_predict_dtm1 < {range[1]}'),title='Snow depth (DTM1)',clim=clim,cmap='YlGnBu',sampling=0.01)
    snow_dtm10 = plot_point_map('sd_predict_dtm10',new_df_era.query(f'{range[0]} < sd_predict_dtm10 < {range[1]}'),title='Snow depth (DTM10)',clim=clim,cmap='YlGnBu',sampling=0.01)
    snow_cop30 = plot_point_map('sd_predict_cop30',new_df_era.query(f'{range[0]} < sd_predict_cop30 < {range[1]}'),title='Snow depth (GLO30)',clim=clim,cmap='YlGnBu',sampling=0.01)
    snow_fab = plot_point_map('sd_predict_fab',new_df_era.query(f'{range[0]} < sd_predict_fab < {range[1]}'),title='Snow depth (FAB)',clim=clim,cmap='YlGnBu',sampling=0.01)
    snow_era5 = plot_point_map('sd_era',new_df_era.query(f'{range[0]} < sd_era < {range[1]}'),title='Snow depth ERA5 Land',clim=clim,cmap='YlGnBu',sampling=0.01)
    snow_df = plot_point_map('df_dtm1_era',new_df_era.query(f'{range[0]} < sd_predict_dtm1 < {range[1]}'),title='Snow depth difference (DTM1 - ERA5 Land)',clim=(-1.5,1.5),cmap='bwr',sampling=0.01)

    return ( snow_dtm10 + snow_dtm1 + snow_era5 + snow_cop30 + snow_fab + snow_df).cols(3),new_df_era

# Produce points

def produce_pts_2022(new_df,date='2022-03-01'):
    '''
    Produce the points for 2022 based on template
    '''

    # template
    if new_df is None:
        new_df = pd.read_csv('new_df.csv')
        new_df['date']='20220301'
        new_df['h_te_best_fit'] = new_df['z']

    # Coulpe daily and monthly ERA5 data
    era_monthly = r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\data\EAR5_land\monthly_data_08_22.nc'
    era = ERA5(era_monthly)
    era.cal_wind_aspect_factor_yearly()
    new_df = era.coupling_dataframe_single_date(new_df,date=date)
    return new_df

def df_from_dem(dem,dem_mask=None, pixel_interval=1,
                v_x=['slope', 'aspect', 'planform_curvature','profile_curvature','curvature','topographic_position_index'],
                order=1,offset='ul'):
    
    '''

    Create a dataframe from dem

    params:
        dem: xdem.DEM
        dem_mask: xdem.DEM
        pixel_interval: int
        v_x: list
        order: int
        offset: str

    Return a DataFrame
    '''
    
    # dem 
    # dem_mask

    # Avoid edge, and mask-out-area in sampling
    width,length = dem.shape 
    i,j = np.meshgrid(np.arange(0,width,pixel_interval),np.arange(0,length,pixel_interval))
    i,j  = i.flatten(),j.flatten()
    mask = dem.data.mask[0]

    # Get value
    x, y = ij2xy(dem,i[~mask[i,j]], j[~mask[i,j]], offset=offset)
    z = ndimage.map_coordinates(dem.data[0, :, :], [i[~mask[i,j]], j[~mask[i,j]]],order=order,mode="nearest")
    
    pts = {}
    if v_x is not None:
        dem_attr = xdem.terrain.get_terrain_attribute(dem, attribute=v_x)
        attr_ = [interp_points(dem_,zip(x,y),input_latlon=False, order=order) for dem_ in dem_attr]
        # pass the results to return_dict
        for i,j in zip(v_x,attr_):
            pts[i] = j

    init_crs = pyproj.CRS(dem.crs)
    dest_crs = pyproj.CRS(4326)
    transformer = pyproj.Transformer.from_crs(init_crs, dest_crs)
    latitude, longitude = transformer.transform(x, y)

    df = pd.DataFrame({'z':z,
                       'N':y,
                       'E':x,
                       'latitude':latitude,
                       'longitude':longitude,
                       'slope':pts['slope'], 
                       'aspect':pts['aspect'], 
                       'planc':pts['planform_curvature'],
                       'profc':pts['profile_curvature'],
                       'curvature':pts['curvature'],
                       'tpi':pts['topographic_position_index']
                       })
    
    if dem_mask is not None:
        pts = np.array((df['E'].values,df['N'].values)).T
        final_mask = ~dem_mask.data.mask
        mask_raster = dem_mask.copy(new_array=final_mask.astype(np.float32))
        ref_inlier = mask_raster.interp_points(pts, input_latlon=False, order=0)
        df = df[ref_inlier.astype(bool)].copy()

    return df

def get_tpi(dem,df):

    '''
    Get tpi_9 and tpi_27 from dem

    '''

    tpi_9 = xdem.terrain.get_terrain_attribute(dem, attribute=['topographic_position_index'], window_size=9)
    tpi_27 = xdem.terrain.get_terrain_attribute(dem, attribute=['topographic_position_index'], window_size=27)

    # interpolate to get value
    x_coords, y_coords = (df['E'].values,df['N'].values)
    pts = np.array((x_coords, y_coords)).T

    attr_value = [interp_points(i,pts,input_latlon=False, order=1) for i in [tpi_9,tpi_27]]
        # pass the results to return_dict
    for i,j in zip(['tpi_9','tpi_27'],attr_value):
        df[i] = j
    return df

def produce_dataframe_from_dem(dem):

    '''
    Example
    '''

    new_df = df_from_dem(dem)
    new_df = get_tpi(dem,new_df)
    new_df.to_csv('datafame_from_dem.csv',index=False)

