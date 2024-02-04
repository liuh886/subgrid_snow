from msilib.schema import Class
import xarray as xr
import pandas as pd
from datetime import datetime,date, timedelta
import numpy as np
from tqdm import tqdm
import xgboost as xgb
from xsnow.goregression import encoding_x_y, regression_xgboost
from xsnow.goregression import plot_df_era_scatter
import matplotlib.pyplot as plt
from xsnow.goplot import final_histogram
from xsnow.goregression import regression_task,check_by_finse_lidar,plot_scater_nve,plot_point_map
from xsnow.goregression import encoding_x_y,regression_xgboost,produce_validation_from_nve
import xdem

class Snow_Regressor:
    def __init__(self, filename_sf,filename_sc,filename_pts=None,sep=','):
        """
        Load a dataframe.
        """
        self.sf = pd.read_csv(filename_sf,sep=sep)
        self.sc = pd.read_csv(filename_sc,sep=sep)

        self.filename_sc = filename_sc
        self.filename_sf = filename_sf

        if filename_pts:
            self.pts = pd.read_csv(filename_sc)
        self.params = {
                    'objective': 'reg:absoluteerror',  # reg:squarederror absoluteerror
                    'max_depth': 10,
                    'learning_rate': 0.1,
                    'n_estimators': 250,
                    'min_child_weight': 1,
                    'subsample': 0.7,
                    'colsample_bytree': 1,
                    'gamma': 0.1,
                }
        self.params_2 = {
                    'objective': 'reg:absoluteerror',
                    'max_depth': 10,
                    'learning_rate': 0.1,
                    'n_estimators': 250,
                    'min_child_weight': 1,
                    'subsample': 1,
                    'colsample_bytree': 0.7,
                    'gamma': 0.1,
                }

        self.sf_qc = 'abs(dh_after_dtm1) < 10 and brightness_flag == 0'
        self.sc_qc = '(0 < sd_correct_dtm1 < 10) and subset_te_flag == 5 and (0 < sd_era < 10)'

        # the basis of calculating elevation difference
        self.z = 'h_te_best_fit'

    def save(self,filename, cover=False):
        if (filename is None) and cover:
            filename = self.filename
        self.df.to_scv(filename,index=False)

    def drop_columns(self):

        drop_list_sc = ['latitude_20m_0', 'latitude_20m_1',
                        'latitude_20m_2', 'latitude_20m_3', 'latitude_20m_4', 'longitude_20m_0',
                        'longitude_20m_1', 'longitude_20m_2', 'longitude_20m_3',
                        'longitude_20m_4', 'h_te_best_fit_20m_0',
                        'h_te_best_fit_20m_1', 'h_te_best_fit_20m_2', 'h_te_best_fit_20m_3',
                        'h_te_best_fit_20m_4', 'z',
                        'coreg_bias_dtm10', 'fid_dtm10', 'coreg_offset_east_px_dtm10',
                        'coreg_offset_north_px_dtm10', 'coreg_bias_cop30', 'fid_cop30',
                        'coreg_offset_east_px_cop30', 'coreg_offset_north_px_cop30','coreg_bias_dtm1', 'fid_dtm1',
                        'coreg_offset_east_px_dtm1', 'coreg_offset_north_px_dtm1', 'coreg_bias_fab', 'fid_fab',
                        'coreg_offset_east_px_fab', 'coreg_offset_north_px_fab']
        cols_to_drop = list(set(self.sc.columns).intersection(set(drop_list_sc)))
        self.sc.drop(cols_to_drop, axis=1, inplace=True)

        drop_list_sf = ['latitude_20m_0', 'latitude_20m_1',
                        'latitude_20m_2', 'latitude_20m_3', 'latitude_20m_4', 'longitude_20m_0',
                        'longitude_20m_1', 'longitude_20m_2', 'longitude_20m_3',
                        'longitude_20m_4', 'h_te_best_fit_20m_0',
                        'h_te_best_fit_20m_1', 'h_te_best_fit_20m_2', 'h_te_best_fit_20m_3',
                        'h_te_best_fit_20m_4', 'z', 'fid', 'fid_dtm10', 'dh_before_cop30',
                        'coreg_bias_cop30', 'fid_cop30', 'dh_before_dtm1',
                        'coreg_bias_dtm1', 'fid_dtm1', 'coreg_bias_fab', 'fid_fab']
        cols_to_drop = list(set(self.sf.columns).intersection(set(drop_list_sf)))
        self.sf.drop(cols_to_drop, axis=1, inplace=True)

    def use_interp(self):
        '''
        the original difference is calculated by 'h_te_best_fit'. Applying this function can change it into 'h_te_interp'.
        '''
        for df in [self.sc,self.sf]:
            if 'dh_after_dtm1' in df:
                dh_list = ['dh_after_dtm1','dh_after_dtm10','dh_after_cop30','dh_after_fab']
            elif 'snowdepth_dtm1' in df:
                dh_list = ['snowdepth_dtm1','snowdepth_dtm10','snowdepth_cop30','snowdepth_fab']

            if self.z == 'h_te_best_fit':
                for dh in dh_list:
                    df[dh] = df[dh] + df['h_te_interp'] - df['h_te_best_fit']
                # update information
                self.z = 'h_te_interp'
    
    def use_best_fit(self):
        '''
        reverse use_interp
        '''
        for df in [self.sc,self.sf]:
            if 'dh_after_dtm1' in df:
                dh_list = ['dh_after_dtm1','dh_after_dtm10','dh_after_cop30','dh_after_fab']
            elif 'snowdepth_dtm1' in df:
                dh_list = ['snowdepth_dtm1','snowdepth_dtm10','snowdepth_cop30','snowdepth_fab']

            if self.z == 'h_te_interp':
                for dh in dh_list:
                    df[dh] = df[dh] + df['h_te_interp'] - df['h_te_best_fit']
                # update information
                self.z = 'h_te_best_fit'

    def correction_validation_dtm1(self,
                                   regressor=None,
                                   v_x=None,
                                   v_y=None,
                                   regressor_name=None,
                                   era='sd_era'):
        if v_x is None:
            v_x=['h_te_std','n_te_photons','subset_te_flag','E', 'N','h_te_best_fit','beam','pair','slope', 'aspect', 'planc','profc','tpi','tpi_9','tpi_27','curvature','segment_cover','h_mean_canopy','canopy_openness', 'urban_flag', 'night_flag','region','h_te_skew','h_te_uncertainty']
        if v_y is None:
            v_y=['dh_after_dtm1']

        # prepare regrressor
        if regressor:
            xg_reg_dtm1 = xgb.Booster()
            xg_reg_dtm1.load_model(regressor)
        else:
            assert regressor_name is True
            # need to train
            X_encoded,y = encoding_x_y(self.sf.query(self.sf_qc),
                             target_col=v_y,
                             feature_col=v_x)
            params = self.params
            xg_reg_dtm1 = regression_xgboost(X_encoded,y[v_y[0]],**params)
            xg_reg_dtm1.save_model(regressor_name)

        # prepare x
        X_encoded,y = encoding_x_y(self.sc,
                                    target_col=['snowdepth_dtm1'],
                                    feature_col=v_x)

        # get corrected snow depth
        self.sc['sd_correct_dtm1'] = self.sc['snowdepth_dtm1'] - xg_reg_dtm1.predict(xgb.DMatrix(X_encoded)) #xgb.DMatrix(X_encoded)
        # generate the difference with ERA5
        self.sc['df_dtm1_era5'] = self.sc['sd_correct_dtm1'] - self.sc[era]
        # generate plots
        self.plot_hist_scatter_era(era=era)
        p1 = self.plot_validation_lidar()
        return p1

    def validation_lidar(self):
        df_sub = check_by_finse_lidar(self.sc,shift_px=(-0.5,1.128),dst_res = (10,10))
        
        #plot_correction_comparision(df_sub, 'sd_correct_dtm1', 'snowdepth_dtm1')

        dtm_1_p =plot_scater_nve(df_sub.query('6 > sd_correct_dtm1 > 0'),'sd_lidar','sd_correct_dtm1',title='correct_dtm1')
        dtm_1_p_raw =plot_scater_nve(df_sub.query('6 > snowdepth_dtm1 > 0'),'sd_lidar','snowdepth_dtm1',title='raw_dtm1')
        p1 = (dtm_1_p + dtm_1_p_raw).cols(2)
        return p1

    def plot_hist_scatter_era(self,era='sd_era'):
        '''
        plot scatter and hist to era5 snow depth
        '''

        # scatter
        fig,ax = plt.subplots(1,2,figsize=(14,5))
        plot_df_era_scatter(self.sc.query(f'(0 < {era} < 6) and subset_te_flag == 5'),x=era,y='sd_correct_dtm1',ax=ax[0],vmax=3000,vmin=0,x_lims=(0,6),y_lims=(-3,6),cmap=plt.cm.magma,n_quantiles=False)
        
        # hist
        df_era = self.sc.query(f'0 < {era} < 10')
        df_dtm1 = self.sc.query('0 < sd_correct_dtm1 < 10')
        final_histogram(df_era[era], df_dtm1['sd_correct_dtm1'], dH_ref=df_dtm1['df_dtm1_era5'], ax=ax[1],legend=['ERA5 Land','ICESat-2 - DTM1', 'ICESat-2(DTM1) - ERA5 Land'],range=(-4,8),perc_t=100);
        plt.show()

    def validation_nve(self,
                       regressor=None,
                       v_x=None,
                       v_y=None,
                       regressor_name=None,
                       df_raw_validation_list=None,
                       era='sd_era'):
        # v_y
        self.sc['df_dtm1_era5'] = self.sc['sd_correct_dtm1'] - self.sc[era]
        self.sc.loc[self.sc[era] < 0.1, ['df_dtm1_era5']] = 0

        if v_x is None:
            v_x=['E', 'N','h_te_best_fit','slope', 'aspect', 'planc','profc','tpi','tpi_9','tpi_27','curvature','segment_cover','h_mean_canopy','canopy_openness','sd_era','wf_positive', 'wf_negative','month','smlt_acc', 'sf_acc']
        if v_y is None:
            v_y=['df_dtm1_era5']

        # regressor
        if regressor is None:
            assert regressor_name is True
            # need to train
            params = self.params_2
            # preprare data
            X_encoded,y = encoding_x_y(self.sc.query(self.sc_qc),
                                        target_col=v_y,
                                        feature_col=v_x)
            sd_reg_dtm1 = regression_xgboost(X_encoded,y[v_y[0]],**params)
            sd_reg_dtm1.save_model(regressor_name)    # train
        else:
            sd_reg_dtm1 = xgb.Booster()
            sd_reg_dtm1.load_model(regressor)
        
        # read validation df
        if df_raw_validation_list:
            df_nve_08,df_nve_09 = self.generate_validation_df_nve()
        else: # load    
            df_nve_08 = pd.read_csv('df_nve_08.csv')
            df_nve_09 = pd.read_csv('df_nve_09.csv')

        # predict and validate
        p_dtm1_08 = list(self.update_raw_and_validate([sd_reg_dtm1],df_nve_08))
        p_dtm1_09 = list(self.update_raw_and_validate([sd_reg_dtm1],df_nve_09))

        return p_dtm1_08,p_dtm1_09

    def generate_validation_df_nve(self):
        
        # generate validation df
        df_era_08 = pd.read_csv(r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\snowdepth_national_20080401.csv')
        sd_nve_08 = xdem.DEM(r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\data\nve_08_merge_m.tif')
        df_nve_08 = produce_validation_from_nve(df_era_08,sd_nve_08)
        df_nve_08.to_csv('df_nve_08.csv',index=False)

        df_era_09 = pd.read_csv(r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\snowdepth_national_20090401.csv')
        sd_nve_09 = xdem.DEM(r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\data\nve_09_merge_m.tif')
        df_nve_09 = produce_validation_from_nve(df_era_09,sd_nve_09)
        df_nve_09.to_csv('df_nve_09.csv',index=False)

        return sd_nve_08,df_nve_09

    def regression_task(self, list_regressor, v_x = None, list_df_name = None, list_sd_name = None,era='sd_era'):
    
        '''
        This function run on self.sc

        4 models can run together.
        1. regresion
        2. generate sd_predict
        '''
        if list_df_name is None:
            list_df_name = ['df_predict_dtm10', 'df_predict_dtm1', 'df_predict_cop30', 'df_predict_fab']

        if list_sd_name is None:
            list_sd_name = ['sd_predict_dtm10', 'sd_predict_dtm1', 'sd_predict_cop30', 'sd_predict_fab']

        if v_x is None:
            v_x=['E', 'N','h_te_best_fit','slope', 'aspect', 'planc','profc','tpi','tpi_9','tpi_27','curvature','segment_cover','h_mean_canopy','canopy_openness','sd_era','wf_positive', 'wf_negative','month','smlt_acc', 'sf_acc']

        self.sc['month'] = pd.DatetimeIndex(self.sc['date']).month

        X_encoded,y = encoding_x_y(self.sc.query(self.sc_qc),
                                     target_col=['E'],
                                     feature_col=v_x)

        # append the result to list, order: dtm10 dtm1 cop30 fab 
        self.sc[list_df_name] = pd.Series(i.predict(X_encoded) for i in list_regressor)
        self.sc[list_sd_name] = pd.Series(self.sc[i] + self.sc[era] for i in list_df_name)

    def update_raw_and_validate(self,
                                regressor_list,
                                df_raw_validation, 
                                list_df_name = ['df_predict_dtm1'],
                                list_sd_name = ['sd_predict_dtm1'],
                                nve_10 = False,
                                era='sd_era'):
        '''
        For fast test

        Plot difference map with ERA
        Validate with NVE Validation
        '''
        # predict & validtion
        df = df_raw_validation

        df_predicted = regression_task(df,regressor_list, list_df_name =list_df_name,list_sd_name =list_sd_name,era=era)
            
        df_predicted['df_dtm1_era'] = df_predicted['sd_predict_dtm1'] - df_predicted[era]
            
        era_p =plot_scater_nve(df_predicted.query('sd_nve > 0.05'), 'sd_nve', era)
        dtm_1_p =plot_scater_nve(df_predicted.query('sd_nve > 0.05'), 'sd_nve','sd_predict_dtm1',title='2 m NVE')

        if nve_10:
            era_p_10 =plot_scater_nve(df_predicted.query('sd_nve_10 > 0.05'), 'sd_nve_10', era, title='10 m NVE')
            dtm_1_p_10 =plot_scater_nve(df_predicted.query('sd_nve_10 > 0.05'), 'sd_nve_10','sd_predict_dtm1',title='10 m NVE')
        else:
            era_p_10 = None
            dtm_1_p_10= None

        return (era_p + era_p_10+dtm_1_p+dtm_1_p_10).cols(4)

    def update_raw_and_plot(regressor_list,
                            df_raw,
                            list_df_name = ['df_predict_dtm1'],
                            list_sd_name = ['sd_predict_dtm1'],
                            range=(0,10),clim=(0,3),
                            era='sd_era'):

        '''
        same with update_raw_and_validate, but plot it
        '''
        # predict
        df = df_raw
        df_predicted = regression_task(df, regressor_list, list_df_name = list_df_name,list_sd_name = list_sd_name,era=era)
        df_predicted['df_dtm1_era'] = df_predicted['sd_predict_dtm1'] - df_predicted['sd_era']
        
        #final_histogram(df_predicted['sd_predict_dtm1'],df_predicted['sd_era'],dH_ref=df_predicted['df_dtm1_era'],legend=['ICESat-2 - DTM1','ERA5 Land','ICESat-2(DTM1) - ERA5 Land'],range=(-4,8),perc_t=100);
        snow_dtm1 = plot_point_map('sd_predict_dtm1',df_predicted.query(f'{range[0]} < sd_predict_dtm1 < {range[1]}'),title='Snow depth (DTM1)',clim=(clim),cmap='YlGnBu',sampling=0.01)
        snow_df = plot_point_map('df_dtm1_era',df_predicted.query(f'{range[0]} < sd_predict_dtm1 < {range[1]}'),title='Snow depth difference (DTM1 - ERA5 Land)',clim=(-1.5,1.5),cmap='bwr',sampling=0.01)
            
        return (snow_dtm1 + snow_df).cols(2)

    def update_sc(self, era_daily=None,era_montly=None):
        '''
        update sde and other predictors from daily and monthly ds for datraframe snowcover 
        '''
        if era_daily is None:
            era_daily = r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\data\EAR5_land\era5.nc'
        # update sd_era
        era = ERA5(era_daily)
        self.sc = era.coupling_dataframe_sde_by_date(self.sc)

        if era_montly is None:
            era_montly = r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\data\EAR5_land\monthly_data_08_22.nc'
        # update others
        era = ERA5(era_montly)
        era.cal_wind_aspect_factor_yearly()
        era.coupling_dataframe_other_by_date(self.sc)


class ERA5:
    def __init__(self, filename):
        """
        Load an xarray dataset from a NetCDF file.
        """
        self.ds = xr.open_dataset(filename)
        self.wf = False
        self.time_list= ['2019-04-01', '2020-04-01', '2021-04-01',              
             '2022-01-01', '2022-02-01', '2022-03-01',              
             '2022-04-01', '2022-05-01']

    def coupling_dataframe_sde_by_date(self, df, interp_method='linear',target='sde',new_column='sde_era', kwargs=None, **coords_kwargs):
        '''
        couple with dataframe by date. ONLY one variable 'sde' at daily resolution.
        '''
        
        # Group the DataFrame by date
        df['date_'] = pd.to_datetime(df['date'], format=ymd).copy()
        grouped_df = df.groupby('date_')

        for date, subset_df in tqdm(grouped_df):
            # Interpolate the data spatially using xarray
            ds = self.ds.sel(time=date,expver=1,method='nearest')

            interpolated_values = ds.interp(latitude=subset_df.latitude.to_xarray(),longitude=subset_df.longitude.to_xarray(), method=interp_method,kwargs=None, **coords_kwargs)
            df.loc[df['date'] == date, new_column] = interpolated_values[target].values

        return df

    def coupling_dataframe_other_by_date(self, df, target = None, new_column = None):
        
        '''
        couple with dataframe by date. Without interpolation, At montly resolution.
        '''
        
        if target is None:
            target = ['sde','wf_positive','wf_negative','smlt_acc','sf_acc']
        if new_column is None:
            new_column = ['sd_era','wf_positive','wf_negative','smlt_acc','sf_acc']
        if self.wf:
            # Group the DataFrame by date
            df['date_offset'] = pd.to_datetime(df['date']) - pd.Timedelta(days=15)

            grouped_df = df.groupby('date_offset')
            for date, subset_df in tqdm(grouped_df):
                # Interpolate the data spatially using xarray

                ds = self.ds.sel(time=date,method='nearest')
                interpolated_values = ds.sel(latitude=subset_df.latitude.to_xarray(),longitude=subset_df.longitude.to_xarray(),aspect=df.aspect.to_xarray(), method='nearest')
                df.loc[df['date'] == date, new_column] = interpolated_values[target].to_array().values

            return df

    def coupling_dataframe(self, df, date_list=None):
        '''
        couple with dataframe. Without interpolation, At montly resolution. All date list.
        '''
        if date_list is None:
            date_list = self.time_list

        if self.wf:
            ds = self.ds[['sde','wf_positive','wf_negative','smlt_acc','sf_acc']]
            grouped_df = df.groupby('aspect')
            # turn latitude and longitude dimentions to index
            new_dfs = []  # create an empty list to hold new_df from each iteration of the loop
            for aspect, subset_df in tqdm(grouped_df):
                ds_ = ds.sel(latitude=subset_df.latitude.to_xarray(), longitude=subset_df.longitude.to_xarray(),aspect=subset_df.aspect.to_xarray(),method='nearest')
                # get values
                new_df = ds_.sel(time=date_list, method='nearest').to_dataframe()        
                new_dfs.append(new_df)  # append new_df to the list
            final_df = pd.concat(new_dfs)  # concatenate all the new_dfs into one final DataFrame
            return final_df

    def coupling_dataframe_single_date(self, df, date,interp_method='linear'):
        '''
        couple with dataframe, interpolate sde.
        '''
        
        if self.wf:
            # confirm the time dimention
            ds = self.ds.sel(time=date, method='nearest')
            
            grouped_df = df.groupby('date')

            for date_, subset_df in tqdm(grouped_df):

                # get values directly
                ds_ = ds.sel(latitude=subset_df.latitude.to_xarray(), longitude=subset_df.longitude.to_xarray(),aspect=subset_df.aspect.to_xarray(),method='nearest')
                df.loc[df['date'] == date_, ['sd_era','wf_positive','wf_negative','smlt_acc','sf_acc']] = ds_[['sde','wf_positive','wf_negative','smlt_acc','sf_acc']].to_array().values
                
                # get values interpolated
                interpolated_values = ds.isel(aspect=1)[['sde']].interp(latitude=subset_df.latitude.to_xarray(), longitude=subset_df.longitude.to_xarray(),method=interp_method)
                df.loc[df['date'] == date_, ['sde_era']] = interpolated_values['sde'].values

            return df
    

    def cal_wind_aspect_factor_yearly(self,snow_depth_threshold=0.1):
        '''
        calculate wind-aspect-factor.
        '''

        self.ds = self.ds.sel(expver=1)

        self.ds['wspd'] = np.sqrt(self.ds.u10 **2 + self.ds.v10 **2)
        self.ds['wd']=np.mod(180+np.rad2deg(np.arctan2(self.ds.u10, self.ds.v10)),360)
        # create new dimension 'aspect'
        aspect = np.arange(0, 360, 11)
        self.ds = self.ds.assign_coords(aspect=aspect)

        # calculate new variables
        cos_diff = - np.cos(np.deg2rad(self.ds['aspect'] - self.ds['wd']))

        # add new variables to the dataset
        self.ds = self.ds.assign(wind_aspect_factor=cos_diff)
        
        # Create a mask based on the sde condition
        #mask = self['sde'] > 0.1

        # Create a new variable to store the accumulative values
        self.ds = self.ds.assign(wf_positive=xr.zeros_like(self.ds['wind_aspect_factor']), 
                            wf_negative=xr.zeros_like(self.ds['wind_aspect_factor']),
                            wfc=xr.zeros_like(self.ds['sde']),
                            sf_acc=xr.zeros_like(self.ds['sde']),
                            smlt_acc=xr.zeros_like(self.ds['sde']))

        # Loop through each time step
        for t in range(len(self.ds['time'])):
            # Get the year and month of the current time step
            year = self.ds['time.year'][t].item()
            month = self.ds['time.month'][t].item()

            # Check if the current time step is within the accumulative period
            if month != 9 and t !=0:

                # Calculate the accumulative values if the sde condition is met
                wf_factor = self.ds['wind_aspect_factor'].isel(time=t)
                wspd = self.ds['wspd'].isel(time=t)

                sf = self.ds['sf'].isel(time=t)
                smlt = self.ds['smlt'].isel(time=t)

                #mask_t = xr.broadcast(mask.isel(time=t), wf_factor)[0]

                d_wf = wf_factor * (wspd ** 3)

                # Update the values in the dataset
                wf_positive_update = xr.where((wf_factor > 0) & (self.ds['sde'].isel(time=t) > snow_depth_threshold),d_wf, 0)
                self.ds['wf_positive'].loc[dict(time=str(year)+'-'+str(month)+'-01')] = wf_positive_update + self.ds['wf_positive'].isel(time=t-1)
                
                wf_negative_update = xr.where((wf_factor < 0) & (self.ds['sde'].isel(time=t) > snow_depth_threshold), d_wf, 0)
                self.ds['wf_negative'].loc[dict(time=str(year)+'-'+str(month)+'-01')] = wf_negative_update + self.ds['wf_negative'].isel(time=t-1)

                self.ds['sf_acc'].loc[dict(time=str(year)+'-'+str(month)+'-01')] = sf*30 + self.ds['sf_acc'].isel(time=t-1)
                self.ds['smlt_acc'].loc[dict(time=str(year)+'-'+str(month)+'-01')] = smlt*30 + self.ds['smlt_acc'].isel(time=t-1)

                #self['wfc'].loc[dict(time=str(year)+'-'+str(month)+'-01')] = d_wf + self['wfc'].isel(time=t-1)

            else:
                # If the current time step is outside the accumulative period, set the values to 0
                print(year,month)
                pass

        self.wf = True


