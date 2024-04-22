import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xdem
    # second option: shift dtm1
from xsnow.godh import dem_difference
from xsnow.godh import load_gdf, get_value_point
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pyproj

def regression_test_regressor(regressor, df, v_y, v_x = None,test_size=0.2, random_state=123,**params):
    '''
    Test regressor if need
    '''

    xg_reg = xgb.Booster()
    xg_reg.load_model(regressor)

    if v_x is None:
        v_x = xg_reg.feature_names

    # Create the training and test sets

    X_train, X_test, y_train, y_test = train_test_split(df[v_x], df[v_y].squeeze(), test_size=test_size, random_state=random_state)

    # Predict the labels of the test set: preds
    preds = xg_reg.predict(xgb.DMatrix(X_test))

    # compute the rmse: rmse
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print('----------')
    print("Validation RMSE: %f" % (rmse))

    mae = mean_absolute_error(y_test, preds)
    print('Validation MAE:', mae)

    r_s = r2_score(y_test, preds)
    print('Validation R-Square:', r_s)

    nmad = xdem.spatialstats.nmad(y_test-preds)
    print("Validation NMAD: %f" % (nmad))

    return y_test, preds

def tune_parameters(X_encoded,y, test_size=0.2,param_grid=None,objective='reg:absoluteerror'):
    '''
    Using grid search to tune parameters.

    Need to change param_grid.
    '''

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=test_size, random_state=123)

    if param_grid is None:
        param_grid = {
                    'n_estimators': [100, 200, 300],
                    #'learning_rate': [0.3, 0.2, 0.1, 0.05, 0.01],
                    'max_depth': [4, 6, 8, 10],
                    #'colsample_bytree': [0.5, 0.7, 1.0],
                    #'subsample': [0.5, 0.7, 1.0],
                    #'gamma': [0, 0.1, 0.5, 1]
                    }
    model = XGBRegressor(objective=objective)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
    grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

    # Print the best hyperparameters
    print('Best hyperparameters:', grid_search.best_params_)

    # Refit the XGBoost regressor with the best hyperparameters
    best_xgb_reg = XGBRegressor(**grid_search.best_params_)
    best_xgb_reg.fit(X_train, y_train)

    # Evaluate the performance of the tuned model on the validation set
    y_pred = best_xgb_reg.predict(X_val)

    # metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r_s = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    nmad = xdem.spatialstats.nmad(y_val-y_pred)
    print('----------')
    print('Validation RMSE:', rmse)
    print('Validation R-Square:', r_s)
    print('Validation MAE:', mae)
    print("Validation NMAD: %f" % (nmad))

    return grid_search.best_params_

def check_by_finse_lidar(sc,shift_px=(-0.5,1.128),dst_res=(1,1)):
    '''
        # LiDAR DEM bands
        # 1	Band 1: min	-9999	1239.9030787158	1338.3040133958
        # 2	Band 2: max	-9999	1240.1534029114	1338.3523308208
        # 3	Band 3: mean	-9999	1240.0323746089	1338.3277163162
        # 4	Band 4: idw	-9999	1240.0317506788	1338.3283852532
        # 5	Band 5: count	-9999	0.0000000000	502.7714250000
        # 6	Band 6: stdev   
    '''

    ## Lidar North
    fid_lidar_n =r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\fieldwork\Hardangervidda_2022_winter\data\processed_2022-03-11_HardangerL1\2022-03-11_HardangerL1_North_raster_1m.tif'
    ## Lidar South
    fid_lidar_s = r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\fieldwork\Hardangervidda_2022_winter\data\processed_2022-03-11_HardangerL1\2022-03-11_HardangerL1_South_raster_1m.tif'
    ## DTM1 
    #r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\fieldwork\Hardangervidda_2022_winter\dtm1\data\dtm1_33_113_119.tif'
    fid_dtm1 = r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\data\DEM\Norway\DTM1_UTM33\DTM1_11-11\33-113-119.tif'

    grid_nn2000 = xdem.DEM('no_kv_HREF2018B_NN2000_EUREF89.tif')

    LiDAR_dem_n = xdem.DEM(fid_lidar_n,indexes=3).reproject(dst_crs=pyproj.CRS(32633),dst_res=dst_res)
    LiDAR_dem_s = xdem.DEM(fid_lidar_s,indexes=3).reproject(dst_crs=pyproj.CRS(32633),dst_res=dst_res)
    dtm1 = xdem.DEM(fid_dtm1).reproject(dst_crs=pyproj.CRS(32633),dst_res=dst_res)
    dtm_1_ref = dtm1 + grid_nn2000.reproject(dtm1,resampling='bilinear')

    # if resolution is not 1 m, devide the pixel 
    if dst_res != (1,1):
        shift_px = [i / dst_res[0] for i in shift_px]

    ddem_1_n = dem_difference(dtm_1_ref.reproject(LiDAR_dem_n),LiDAR_dem_n,shift_px=shift_px)
    ddem_1_s = dem_difference(dtm_1_ref.reproject(LiDAR_dem_s),LiDAR_dem_s,shift_px=shift_px)

    df_sub_n = load_gdf(sc,ddem_1_n)
    df_sub_s = load_gdf(sc,ddem_1_s)
    df_sub_n['sd_lidar'] = get_value_point(ddem_1_n,df_sub_n)['h']
    df_sub_s['sd_lidar'] = get_value_point(ddem_1_s,df_sub_s)['h']
    df_sub = pd.concat([df_sub_n, df_sub_s],axis=0)

    #if 'date_' in df_sub.columns:
    #    new_df_era_sub = df_sub[df_sub['date_'] == '2022-03-05'].query('sd_lidar > 0')
    #else:
    #    new_df_era_sub = df_sub[pd.to_datetime(df_sub['date'], format= 'Y%m%d').dt.date == pd.to_datetime('2022-03-05').date()].query('sd_lidar > 0')
    return df_sub.query('sd_lidar > 0')

def produce_validation_from_nve(raw_df,sd_nve):
    
    if sd_nve is None:
        # 2008 NVE
        sd_nve = xdem.DEM(r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\data\nve_08_merge_m.tif')
    df_sub = load_gdf(raw_df,sd_nve)
    sd_nve_10 = sd_nve.reproject(dst_res = (10,10))
    sd_nve_20 = sd_nve.reproject(dst_res = (20,20))
    sd_nve_30 = sd_nve.reproject(dst_res = (30,30))

    # get nve validation from 2 m resoluition
    df_sub['sd_nve'] = get_value_point(sd_nve,df_sub)['h']

    # get nve validation from 10 m resoluition
    df_sub['sd_nve_10'] = get_value_point(sd_nve_10,df_sub)['h']
    
    # get nve validation from 20 m resoluition
    df_sub['sd_nve_20'] = get_value_point(sd_nve_20,df_sub)['h']

    # get nve validation from 30 m resoluition
    df_sub['sd_nve_30'] = get_value_point(sd_nve_30,df_sub)['h']

    return df_sub




