from typing_extensions import Self
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split # split data into training and testing data
from sklearn.model_selection import GridSearchCV # cross validation
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import holoviews as hv, datashader as ds, geoviews as gv, geoviews.tile_sources as gvts
from holoviews.operation.datashader import rasterize
hv.extension('bokeh')
from xsnow.goplot import final_histogram,threshold,RMSE
from msilib.schema import Class
import xarray as xr
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
import matplotlib.pyplot as plt
import xdem
import pyproj
import pickle
from typing import Dict
from scipy.stats import gamma
import shap
from xsnow.godh import load_gdf,get_value_point
from scipy.spatial import cKDTree
import numpy as np
from xsnow.test_ import check_by_finse_lidar, produce_validation_from_nve


def evaluate_downscaling(df,offset=0,cols=['sd_predict_dtm1','sd_predict_dtm10','sd_predict_cop30','sd_predict_fab']):
    
    evaluation_results = pd.DataFrame(columns=['Mean', 'RMSE', 'MAE', 'R2','Spearman_R','N'], index=cols)

    for column in cols:
        true_values = df['sd_nve_10']
        predicted_values = df[column] - offset

        mask = ~np.isnan(predicted_values) & ~np.isnan(true_values)
        
        mean_value = predicted_values.mean()
        rmse = np.sqrt(mean_squared_error(true_values[mask], predicted_values[mask]))
        mae = mean_absolute_error(true_values[mask], predicted_values[mask])
        r2 = r2_score(true_values[mask], predicted_values[mask])
        n = len(true_values[mask])
        spear_r = true_values[mask].corr(predicted_values[mask], method='spearman')

        evaluation_results.loc[column] = [mean_value, rmse, mae, r2, spear_r, n]
    
    # Print the DataFrame
    print(evaluation_results)

    return evaluation_results

def evaluate_bias_correction(df,
                             df_ref,
                             cols=['sd_correct_dtm1','sd_correct_dtm10','sd_correct_cop30','sd_correct_fab'],
                             cols_ref=['snowdepth_dtm1','snowdepth_dtm10','snowdepth_cop30','snowdepth_fab']):

    columns_to_match = ['h_te_best_fit', 'date', 'region', 'E', 'N']

    # Perform an inner join to merge 'sd_lidar' column from df_small into df
    merged_df = pd.merge(df, df_ref[columns_to_match + ['sd_lidar']], on=columns_to_match, how='inner')

    # Initialize an empty DataFrame to store the results
    evaluation_results = pd.DataFrame(columns=['Mean', 'RMSE', 'MAE', 'R2','Spearman_R','N'], index=cols)

    # Calculate the metrics for each column
    for column in cols+cols_ref:
        true_values = merged_df['sd_lidar']
        predicted_values = merged_df[column]

        mask = ~np.isnan(predicted_values) & ~np.isnan(true_values)
        
        mean_value = predicted_values.mean()
        rmse = np.sqrt(mean_squared_error(true_values[mask], predicted_values[mask]))
        mae = mean_absolute_error(true_values[mask], predicted_values[mask])
        r2 = r2_score(true_values[mask], predicted_values[mask])
        n = len(true_values[mask])
        spear_r = true_values[mask].corr(predicted_values[mask], method='spearman')

        evaluation_results.loc[column] = [mean_value, rmse, mae, r2, spear_r, n]
    # Print the DataFrame
    print(evaluation_results)

def get_shap(X, 
             model='sd_dtm1_abserror_250_10_qc_nve.json',
             n=100000):
    '''
    X: the features to be feed into model.
    model: the regressor to be explained.
    '''
    # Load the saved XGBoost model as a Booster object
    booster_model = xgb.Booster()
    booster_model.load_model(model)
    # Create an empty XGBRegressor and set its internal booster to the loaded model
    xgb_model = xgb.XGBRegressor()
    xgb_model._Booster = booster_model

    # Initialize the SHAP explainer
    explainer = shap.Explainer(xgb_model)
    if n:
        sampled_data = X.sample(n=n, random_state=42)  # Sample instances
    else:
        sampled_data = X
    shap_values = explainer(sampled_data)
    #shap_interaction_values = explainer.shap_interaction_values(sampled_data[:3000, :])

    return shap_values,sampled_data,explainer

def encoding_x_y(sf_cop, 
                 target_col = None, 
                 feature_col = None, 
                 weight_col = None, 
                 dummy_col = ['class'],
                 test_size=0, 
                 random_state=123,
                 add=0):
    """
    Encode categorical variables and return feature matrix X and target variable y for regression analysis.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input data.
    target_col : str, optional (default='dh_after_dtm10')
        Name of the target column.
    feature_cols : list of str, optional (default=None)
        Names of the feature columns.
    dummy_cols : list of str, optional (default=['class'])
        Names of the categorical columns.

    Returns:
    --------
    X : pandas.DataFrame
        Encoded feature matrix.
    y : pandas.Series
        Target variable.
    """
    if target_col is None:
        target_col = ['dh_after_dtm10']

    if feature_col is None:
        feature_col = ['h_te_std', 'h_te_uncertainty', 'h_te_best_fit', 'beam', 'pair', 'slope', 'aspect', 'planc', 'profc', 'maxc_arr', 'class']

    X = sf_cop[feature_col]
    y = sf_cop[target_col] + add

    if dummy_col is not None:
        if set(X.columns).issuperset(dummy_col):
            X = pd.get_dummies(X,columns=dummy_col)

    if test_size !=0:
        # Create the training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        return xgb.DMatrix(X_train, label=y_train,weight=sf_cop.loc[X_train.index,weight_col] if weight_col else None), xgb.DMatrix(X_test, label=y_test,weight=sf_cop.loc[X_test.index,weight_col] if weight_col else None)
    else:
        return xgb.DMatrix(X, label=y, weight=sf_cop.loc[X.index,weight_col] if weight_col else None), y

def regression_xgboost(dtrain,
                       dtest,
                       regression='quantile_regression',
                       alpha = np.array([0.05, 0.25, 0.5, 0.75, 0.95]),
                       **params):
    '''
    Train an XGBoost regressor on the input data and return the trained model.

    Parameters:
    -----------
    dtrain : xgboost.DMatrix
        Training data.
    regression : str, optional (default='quantile_regression')
    alpha : list of float, optional (default=[0.05, 0.25, 0.5, 0.75, 0.95])
        Quantile levels to use for quantile regression.
    **params : dict, optional
        Additional parameters to pass to the XGBoost regressor.

    Returns:
    --------
    xg_reg : xgboost.XGBRegressor or xgboost.XGBQuantileRegressor
        Trained XGBoost regressor.
    '''
    
    if regression == 'quantile_regression':
        # Modify params for quantile regression
        params['objective'] = 'reg:quantileerror'
        params['quantile_alpha'] = alpha
        params['tree_method'] = 'hist'

    # Fit the regressor to the training set
    xg_reg = xgb.train(params=params,
                       dtrain=dtrain,
                       num_boost_round=params.get('n_estimators', 400),  # Use the value from params or default to 300
                       early_stopping_rounds=params.get('early_stopping_rounds', 25),
                       evals=[(dtrain, "Train"), (dtest, "Test")],
                       verbose_eval=params.get('verbose_eval', False))

    # Predict the labels of the test set: preds
    preds = xg_reg.predict(dtest)

    if regression == 'quantile_regression':
        preds = preds[:,2].copy()

    rmse = np.sqrt(mean_squared_error(dtest.get_label(), preds))
    mae = mean_absolute_error(dtest.get_label(), preds)
    r_s = r2_score(dtest.get_label(), preds)
    nmad = xdem.spatialstats.nmad(dtest.get_label() - preds)

    # print the result
    print('----------')
    print("Validation RMSE: %f" % (rmse))
    print('Validation MAE:', mae)
    print('Validation R-Square:', r_s)
    print("Validation NMAD: %f" % (nmad))

    return xg_reg

def regression_task(new_df_era, 
                    list_regressor, 
                    v_x=None,
                    list_df_name = None, 
                    list_sd_name = None,
                    regression='quantile_regression',
                    era='sde_era',
                    weight=None,
                    add=0.1):
    
    '''
    This is for 4 models running together.
    1. regression
    2. generate sd_predict

    Note, gamma regression only support postiive values, so the add=0.1 will be deducted.
    
    You can choose which sd to add to. The default is sde_era (You need choose the regressor accordingdly).
    era:
        sd_era: snow depth era5 land
        sde_era: snow depth era5 land interpolated
        sd_se: snow depth senorge
        sde_se: snow depth senorge interpolated
    '''

    if list_df_name is None:
        list_df_name = ['df_predict_dtm10', 'df_predict_dtm1', 'df_predict_cop30', 'df_predict_fab']

    if list_sd_name is None:
        list_sd_name = ['sd_predict_dtm10', 'sd_predict_dtm1', 'sd_predict_cop30', 'sd_predict_fab']
    
    if 'month' not in new_df_era.columns:
        new_df_era['month'] = pd.DatetimeIndex(new_df_era['date']).month

    if v_x is None:
        v_x=['E', 'N','h_te_best_fit','slope', 'aspect', 'planc','profc','tpi','tpi_9','tpi_27','curvature','segment_cover','h_canopy','canopy_openness','sde_se','sde_era','wf_positive', 'wf_negative','month','smlt_acc', 'sf_acc']
    
    X_encoded, _ = encoding_x_y(new_df_era,target_col=['E'],feature_col=v_x)

    # append the result to list, order: dtm10 dtm1 cop30 fab 
    for i,df,sd in zip(list_regressor,list_df_name,list_sd_name):
        # load model
        reg = xgb.Booster()
        reg.load_model(i)
        if regression == 'quantile_regression':
            pred = reg.predict(X_encoded)
            new_df_era[df] = pred[:, 2]   # alpha=0.5, median
            new_df_era[sd] = pred[:, 2] + new_df_era[era] # alpha=0.5, median
            new_df_era[sd +'_05'] = pred[:, 0] + new_df_era[era] # alpha=0.05
            new_df_era[sd +'_95'] = pred[:, 4] + new_df_era[era] # alpha=0.95
            new_df_era[sd +'_75'] = pred[:, 3] + new_df_era[era] # alpha=0.75
            new_df_era[sd +'_25'] = pred[:, 1] + new_df_era[era] # alpha=0.25

            # clip in thresholds 0 and 10
            new_df_era[[sd, sd+'_05', sd+'_95', sd+'_25', sd+'_75']] = new_df_era[[sd, sd+'_05', sd+'_95', sd+'_25', sd+'_75']].clip(0, 10)

        elif regression == 'gamma_regression':
            new_df_era[sd] = reg.predict(X_encoded) - add
        else:
            new_df_era[df] = reg.predict(X_encoded)
            # turn df(subgrid variability) into snow depth
            new_df_era[sd] = new_df_era[df] + new_df_era[era]

    return new_df_era

def quantile_calculate(df, nve='sd_nve_10', dem='sd_predict_dtm1',split=0):

    _df = df.query(f'10 > {nve} > 0 & {dem} > 0')

    # Calculate quantiles for _df[dem] < 1
    dem_q_lt1 = _df.loc[_df[dem] < split, dem].quantile(q=np.linspace(0, 1, 250))
    # Calculate quantiles for _df[dem] > 1
    dem_q_gt1 = _df.loc[_df[dem] > split, dem].quantile(q=np.linspace(0, 1, 250))

    # Calculate delta for _df[dem] < 1 and _df[dem] > 1
    dq_lt1 = _df.loc[_df[dem] < split, nve].quantile(q=np.linspace(0, 1, 250)) / dem_q_lt1
    dq_gt1 = _df.loc[_df[dem] > split, nve].quantile(q=np.linspace(0, 1, 250)) / dem_q_gt1
    
    return {'delta_lt1': dq_lt1, 'dem_q_lt1':dem_q_lt1, 'delta_gt1': dq_gt1,'dem_q_gt1':dem_q_gt1 }

def quantile_mapping(df, dq_lt1, dq_gt1, split=0, dem='sd_predict_dtm1'):
    
    df[dem + '_'] = df[dem].copy()
    
    # for part of values below than split
    if split != 0:
        # Calculate quantiles for _df[dem] < 1
        dem_q_lt1 = df.loc[df[dem] < split, dem].quantile(q=np.linspace(0, 1, 250))
         # Apply quantile mapping for _df[dem] < 1
        condition = (df[dem] >= 0) & (df[dem] <= split)
        df.loc[condition, dem + '_'] = np.interp(df.loc[condition, dem], dem_q_lt1, dem_q_lt1 * dq_lt1)

    # for part of values above than split
    # Calculate quantiles for _df[dem] > 1
    dem_q_gt1 = df.loc[df[dem] > split, dem].quantile(q=np.linspace(0, 1, 250))
    
    # Apply quantile mapping for _df[dem] > 1
    df.loc[df[dem] > split, dem + '_'] = np.interp(df.loc[df[dem] > split, dem], dem_q_gt1, dem_q_gt1 * dq_gt1)

    # exclude negative values
    df.loc[df[dem+ '_'] < 0, dem + '_'] = 0


def quantile_mapping_original(df, dq_gt1, dem_q_gt1, split=0, dem='sd_predict_dtm1'):

    '''
    Quantile mapping using the quantiles from samples. Not from iteself.

    split == 1, means the values below 1 does not be corrected.

    Return df with a new columns: dem_
    '''

    df[dem + '_'] = df[dem].copy()

    if split != 0:
        # Apply quantile mapping for _df[dem] < 1
        condition = (df[dem] >= 0) & (df[dem] <= split)
        df.loc[condition, dem + '_'] = df.loc[condition,dem]

    # Apply quantile mapping for _df[dem] > 1. The main difference between quantile mapping is dem_q_gt1 is from original not computed.
    df.loc[df[dem] > split, dem + '_'] = np.interp(df.loc[df[dem] > split, dem], dem_q_gt1, dem_q_gt1 * dq_gt1)
    
    # exclude negative values
    df.loc[df[dem+ '_'] < 0, dem + '_'] = 0


def quantile_mapping_xr(data, dq_lt1, dq_gt1, split_f='sd_predict_dtm1', split=0, dem='sd_predict_dtm1',offset=0):
    """
    Apply quantile mapping to an xarray Dataset for values below and above a split point.

    Parameters:
    data (xr.Dataset): The input dataset.
    dq_lt1 (float): Correction factor for values below the split.
    dq_gt1 (float): Correction factor for values above the split.
    split (float): The value at which to split the dataset for separate corrections.
    dem (str): The name of the variable in the dataset to apply quantile mapping to.

    Returns:
    xr.Dataset: The dataset with quantile mapping applied to the specified variable.
    """

    if not isinstance(data, xr.Dataset):
        raise ValueError("Input data must be an xarray Dataset.")

    # offset
    data[dem] = data[dem] - offset

    # Copy the data to a new variable
    data[dem + '_'] = data[dem].copy()
    
    if split_f == 'curvarture':
        pass
    else:
        split_f = dem
     
    # Apply quantile mapping if there is a new split_f
    if split_f != dem:
        # Calculate quantiles for data[dem] < split
        dem_q_lt1 = data[dem].where(data[split_f] < split, drop=True).quantile(q=np.linspace(0, 1, 250))
        # Apply quantile mapping for data[dem] < split
        condition_lt = (data[split_f] <= split)
        subset_lt = data[dem].where(condition_lt, drop=True)
        interp_values_lt = np.interp(subset_lt.values.flatten(), dem_q_lt1.values, dem_q_lt1.values * dq_lt1)
        interp_da_lt = xr.DataArray(interp_values_lt.reshape(subset_lt.shape), dims=subset_lt.dims, coords=subset_lt.coords)
        data[dem + '_'] = xr.where(condition_lt, interp_da_lt, data[dem + '_'])

    # Apply quantile mapping for values above the split
    # Calculate quantiles for data[dem] > split
    dem_q_gt1 = data[dem].where(data[split_f] > split, drop=True).quantile(q=np.linspace(0, 1, 250))
    # Apply quantile mapping for data[dem] > split
    condition_gt = data[split_f] > split
    subset_gt = data[dem].where(condition_gt, drop=True)
    interp_values_gt = np.interp(subset_gt.values.flatten(), dem_q_gt1.values, dem_q_gt1.values * dq_gt1)
    interp_da_gt = xr.DataArray(interp_values_gt.reshape(subset_gt.shape), dims=subset_gt.dims, coords=subset_gt.coords)
    #data[dem + '_'] = xr.where(condition_gt, interp_da_gt, data[dem + '_'])
    interp_da_gt_aligned, data_dem_aligned = xr.align(interp_da_gt, data[dem + '_'], join='outer')
    data[dem + '_'] = xr.where(condition_gt, interp_da_gt_aligned, data_dem_aligned)
    # Exclude negative values
    data[dem + '_'] = data[dem + '_'].where(data[dem + '_'] >= 0, 0)

    return data

def quantile_mapping_original_xr(data, dq_gt1, dem_q_gt1, split=0, dem='sd_predict_dtm1',offset=0):

    '''
    Quantile mapping using the quantiles from samples. Not from itself.
    split == 1, means the values below 1 does not be corrected.
    Return data with a new variable: dem_
    '''
    # offset
    data[dem] = data[dem] - offset

    if not isinstance(data, xr.Dataset):
        raise ValueError("Input data must be an xarray Dataset.")

    # Copy the data to a new variable
    data[dem + '_'] = data[dem].copy()

    if split != 0:
        # Apply quantile mapping for data[dem] < 1
        condition_lt = (data[dem] >= 0) & (data[dem] <= split)
        data[dem + '_'] = xr.where(condition_lt, data[dem], data[dem + '_'])

    # Apply quantile mapping for data[dem] > split
    condition_gt = data[dem] > split
    # Make sure dem_q_gt1 and dq_gt1 are 1D numpy arrays
    dem_q_gt1 = np.array(dem_q_gt1)
    dq_gt1_values = np.array(dem_q_gt1 * dq_gt1)

    # Apply the interpolation only to the points that satisfy the condition
    subset = data[dem].where(condition_gt, drop=True)
    interp_values = np.interp(subset.values, dem_q_gt1, dq_gt1_values)

    # Create a new DataArray from the interpolated values
    interp_da = xr.DataArray(interp_values, dims=subset.dims, coords=subset.coords)

    # Update the values where the condition is met
    #data[dem + '_'] = xr.where(condition_gt, interp_da, data[dem + '_'])
    interp_da_gt_aligned, data_dem_aligned = xr.align(interp_da, data[dem + '_'], join='outer')
    data[dem + '_'] = xr.where(condition_gt, interp_da_gt_aligned, data_dem_aligned)
    # Exclude negative values
    data[dem + '_'] = data[dem + '_'].where(data[dem + '_'] >= 0, 0)

    return data

                                       
def match_frost_station(station_df,sd,threshold_distance=100):

    '''
    station_df: DataFrame has id, latitude and longitude columns
    sd: DataFrame has N and E columns.

    Return a paired point and data
    '''

    # Define the projected coordinate system (EPSG:32633)
    crs_target = pyproj.CRS('EPSG:32633')
    # Convert latitude and longitude columns to projected coordinates in EPSG 32633
    transformer = pyproj.Transformer.from_crs('EPSG:4326', crs_target)
    station_coords = station_df.apply(lambda row: transformer.transform(row['latitude'], row['longitude']), axis=1).tolist()
    
    sd_coords = sd[['E', 'N']].values

    # Build a KDTree from the station coordinates
    tree = cKDTree(station_coords)

    # Find the nearest station for each point within the threshold distance
    # Note: The distance_upper_bound should be provided in the projected coordinate system units (meters in this case)
    distances, indices = tree.query(sd_coords, distance_upper_bound=threshold_distance)

    # Filter out points where the nearest station is beyond the threshold distance
    valid_indices = np.where(distances <= threshold_distance)[0]
    matched_sd = sd.iloc[valid_indices]
    matched_stations = station_df[['id', 'name', 'masl']].iloc[indices[valid_indices]]

    # Create a mask to identify unmatched points
    unmatched_mask = np.isinf(distances)

    # Create an empty DataFrame for unmatched points
    unmatched_sd = sd[unmatched_mask]
    unmatched_stations = pd.DataFrame(columns=['id', 'name', 'masl'])

    # Merge the matched and unmatched stations with the sd dataframe based on indices and add distance column
    result_df = pd.concat([matched_sd, unmatched_sd], ignore_index=True)
    result_df = result_df.merge(pd.concat([matched_stations, unmatched_stations], ignore_index=True), left_index=True, right_index=True)
    result_df['distance'] = distances[valid_indices]

    return result_df



from msilib.schema import Class
import xarray as xr
import pandas as pd
from datetime import datetime,date, timedelta
import numpy as np
from tqdm import tqdm
import xgboost as xgb
import matplotlib.pyplot as plt
from xsnow.goplot import final_histogram,plot_scater_nve,plot_df_era_scatter,plot_point_map
from sklearn.metrics import accuracy_score
import numpy as np

class Snow_Regressor:
    def __init__(self, snow_depth_file, snow_cover_file, point_file=None):
        """
        Initialize a Snow_Regressor object.

        Parameters
        ----------
        snow_depth_file : str or pandas.DataFrame
            The name of a CSV file containing snow depth data, or a pandas DataFrame of snow depth data.
        snow_cover_file : str or pandas.DataFrame
            The name of a CSV file containing snow cover data, or a pandas DataFrame of snow cover data.
        point_file : str or None, optional
            The name of a CSV file containing point data, or None if no point data is available.

        Returns
        -------
        Snow_Regressor
            A Snow_Regressor object.

        """
        if isinstance(snow_depth_file, pd.DataFrame):
            self.sf = snow_depth_file
        else:
            self.sf = pd.read_csv(snow_depth_file)

        if isinstance(snow_cover_file, pd.DataFrame):
            self.sc = snow_cover_file
        else:
            self.sc = pd.read_csv(snow_cover_file)

        if point_file is not None:
            if isinstance(point_file, pd.DataFrame):
                self.pts = point_file
            else:
                self.pts = pd.read_csv(point_file)
        else:
            self.pts = None

        self.set_default_params()
        self.z = 'h_te_best_fit'
        self.era = 'sde_era'
        self.add_offset = False

        self.set_qc_paras()
        self.set_month()
        self.metrics = None
        self.drop_canopy()

    def set_month(self):
        if 'date_' in self.sc.columns :
            self.sc['month'] = pd.DatetimeIndex(self.sc['date_']).month

    def drop_canopy(self):
        # drop wrong values
        if 'h_mean_canopy' in self.sf.columns:
            self.sf.loc[abs(self.sf['h_mean_canopy']) > 60, 'h_mean_canopy'] = np.nan
            self.sf.loc[abs(self.sf['canopy_openness']) > 50, 'h_mean_canopy'] = np.nan
        if 'h_mean_canopy' in self.sc.columns:
            self.sc.loc[abs(self.sc['h_mean_canopy']) > 60,'h_mean_canopy'] = np.nan
            self.sc.loc[abs(self.sc['canopy_openness']) > 50,'canopy_openness'] = np.nan

        #self.sf.loc[(self.sf['segment_cover'] < 15 & self.sf['h_mean_canopy'].isnull()), 'h_mean_canopy'] = np.nan
        #self.sc.loc[(self.sc['segment_cover'] < 15 & self.sc['h_mean_canopy'].isnull()),'h_mean_canopy'] = np.nan

        #self.sf.loc[(self.sf['segment_cover'] > 15 & self.sf['canopy_openness'].isnull()), 'canopy_openness'] = np.nan
        #self.sc.loc[(self.sc['segment_cover'] > 15 & self.sc['canopy_openness'].isnull()), 'canopy_openness'] = np.nan


    def set_qc_paras(self,sf_qc=None,sc_qc=None):

        if sf_qc is None:
            #self.sf_qc = {'dtm1':' abs(dh_after_dtm1) < 10 and brightness_flag == 0',
            #              'dtm10':'abs(dh_after_dtm10) < 10 and brightness_flag == 0',
            #              'cop30':'abs(dh_after_cop30) < 10 and brightness_flag == 0',
            #              'fab':'abs(dh_after_fab) < 10 and brightness_flag == 0'}
            self.sf_qc = {'dtm1':'abs(dh_after_dtm10) < 10 & abs(dh_after_dtm1) < 10 & abs(dh_after_cop30) < 20 & abs(dh_after_fab) < 20 & brightness_flag == 0',
                          'dtm10':'abs(dh_after_dtm10) < 10 & abs(dh_after_dtm1) < 10 & abs(dh_after_cop30) < 20 & abs(dh_after_fab) < 20 & brightness_flag == 0',
                          'cop30':'abs(dh_after_dtm10) < 10 & abs(dh_after_dtm1) < 10 & abs(dh_after_cop30) < 20 & abs(dh_after_fab) < 20 & brightness_flag == 0',
                          'fab':'abs(dh_after_dtm10) < 10 & abs(dh_after_dtm1) < 10 & abs(dh_after_cop30) < 20 & abs(dh_after_fab) < 20 & brightness_flag == 0'}
        
        # set cutting-out value
        if sf_qc is None:
            self.sc_qc = {'dtm1':f'(-0.1 < sd_correct_dtm1 < 8) & subset_te_flag == 5 & n_te_photons > 10 & slope < 45 & abs(tpi_9) < 12 & {self.era} < 15',
                        'dtm10':f'(-0.1 < sd_correct_dtm10 < 8) & subset_te_flag == 5 & n_te_photons > 10 & slope < 45 & abs(tpi_9) < 12 & {self.era} < 15',
                        'cop30':f'(-0.5 < sd_correct_cop30 < 8) & subset_te_flag == 5 & n_te_photons > 10 & slope < 45 & abs(tpi_9) < 12 & {self.era} < 15',
                        'fab':f'(-0.5 < sd_correct_fab < 8) & subset_te_flag == 5 & n_te_photons > 10 & slope < 45 & abs(tpi_9) < 12 & {self.era} < 15'}

    def set_default_params(self,
                           params=None,
                           params_2 = None,
                           params_3 = None,
                           params_4 = None):

        """
        Set the default parameters for the XGBoost regressor.

        """

        if params is None:
            self.params = {
                'objective': 'reg:absoluteerror',
                'max_depth': 10,
                'learning_rate': 0.1,
                'n_estimators': 250,
                'min_child_weight': 1,
                'subsample': 0.7,
                'colsample_bytree': 1,
                'gamma': 0.1,
            }
        else:
            self.params = params

        if params_2 is None:
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
        else:
            self.params_2 = params_2

        if params_3 is None:
            self.params_3 = {
                'objective': 'reg:quantileerror',
                "tree_method": "hist",
                "quantile_alpha": np.array([0.05, 0.25, 0.5, 0.75, 0.95]),
                'max_depth': 8,
                'learning_rate': 0.1,
            }
        else:
            self.params_3 = params_3

        if params_4 is None:
            self.params_4 = {
                'objective': 'reg:gamma',
                "n_estimators": 300,
                'max_depth': 10,
                'learning_rate': 0.3,
            }
        else:
            self.params_4 = params_4

    def drop_columns_rows(self):
        '''
        drop columns and duplicated rows
        '''

        # snow cover
        self.sc = self.sc.drop_duplicates(subset=['latitude', 'longitude', 'date','h_te_best_fit'], keep='last')
        drop_list_sc = ['latitude_20m_0', 'latitude_20m_1',
                        'latitude_20m_2', 'latitude_20m_3', 'latitude_20m_4', 'longitude_20m_0',
                        'longitude_20m_1', 'longitude_20m_2', 'longitude_20m_3',
                        'longitude_20m_4', 'h_te_best_fit_20m_0',
                        'h_te_best_fit_20m_1', 'h_te_best_fit_20m_2', 'h_te_best_fit_20m_3',
                        'h_te_best_fit_20m_4', 'z','geometry','segment_watermask',
                        'fid_dtm10', 'coreg_offset_east_px_dtm10',
                        'coreg_offset_north_px_dtm10', 'fid_cop30',
                        'coreg_offset_east_px_cop30', 'coreg_offset_north_px_cop30', 'fid_dtm1',
                        'coreg_offset_east_px_dtm1', 'coreg_offset_north_px_dtm1', 
                        #'coreg_bias_dtm1','coreg_bias_fab', 'coreg_bias_cop30','coreg_bias_dtm10',
                        'fid_fab','coreg_offset_east_px_fab', 'coreg_offset_north_px_fab']
        cols_to_drop = list(set(self.sc.columns).intersection(set(drop_list_sc)))
        self.sc.drop(cols_to_drop, axis=1, inplace=True)
        print('---length of sc:', len(self.sc))

        # snow free
        self.sf = self.sf.drop_duplicates(subset=['latitude', 'longitude', 'date','h_te_best_fit'], keep='last')
        drop_list_sf = ['latitude_20m_0', 'latitude_20m_1','segment_watermask',
                        'latitude_20m_2', 'latitude_20m_3', 'latitude_20m_4', 'longitude_20m_0',
                        'longitude_20m_1', 'longitude_20m_2', 'longitude_20m_3',
                        'longitude_20m_4', 'h_te_best_fit_20m_0','geometry',
                        'h_te_best_fit_20m_1', 'h_te_best_fit_20m_2', 'h_te_best_fit_20m_3',
                        'h_te_best_fit_20m_4', 'z', 'fid', 'fid_dtm10', 
                        #'coreg_bias_cop30', 'coreg_bias_dtm1', 'coreg_bias_fab','coreg_bias_dtm10',
                        'fid_cop30', 'fid_dtm1', 'fid_fab']
        cols_to_drop = list(set(self.sf.columns).intersection(set(drop_list_sf)))
        self.sf.drop(cols_to_drop, axis=1, inplace=True)
        print('---length of sf:', len(self.sf))

        if self.pts:
            self.pts = self.pts.drop_duplicates(subset=['latitude', 'longitude', 'date','h_te_best_fit'], keep='last')
            print('---length of pts:', len(self.pts))

    def add_columns(self,df, df_add=None,add_columns=None,on_columns=None):
        '''
        add new columns into df by merge a new dataframe
        '''

        if df_add is None:
            df_add = pd.read_csv(r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\all_segments.csv', encoding='utf-8')
            df_add = df_add.drop_duplicates(subset=['latitude', 'longitude', 'date','h_te_best_fit'], keep='last')
        if add_columns is None:
            add_columns = ['cloud_flag_atm','snr','terrain_slope','h_canopy','h_te_skew','urban_flag', 'night_flag','h_te_uncertainty','brightness_flag','h_te_interp']
            add_columns = list(set(add_columns).intersection(set(df_add.columns)).difference(set(df.columns)))
        if on_columns is None:
            on_columns = list(set(df_add.columns).intersection(set(df.columns)))
        return pd.merge(df , df_add[add_columns+on_columns], on=on_columns, how='inner')

    def concat_sc_sf(self, on_columns = None):

        '''
        concat two dataframe by shared columns
        '''

        if on_columns is None:
            on_columns = ['latitude', 'longitude', 'E', 'N', 'h_te_best_fit', 'slope', 'aspect', 'planc', 'profc', 'tpi', 'tpi_9', 'tpi_27', 'curvature', 'segment_cover', 'h_mean_canopy','h_canopy', 'canopy_openness', 'date']
            on_columns = list(set(on_columns).intersection(set(self.sc.columns)).intersection(set(self.sf.columns)))

        # Concatenate the two DataFrames along the rows
        concatenated_df = pd.concat([self.sf, self.sc])
        return concatenated_df[on_columns] 


    def check_duplicates(self):
        print('---sc:duplicates---',self.sc.duplicated().sum())
        print('---sf:duplicates---',self.sf.duplicated().sum())
        print('---sf:null---',self.sc[['latitude', 'longitude', 'n_te_photons','h_te_best_fit']].isnull().sum())
        print('---sc:null---',self.sf[['latitude', 'longitude', 'n_te_photons','h_te_best_fit']].isnull().sum())

    def print_metrics(self,df=None,after_list=['dh_after_dtm1'],before_list=['dh_before_dtm1'],name_list=['dtm1'],perc_t=99.75,dlim=100,std_t=None,round=6,window=None):
        '''
        metrics for snow free, before and after coreg.
        '''
        dic = {}
        if df is None:
            df = self.sf

        for before,name in zip(before_list,name_list):
            N = len(df[before])
            df_ = threshold(df[before],perc_t=perc_t,dlim=dlim,std_t=std_t,window=window)

            n05_before = len(df_[abs(df_)<0.5])/N*100
            nmad_before = xdem.spatialstats.nmad(df_)
            n1_before = len(df_[abs(df_)<1])/N*100
            rmse_before = RMSE(df_)

            dic.setdefault(name, {})
            dic[name]['N'] = N
            dic[name]['n05_before'] = n05_before
            dic[name]['n1_before'] = n1_before
            dic[name]['nmad_before'] = nmad_before
            dic[name]['rmse_before'] = rmse_before

        for after,name in zip(after_list,name_list):
            N = len(df[after])
            df_ = threshold(df[after],perc_t=perc_t,dlim=dlim,std_t=std_t,window=window)

            n05_after = len(df_[abs(df_)<0.5])/N*100
            nmad_after = xdem.spatialstats.nmad(df_)
            n1_after = len(df_[abs(df_)<1])/N*100
            rmse_after = RMSE(df_)

            dic.setdefault(name, {})
            dic[name]['n05_after'] = n05_after
            dic[name]['n1_after'] = n1_after
            dic[name]['nmad_after'] = nmad_after
            dic[name]['rmse_after'] = rmse_after

        df_metrics = pd.DataFrame(dic)
        print(df_metrics.round(round))
        self.metrics = df_metrics

        return df_metrics

    def use_interp(self):
        '''
        the original difference is calculated by 'h_te_best_fit'. Applying this function can change it into 'h_te_interp'.
        '''
        for df in [self.sc,self.sf]:
            if 'dh_after_dtm1' in df:
                dh_list = ['dh_after_dtm1','dh_after_dtm10','dh_after_cop30','dh_after_fab']
            elif 'snowdepth_dtm1' in df:
                dh_list = ['snowdepth_dtm1','snowdepth_dtm10','snowdepth_cop30','snowdepth_fab']

            if self.z != 'h_te_interp':
                for dh in dh_list:
                    df[dh] = df[dh] + df['h_te_interp'] - df['h_te_best_fit']
                # update information
        self.z = 'h_te_interp'   
        print('---switched to h_te_interp, please do regression to refresh snow_depth_correct')

    def use_best_fit(self):
        '''
        reverse use_interp
        '''
        for df in [self.sc,self.sf]:
            if 'dh_after_dtm1' in df:
                dh_list = ['dh_after_dtm1','dh_after_dtm10','dh_after_cop30','dh_after_fab']
            elif 'snowdepth_dtm1' in df:
                dh_list = ['snowdepth_dtm1','snowdepth_dtm10','snowdepth_cop30','snowdepth_fab']

            if self.z != 'h_te_best_fit':
                for dh in dh_list:
                    df[dh] = df[dh] + df['h_te_best_fit'] - df['h_te_interp']
                # update information
        self.z = 'h_te_best_fit'
        print('---switched to h_te_best_fit, please do regression to refresh snow_depth_correct')

    def use_geosegment(self):
        '''
        Which elevation from ICESat-2 to be corrected?
        '''
        if self.z != 'h_te_best_fit_20m_2':
            # swap dh 
            self.sf['dh_after_dtm1'] = self.sf['dh_after_dtm1'] - self.sf['difference']
            self.sf['dh_after_dtm10'] = self.sf['dh_after_dtm10'] - self.sf['difference']
            self.sf['dh_after_cop30'] = self.sf['dh_after_cop30'] - self.sf['difference']
            self.sf['dh_after_fab'] = self.sf['dh_after_fab'] - self.sf['difference']
            # swap sd
            self.sc['snowdepth_dtm1'] = self.sc['snowdepth_dtm1'] - self.sc['difference']
            self.sc['snowdepth_dtm10'] = self.sc['snowdepth_dtm10'] - self.sc['difference']
            self.sc['snowdepth_cop30'] = self.sc['snowdepth_cop30'] - self.sc['difference']
            self.sc['snowdepth_fab'] = self.sc['snowdepth_fab'] - self.sc['difference']

            self.z = 'h_te_best_fit_20m_2'

            print('---switched to geosegment, please do regression to refresh snow_depth_correct')

    def use_segment(self):

        if self.z != 'h_te_best_fit':
            # swap dh 
            self.sf['dh_after_dtm1'] = self.sf['dh_after_dtm1'] + self.sf['difference']
            self.sf['dh_after_dtm10'] = self.sf['dh_after_dtm10'] + self.sf['difference']
            self.sf['dh_after_cop30'] = self.sf['dh_after_cop30'] + self.sf['difference']
            self.sf['dh_after_fab'] = self.sf['dh_after_fab'] + self.sf['difference']
            # swap sd
            self.sc['snowdepth_dtm1'] = self.sc['snowdepth_dtm1'] + self.sc['difference']
            self.sc['snowdepth_dtm10'] = self.sc['snowdepth_dtm10'] + self.sc['difference']
            self.sc['snowdepth_cop30'] = self.sc['snowdepth_cop30'] + self.sc['difference']
            self.sc['snowdepth_fab'] = self.sc['snowdepth_fab'] + self.sc['difference']
            
            self.z = 'h_te_best_fit'

            print('---switched to segment, please do regression to refresh snow_depth_correct')
        else:
            print('---By default, the elevation is segment')
    
    def use_era(self):
        '''
        Which snow depth to be downscaled?

        use era (sd_era), reverse of use_era_interp, 
        '''

        if self.era == 'sde_era':
            col_list = ['df_dtm1_era5','df_dtm10_era5','df_cop30_era5','df_fab_era5']

            for col in list(set(col_list).intersection(set(self.sf.columns))):
                self.sc[col] = self.sc[col] + self.sc['sd_era'] - self.sc['sde_era']
            # update information
            self.era = 'sd_era'
            print('---df to era has been refreshed to sd_era')
            # update qc
            self.set_qc_paras()

    def use_era_interp(self):
        '''
        Which snow depth to be downscaled?

        use era_interp (sde_era) by default.
        '''
        if self.era == 'sd_era':
            col_list = ['df_dtm1_era5','df_dtm10_era5','df_cop30_era5','df_fab_era5']

            for col in list(set(col_list).intersection(set(self.sf.columns))):
                self.sc[col] = self.sc[col] + self.sc['sde_era'] - self.sc['sd_era']
            # update information
            self.era = 'sde_era'
            print('---df to era has been refreshed to sde_era')
            # update qc
            self.set_qc_paras()

    def plot_hist_correction(self,
                             ax=None,
                             dem='dtm1',  
                             raw_dh=None,
                             coreg_dh=None,
                             cor_dh=None,
                             perc_t=100,
                             std_t = None,
                             window=None):
        '''
        print hist to show coreg vs correction vs raw  
        '''

        if raw_dh is None:
            raw_dh = f'dh_before_{dem}'
        if coreg_dh is None:          
            coreg_dh = f'dh_after_{dem}'
        if cor_dh is None:
            cor_dh = f'dh_reg_{dem}'
        if ax is None:
            fig,ax = plt.subplots(figsize=(7,5))

        final_histogram(self.sf[coreg_dh],
                        self.sf[cor_dh],
                        dH_ref=self.sf[raw_dh],
                        ax=ax,
                        legend=['After coreg','After bias-correction','Raw'],
                        range=(-10,10),
                        perc_t=perc_t,
                        std_t = std_t,
                        window=window)


    def regression_tree_binary(self,
                               v_x = ['terrain_slope','E', 'N','h_te_best_fit','slope', 'aspect', 'planc','profc','tpi','tpi_9','tpi_27','curvature','segment_cover']
                               ):
        '''
        EXAMPLE TO USE:
        # WHEN we need to adjust the tree parameters for sc, because the DEM error is related to vegetation.
        # IT WILL save the winter canopy
        dems.sc['h_canopy_winter'] = dems.sc['h_canopy']
        dems.sc['h_mean_canopy_winter'] = dems.sc['h_mean_canopy']
        dems.sc['canopy_openness_winter'] = dems.sc['canopy_openness']

        dems.regression_tree_binary()
        dems.regression_tree_parameters(
            regressor=['tree_mean_abserror_250_10_new.json','tree_openness_abserror_250_10_new.json','tree_abserror_250_10_new.json'],
            vegetation_mask= dems.sc['tree_presence']>0,
            mode = 'times'
            #regressor_name=['tree_mean_abserror_250_10.json','tree_openness_abserror_250_10.json']
            )

        '''
        
        
        # Define the features and target variable
        features = v_x
        target = 'tree_presence'

        # Create the target variable by checking if the canopy height is null
        self.sf['tree_presence'] = self.sf['h_canopy'].notnull().astype(int)
        self.sc['tree_presence'] = self.sc['h_canopy_winter'].notnull().astype(int)
        print('SC - Before - tree presence % :',self.sc['tree_presence'].sum()/len(self.sc['tree_presence'])*100)

        #fig,ax =plt.subplots(1,3,figsize=(13,4))
        #plot_segment_cover(self.sf,ax=ax[0],title='Snow-free')
        #plot_segment_cover(self.sc,ax=ax[1],title='Snow-on before')

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.sf[features], self.sf[target], test_size=0.2, random_state=42)

        # Train an XGBoost binary classification model
        params = {'objective': 'binary:logistic', 'eval_metric': 'error','max_depth':7}
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(params, dtrain)

        # Make predictions on the test set
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)
        y_pred = [round(value) for value in y_pred]

        # Evaluate the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        
        # run on sc
        dmissing = xgb.DMatrix(self.sc.loc[self.sc['tree_presence']==0,features])
        self.sc.loc[self.sc['tree_presence']==0,'tree_presence'] = model.predict(dmissing)
        print('SC - After - tree presence % :',self.sc['tree_presence'].sum()/len(self.sc['tree_presence'])*100)
        #plot_segment_cover(self.sc,ax=ax[2],title='Snow-on after')
        
        #plt.show()


    def regression_tree_parameters(self,
                                   regressor=None,                                   
                                   v_x = ['terrain_slope','E', 'N','h_te_best_fit','slope', 'aspect', 'planc','profc','tpi','tpi_9','tpi_27','curvature','segment_cover'],
                                   v_y = ['h_mean_canopy','canopy_openness','h_canopy'],
                                   regressor_name=['tree_mean_abserror_250_10.json','tree_openness_abserror_250_10.json'],
                                   col= ['h_mean_canopy','canopy_openness','h_canopy'],
                                   vegetation_mask = None,
                                   mode='mask'):
        '''
        update canopy by a regressor (if available) or a regression task
        '''
        
        # prepare regressor

        if regressor:
            xg_reg_height = xgb.Booster()
            xg_reg_height.load_model(regressor[0])

            xg_reg_openness = xgb.Booster()
            xg_reg_openness.load_model(regressor[1])

            xg_reg_h = xgb.Booster()
            xg_reg_h.load_model(regressor[2])

            v_x = xg_reg_openness.feature_names 
        else:
            # need to train
            X_height,X_height_t = encoding_x_y(self.sf.query('abs(h_canopy) < 80 and abs(h_mean_canopy) < 50 and (0 < canopy_openness < 50)'),
                                       target_col=v_y,
                                       feature_col=v_x,
                                       test_size=0.2)
            X_o,X_o_t = encoding_x_y(self.sf.query('abs(h_canopy) < 80 and abs(h_mean_canopy) < 50 and (0 < canopy_openness < 50)'),
                                       target_col=v_y,
                                       feature_col=v_x,
                                       test_size=0.2)
            X_h,X_h_t = encoding_x_y(self.sf.query('abs(h_canopy) < 80 and abs(h_mean_canopy) < 50 and (0 < canopy_openness < 50)'),
                                       target_col=v_y,
                                       feature_col=v_x,
                                       test_size=0.2)
            params = self.params
            xg_reg_height = regression_xgboost(X_height,X_height_t,**params)
            xg_reg_openness = regression_xgboost(X_o,X_o_t,**params)
            xg_reg_h = regression_xgboost(X_h,X_h_t,**params)

            if regressor_name:
                xg_reg_height.save_model(regressor_name[0])
                xg_reg_openness.save_model(regressor_name[1])
                xg_reg_h.save_model(regressor_name[2])

        # sc
        if mode == 'mask':
            X_encoded,y = encoding_x_y(self.sc[vegetation_mask],target_col=['E'],feature_col=v_x)
            print('SC - Before - mean of mean_canopy:',self.sc['h_mean_canopy'].mean())
            self.sc.loc[vegetation_mask,col[0]] = xg_reg_height.predict(X_encoded)
            self.sc.loc[vegetation_mask,col[1]] = xg_reg_openness.predict(X_encoded)
            self.sc.loc[vegetation_mask,col[2]] = xg_reg_h.predict(X_encoded)

        if mode == 'times':
            X_encoded,y = encoding_x_y(self.sc[vegetation_mask],target_col=['E'],feature_col=v_x)

            self.sc.loc[vegetation_mask,col[0]] = np.nanmax(np.vstack((self.sc.loc[vegetation_mask,col[0]], self.sc.loc[vegetation_mask,'tree_presence'] * xg_reg_height.predict(X_encoded))),axis=0)
            self.sc.loc[vegetation_mask,col[1]] = np.nanmax(np.vstack((self.sc.loc[vegetation_mask,col[1]], self.sc.loc[vegetation_mask,'tree_presence'] * xg_reg_openness.predict(X_encoded))),axis=0)
            self.sc.loc[vegetation_mask,col[2]] = np.nanmax(np.vstack((self.sc.loc[vegetation_mask,col[2]], self.sc.loc[vegetation_mask,'tree_presence'] * xg_reg_h.predict(X_encoded))),axis=0)
        print('SC - After - mean of mean_canopy:',self.sc['h_mean_canopy'].mean())

    def regression_sf_sc(self,
                        dem='dtm1',                                   
                        regressor=None,
                        v_x=None,
                        v_y=None,
                        regressor_name='dtm1_abserror_250_10.json',
                        col=None,
                        col_dh=None,
                        coeff=1.0,
                        weight=None,
                        ):
        '''
        The first regression, based on snow-free data.

        Train or use a regressor to get correct DEM (get the bais and use it on snow depth of sc,
        to get 'sd_correct_', and the difference to ERA5 'df_{dem}_era5.'

        Results returned as columns in sc:
        (1) real snow depth; 'sd_correct_dtm1' =  'snowdepth_dtm1'(from oberservation) - 'pred_correct_dtm1'(from prediction, trained by dh_after_coreg)
        (2) subgrid variability: 'df_{dem}_era5' = sd_correct_dtm1 - sc[era5]

        This regressor is on DEM errors side.
        '''
        if v_x is None:
            # if not set, v_x will be full..
            v_x=['cloud_flag_atm','snr','terrain_slope','h_canopy','h_te_std','n_te_photons','subset_te_flag','E', 'N','h_te_best_fit','beam','pair','slope', 'aspect', 'planc','profc','tpi','tpi_9','tpi_27','curvature','segment_cover','h_mean_canopy','canopy_openness', 'urban_flag', 'night_flag','region','h_te_skew','h_te_uncertainty']
        
        # Y: dh_after_dem is the elevation difference between ICESat-2 snow-free segment and DEM.
        if v_y is None:
            v_y=[f'dh_after_{dem}']
        if col is None:
            col=[f'snowdepth_{dem}',f'sd_correct_{dem}',f'df_{dem}_era5',f'pred_correct_{dem}']
        if col_dh is None:
            col_dh=[f'dh_after_{dem}',f'dh_reg_{dem}']

        # no need dummy
        v_dummies = None
        
        # (0) Prepare a regressor
        if regressor:
            xg_reg = xgb.Booster()
            xg_reg.load_model(regressor)
            v_x = xg_reg.feature_names
        else:
            # if need to train
            dtrain,dtest = encoding_x_y(self.sf.query(self.sf_qc[dem]),
                                       target_col=v_y,
                                       feature_col=v_x,
                                       dummy_col=v_dummies,
                                       test_size=0.2,
                                       weight_col=weight)
            params = self.params
            xg_reg = regression_xgboost(dtrain,dtest,regression='mae',**params)
            if regressor_name:
                xg_reg.save_model(regressor_name)

        # (1) predict for sf: get dh_reg_dtm1 by regression based on y amd dh_after_dtm1 from snow-free
        if col_dh: 
            X_encoded,y = encoding_x_y(self.sf, target_col=['E'], feature_col=v_x, dummy_col=v_dummies)
            self.sf[col_dh[1]] = self.sf[col_dh[0]] - xg_reg.predict(X_encoded)

        # (2) predict for sc: use the regreesor to get 'pred_correct_dtm1'
        X_encoded,y = encoding_x_y(self.sc,target_col=['E'],feature_col=v_x,dummy_col=v_dummies)
        self.sc[col[3]] = xg_reg.predict(X_encoded)

        #  'sd_correct_dtm1' =  'snowdepth_dtm1' - 'pred_correct_dtm1'
        self.sc[col[1]] = self.sc[col[0]] - self.sc[col[3]] * coeff
        # generate the difference with ERA5: df =  'sd_correct_dtm1' - era
        self.sc[col[2]] = self.sc[col[1]] - self.sc[self.era]                   

    def gamma_transform(self,
                        dem=['dtm1','dtm10','cop30','fab'],
                        ):
        '''
        Transform retrived snow depth into gamma distribution
        '''
        self.shift_value = 2

        for dems in dem:
            # shift to positive
            shift_sd = self.sc[f'sd_correct_{dems}'] + self.shift_value
            shift_sd_clip =shift_sd.clip(lower=0.01)

            # gamma transform
            self.shape, self.loc, self.scale = gamma.fit(shift_sd_clip)
            self.sc[f'sd_gamma_{dems}'] = gamma.ppf(self.sc[f'sd_shift_{dems}'], self.shape, loc=self.loc, scale=self.scale)


    def reverse_gamma_transform(self,
                                dem=['dtm1','dtm10','cop30','fab']
                                ):
        # Inverse gamma transformation
        for dems in dem:
            self.sc[f'sd_pred_{dems}'] = gamma.cdf(self.sc[f'sd_predict_{dems}'], self.shape, loc=self.loc, scale=self.scale) - self.shift_value
        
    
    def validation_lidar(self,
                         dem = 'dtm1',
                         raw_sd= None,
                         corrected_sd = None,
                         ref = None,
                         dst_res = (1,1),
                         format='%Y-%m-%d'):
        
        '''
        validate the corrected snow depth by lidar validtion dataset
        '''

        if raw_sd is None:
            raw_sd = f'snowdepth_{dem}'
        if corrected_sd is None:
            corrected_sd= f'sd_correct_{dem}'
        if ref is None:
            ref = f'df_{dem}_era5'    
    
        # hist plot
        #self.plot_hist_scatter_era(raw_sd=raw_sd,corrected_sd=corrected_sd,ref=ref)

        # get 'sd_lidar' columns
        df_sub = check_by_finse_lidar(self.sc.query(self.sc_qc[dem]),shift_px=(-0.5,1.128),dst_res = dst_res)
        df_sub = df_sub[pd.to_datetime(df_sub['date'],format=format).dt.date == pd.to_datetime('2022-03-05').date()]
        
        #plot_correction_comparision(df_sub, 'sd_correct_dtm1', 'snowdepth_dtm1')

        # compare  'sd_lidar' with 'corrected_sd' in the same date
        p =plot_scater_nve(df_sub.query(f'6 > {corrected_sd} > 0'),'sd_lidar',corrected_sd,title='corrected snow depth vs lidar')
        p_raw =plot_scater_nve(df_sub.query(f'6 > {raw_sd} > 0'),'sd_lidar',raw_sd,title='raw snow depth vs lidar')

        return (p + p_raw).cols(2), df_sub

    def plot_hist_scatter_era(self,
                                dem = 'dtm1',
                                raw_sd= None,
                                corrected_sd = None,
                                ref = None):
        '''
        comapre with era5

        plot scatter and hist to era5 snow depth
        '''

        if raw_sd is None:
            raw_sd = f'snowdepth_{dem}'
        if corrected_sd is None:
            corrected_sd= f'sd_correct_{dem}'
        if ref is None:
            ref = f'df_{dem}_era5'  
        # scatter
        fig,ax = plt.subplots(1,2,figsize=(14,5))
        plot_df_era_scatter(self.sc.query(self.sc_qc[dem]),x=self.era,y=corrected_sd,ax=ax[0],vmax=2000,vmin=0,x_lims=(0,6),y_lims=(0,6),cmap=plt.cm.twilight,n_quantiles=50,fit_line=False)
        
        # hist
        df_era = self.sc.query(f'0 < {self.era} < 10')
        df_dtm = self.sc.query(f'-4 < {corrected_sd} < 10')
        final_histogram(df_era[self.era], df_dtm[corrected_sd], dH_ref=df_dtm[ref], ax=ax[1],legend=['ERA5 Land','Corrected snow depth', 'Difference to ERA'],range=(-4,8),window=(-4,8));
        plt.show()

    def regression_sc_sd_predict(self,
                                dem='dtm1',
                                raw_pts=None,
                                regressor=None,
                                v_x=None,
                                v_y=None,
                                regressor_name='sd_dtm1_abserror_250_10.json',
                                regression='others',
                                col=None,
                                alpha = np.array([0.05, 0.25, 0.5, 0.75, 0.95]),
                                add=0,
                                weight=None):
        '''
        The second regressor for snow depth.

        The input sc must contains 'df_dtm1_era5' (from regression_sc_sd)

        Train (or use) a regressor to predict snow depth for raw_pts:

        'sd_predict' = 'era' + df_predict (train by df_dem_era, the difference between correct snow depth and era5)

        '''

        # Preparetion
        if v_x is None:
            v_x=['E', 'N', 'h_te_best_fit','slope', 'aspect', 'planc','profc','tpi','tpi_9','tpi_27','curvature','segment_cover','h_canopy','canopy_openness','sd_era','wf_positive', 'wf_negative','month','smlt_acc', 'sf_acc']
        if v_y is None:
            v_y=[f'df_{dem}_era5',f'sd_correct_{dem}']
        if col is None:
            col=[f'df_predict_{dem}',f'sd_predict_{dem}',f'sd_predict_{dem}_lower',f'sd_predict_{dem}_upper']
        
        if regression == 'quantile_regression':
            params = self.params_3
            name_y = f'df_{dem}_era5'
        elif regression == 'gamma_regression':
            # make sure the vy is snow depth istead of difference
            assert f'sd_correct_{dem}' in v_y
            add = 0.1
            params = self.params_4
            name_y = f'sd_correct_{dem}'
        else:
            params = self.params_2 
            name_y = f'df_{dem}_era5'
            
        # raw_pts: add month if not exist
        if raw_pts is None:
            raw_pts = self.pts
            raw_pts['month'] = pd.DatetimeIndex(raw_pts['date_']).month

        # (1) Prepare regressor
        if regressor is None:
            # Train regressor
            dtrain, dtest = encoding_x_y(self.sc.query(self.sc_qc[dem]), 
                                        target_col=name_y, 
                                        feature_col=v_x,
                                        test_size=0.2,
                                        weight_col=weight,
                                        add=add)
            xg_reg = regression_xgboost(dtrain, dtest, regression=regression, alpha=alpha, **params)
            if regressor_name:
                xg_reg.save_model(regressor_name)
        else:
            # Or, Load regressor, using v_x from regressor not from v_x
            xg_reg = xgb.Booster()
            xg_reg.load_model(regressor)
            v_x = xg_reg.feature_names

        # (2) Predict snow depth and return it by raw_pts
        X_encoded, y = encoding_x_y(raw_pts, target_col=['E'], feature_col=v_x)

        if regression == 'quantile_regression':
            raw_pts[col[0]] = xg_reg.predict(X_encoded)[:, 2]   # alpha=0.5, median
            raw_pts[col[2]] = xg_reg.predict(X_encoded)[:, 1] + raw_pts[self.era]  # alpha=0.25
            raw_pts[col[3]] = xg_reg.predict(X_encoded)[:, 3] + raw_pts[self.era]  # alpha=0.75
            raw_pts[col[1]] = raw_pts[col[0]] + raw_pts[self.era]
        elif regression == 'gamma_regression':
            raw_pts[col[1]] = xg_reg.predict(X_encoded) - add 
        else:
            raw_pts[col[1]] = xg_reg.predict(X_encoded) + raw_pts[self.era] 

        return raw_pts

    def regression_pts_2022(self,df_2022,month=3):
        '''
        return df with predicted snow depth 4 in 1. Note: 100 km local training

        df_2022: raw_pts
        '''

        bottom = df_2022.N.min()
        top = df_2022.N.max()
        left = df_2022.E.min()
        right = df_2022.E.max()

        self.sc_qc = {'dtm1':f'({bottom-100000} < N < {top+100000}) & ({left-100000} < E < {right+100000}) & (0 < sd_correct_dtm1 < 8) & subset_te_flag >=4  & n_te_photons > 10 & slope < 45 & abs(tpi_9) < 12 & {self.era} < 8 and abs(df_dtm1_era5) < 10',
                    'dtm10':f'({bottom-100000} < N < {top+100000}) & ({left-100000} < E < {right+100000}) & (0 < sd_correct_dtm10 < 8) & subset_te_flag >=4 & n_te_photons > 10 & slope < 45 & abs(tpi_9) < 12 and {self.era} < 8 and abs(df_dtm10_era5) < 10',
                    'cop30':f'({bottom-100000} < N < {top+100000}) & ({left-100000} < E < {right+100000}) & (0 < sd_correct_cop30 < 8) & subset_te_flag >=4 & n_te_photons > 10 & slope < 45 & abs(tpi_9) < 12 and {self.era} < 8 and abs(df_cop30_era5) < 10',
                    'fab':f'({bottom-100000} < N < {top+100000}) & ({left-100000} < E < {right+100000}) & (0 < sd_correct_fab < 8) & subset_te_flag >=4 & n_te_photons > 10 & slope < 45 & abs(tpi_9) < 12 and {self.era} < 8 and abs(df_fab_era5) < 10'}

        df_2022['month'] = month

        df_2022 = self.regression_sc_sd_predict(dem='cop30',
                                                raw_pts=df_2022,
                                                v_x=['E', 'N','h_te_best_fit','slope', 'aspect', 'planc','profc','tpi','tpi_9','tpi_27','curvature','sd_era','wf_positive', 'wf_negative','month'],
                                                regressor_name='sd_cop30_abserror_250_10_qc_2022.json')
        df_2022 = self.regression_sc_sd_predict(dem='fab',
                                                raw_pts=df_2022,
                                                v_x=['E', 'N','h_te_best_fit','slope', 'aspect', 'planc','profc','tpi','tpi_9','tpi_27','curvature','sd_era','wf_positive', 'wf_negative','month'],
                                                regressor_name='sd_fab_abserror_250_10_qc_2022.json')
        df_2022 = self.regression_sc_sd_predict(dem='dtm10',
                                                raw_pts=df_2022,
                                                v_x=['E', 'N','h_te_best_fit','slope', 'aspect', 'planc','profc','tpi','tpi_9','tpi_27','curvature','sd_era','wf_positive', 'wf_negative','month'],
                                                regressor_name='sd_dtm10_abserror_250_10_qc_2022.json')
        df_2022 = self.regression_sc_sd_predict(dem='fab',
                                                raw_pts=df_2022,
                                                v_x=['E', 'N','h_te_best_fit','slope', 'aspect', 'planc','profc','tpi','tpi_9','tpi_27','curvature','sd_era','wf_positive', 'wf_negative','month'],
                                                regressor_name='sd_dtm1_abserror_250_10_qc_2022.json')
        return df_2022

    def validation_nve(self,
                       dem='dtm1',   
                       raw_pts=None,                                
                       regressor=None,
                       regressor_name='sd_dtm1_abserror_250_10.json',
                       df_raw_validation_list=None,
                       v_x = None,
                       nve='sd_nve_10',
                       fit_line=True,
                       ):
        '''
        validate the snow depth predictor by nve validtion dataset.
        '''

        # read raw_validation df
        if df_raw_validation_list:
            df_nve_08,df_nve_09 = self.generate_validation_df_nve()
        else: # load    
            df_nve_08 = pd.read_csv('df_nve_08.csv')
            df_nve_09 = pd.read_csv('df_nve_09.csv')
        
        df_nve_08['month'] = 4
        df_nve_09['month'] = 4

        p_list = []

        for df_nve in [df_nve_08,df_nve_09]:
            df_predicted = self.regression_sc_sd_predict(dem=dem,
                                            raw_pts=df_nve,
                                            v_x = v_x,
                                            regressor=regressor,
                                            regressor_name=regressor_name)
            if regressor is None:
                regressor= regressor_name

            fig,ax = plt.subplots(2,2,figsize=(12,10))

            #a = plot_df_era_scatter(df_predicted.query('sd_nve_10 > 0'), 'sd_nve_10', self.era, title='10 m NVE',ax=ax[0][0],x_lims=(0,8),y_lims=(0,8),fit_line=fit_line,text=True,vmax=None,bar=False)
            a = final_histogram(df_predicted.query(f'{nve} > 0')[f'{nve}'],df_predicted.query(f'{nve} > 0 and sd_predict_{dem}>0')[f'sd_predict_{dem}'],dH_ref=df_predicted.query(f'{nve} > 0')[f'df_predict_{dem}'],ax=ax[0][0],legend=['ALS Validation','Model','Local Variability'],range=(-4,10),window=(-4,10));
            b = plot_df_era_scatter(df_predicted.query(f'({nve} > 0) & 0 < sd_predict_{dem}'),f'{nve}',f'sd_predict_{dem}',title=f'{nve}',ax=ax[0][1],x_lims=(0,8),y_lims=(0,8),fit_line=fit_line,text=True,vmax=None,bar=False)
            c = plot_df_era_scatter(df_predicted.query(f'{nve} > 0'),'E',f'{nve}',ax=ax[1][0],y_lims=(0,8),fit_line=False,text=False,vmax=None,bar=False)
            d = plot_df_era_scatter(df_predicted.query(f'sd_predict_{dem} > 0'),'E',f'sd_predict_{dem}',ax=ax[1][1],y_lims=(0,8),fit_line=False,text=False,vmax=None,bar=False)
            p_list.append(fig)
    
        return p_list

    def generate_validation_df_nve(self):

        # List of file paths
        file_paths = [r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\data\senorge\sd_2008.nc', 
                    r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\data\senorge\sd_2009.nc',
                    r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\data\senorge\sd_2020.nc']

        # Load datasets
        se = [xr.open_dataset(fp) for fp in file_paths]
        # Concatenate along the time dimension
        se_all = xr.concat(se, dim='time')
        senorge = ERA5(se_all)
        
        # generate validation df
        df_era_08 = pd.read_csv(r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\snowdepth_national_20080401_interp_era.csv')
        sd_nve_08 = xdem.DEM(r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\data\nve_08_merge_m.tif')
        
        df_nve_08 = produce_validation_from_nve(df_era_08,sd_nve_08)
        df_nve_08['date'] = ('2008-04-01')
        df_nve_08 = senorge.coupling_dataframe_sde_by_month_senorge(df_nve_08,'%Y-%m-%d')
        df_nve_08.to_csv('df_nve_08.csv',index=False)


        df_era_09 = pd.read_csv(r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\snowdepth_national_20090401_interp_era.csv')
        sd_nve_09 = xdem.DEM(r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\data\nve_09_merge_m.tif')
        df_nve_09 = produce_validation_from_nve(df_era_09,sd_nve_09)
        df_nve_09['date'] = ('2009-04-01')
        df_nve_09 = senorge.coupling_dataframe_sde_by_month_senorge(df_nve_09,'%Y-%m-%d')
        df_nve_09.to_csv('df_nve_09.csv',index=False)
        
        return df_nve_08,df_nve_09


    def validation_nve_compare_with_raw(self,dem='dtm1',
                                             raw_pts=None, 
                                             regressor = None, 
                                             regressor_name=None, 
                                             v_x = None,
                                             nve='sd_nve_10'):
        '''
        compare the dtm1 after bias correction and dtm1_raw.
        '''

        if regressor_name is None:
            regressor_name= ['sd_dtm1_abserror_250_10_qc_no_canopy_nomonth_raw.json',
                             'sd_dtm1_abserror_250_10_qc_no_canopy_nomonth.json']
    
        if v_x is None:
            v_x=['E', 'N','h_te_best_fit','slope', 'aspect', 'planc','profc','tpi','tpi_9','tpi_27','curvature','segment_cover','sd_era','wf_positive', 'wf_negative','smlt_acc', 'sf_acc'],

        if raw_pts is None:
        # you need prepared the nve validation dataset
            raw_pts = pd.read_csv('df_nve_08.csv')

        # generate the difference with ERA5 'snowdepth_dtm1' - era
        coef = 1
        self.sc[f'df_{dem}_era5'] = self.sc[f'snowdepth_{dem}'] - self.sc['sd_era']* coef
        #self.sc.loc[self.sc['sd_era'] < 0.1, [f'df_{dem}_era5']] = 0

        # regression. Get df_predict_dtm1  and df_predict_dtm1_raw
        pts_nve = self.regression_sc_sd_predict(dem=dem,
                                                raw_pts=raw_pts,
                                                v_x=v_x,
                                                regressor=regressor[0],
                                                regressor_name = regressor_name[0],
                                                col=[f'df_predict_{dem}_raw',f'sd_predict_{dem}_raw'])

        self.sc[f'df_{dem}_era5'] = self.sc[f'sd_correct_{dem}'] - self.sc['sd_era']* coef
        #self.sc.loc[self.sc['sd_era'] < 0.1, [f'df_{dem}_era5']] = 0

        pts_nve = self.regression_sc_sd_predict(dem=dem,
                                                raw_pts=raw_pts,
                                                v_x=v_x,
                                                regressor=regressor[1],
                                                regressor_name = regressor_name[1],
                                                col=[f'df_predict_{dem}',f'sd_predict_{dem}'])
        
        # plot
        fig,ax = plt.subplots(2,2,figsize=(12,10))
        final_histogram(pts_nve.query(f'{nve} > 0')[f'{nve}'],
                        pts_nve.query(f'{nve} > 0 and sd_predict_{dem}>0')[f'sd_predict_{dem}'],
                        dH_ref=pts_nve.query(f'{nve} > 0')[f'df_predict_{dem}'],
                        title=f'{nve}',
                        ax=ax[0,0],legend=['ALS Validation','Model','Local Variability'],range=(-4,10),window=(-4,10));
        final_histogram(pts_nve.query(f'{nve} > 0')[f'{nve}'],
                        pts_nve.query(f'{nve} > 0 and sd_predict_{dem}_raw>0')[f'sd_predict_{dem}_raw'],
                        dH_ref=pts_nve.query(f'{nve} > 0')[f'df_predict_{dem}_raw'],
                        title=f'{nve} - raw',
                        ax=ax[0,1],legend=['ALS Validation','Model','Local Variability'],range=(-4,10),window=(-4,10));
        
        plot_df_era_scatter(pts_nve.query(f'({nve} > 0) & 0 < sd_predict_{dem}'),
                            f'{nve}',f'sd_predict_{dem}',
                            title=None,
                            ax=ax[1][0],x_lims=(0,8),y_lims=(0,8),fit_line=True,text=True,vmax=None,bar=False)
        plot_df_era_scatter(pts_nve.query(f'({nve} > 0) & 0 < sd_predict_{dem}_raw'),
                            f'{nve}',f'sd_predict_{dem}_raw',
                            title=None,
                            ax=ax[1][1],x_lims=(0,8),y_lims=(0,8),fit_line=True,text=True,vmax=None,bar=False)

        return pts_nve                                                

    def update_raw_and_plot(self,
                            regressor_list,
                            df_raw,
                            list_df_name = ['df_predict_dtm1'],
                            list_sd_name = ['sd_predict_dtm1'],
                            range=(0,10),clim=(0,3)):

        '''
        same with update_raw_and_validate, but plot it
        '''
        # predict
        df = df_raw
        df_predicted = regression_task(df, regressor_list, list_df_name = list_df_name,list_sd_name = list_sd_name,era=self.era)
        df_predicted['df_dtm1_era'] = df_predicted['sd_predict_dtm1'] - df_predicted[self.era]
        
        #final_histogram(df_predicted['sd_predict_dtm1'],df_predicted[self.era],dH_ref=df_predicted['df_dtm1_era'],legend=['ICESat-2 - DTM1','ERA5 Land','ICESat-2(DTM1) - ERA5 Land'],range=(-4,8),perc_t=100);
        snow_dtm1 = plot_point_map('sd_predict_dtm1',df_predicted.query(f'{range[0]} < sd_predict_dtm1 < {range[1]}'),title='Snow depth (DTM1)',clim=(clim),cmap='YlGnBu',sampling=0.01)
        snow_df = plot_point_map('df_dtm1_era',df_predicted.query(f'{range[0]} < sd_predict_dtm1 < {range[1]}'),title='Snow depth difference (DTM1 - ERA5 Land)',clim=(-1.5,1.5),cmap='bwr',sampling=0.01)
            
        return (snow_dtm1 + snow_df).cols(2)

    def update_sc(self, era_daily=None,era_montly=None):
        '''
        update sde and other predictors from daily and monthly ds for datraframe snowcover 
        '''

        if era_montly is None:
            era_montly = r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\data\EAR5_land\monthly_data_08_22.nc'
        # update others
        era = ERA5(era_montly)
        era.cal_wind_aspect_factor_yearly()
        era.coupling_dataframe_other_by_date(self.sc)

        if era_daily is None:
            era_daily = r'\\hypatia.uio.no\lh-mn-geofag-felles\projects\snowdepth\zhihaol\data\EAR5_land\era5.nc'
        # update sd_era
        era = ERA5(era_daily)
        self.sc = era.coupling_dataframe_sde_by_date(self.sc)


class ERA5:
    def __init__(self, filename):
        """
        Load an xarray dataset from a NetCDF file.
        """
        if isinstance(filename, xr.Dataset):
            self.ds = filename
        else:
            self.ds = xr.open_dataset(filename)
        
        # have calcukated wind-aspect factor
        self.wf = False

        self.time_list= ['2019-04-01', '2020-04-01', '2021-04-01',              
             '2022-01-01', '2022-02-01', '2022-03-01',              
             '2022-04-01', '2022-05-01']
        
    def coupling_dataframe_sde_by_date_senorge(self, df, ymd='%Y%m%d',interp_method='linear',target='snow_depth',new_column='sde_se', kwargs=None, **coords_kwargs):
        '''
        couple with dataframe by date. ONLY one variable 'sde' at daily resolution.
        '''
        
        # Group the DataFrame by date
        df['date_'] = pd.to_datetime(df['date'], format=ymd).copy()
        grouped_df = df.groupby('date_')

        for date, subset_df in tqdm(grouped_df):
            # Interpolate the data spatially using xarray
            ds = self.ds.sel(time=date,method='nearest')

            # direct
            ds_values = ds.sel(x=subset_df.E.to_xarray(),y=subset_df.N.to_xarray(), method='nearest')
            df.loc[df['date_'] == date, 'sd_se'] = ds_values[target].values / 100
            
            # interpolated
            interpolated_values = ds.interp(x=subset_df.E.to_xarray(),y=subset_df.N.to_xarray(), method=interp_method,kwargs=None, **coords_kwargs)
            df.loc[df['date_'] == date, new_column] = interpolated_values[target].values / 100
        
        print('Daily coupling - the last date:',date)
        return df
    
    def coupling_dataframe_sde_by_month_senorge(self, df, ymd='%Y%m%d',interp_method='linear',target='snow_depth',new_column='sde_se', kwargs=None, **coords_kwargs):
        '''
        couple with dataframe by date. ONLY one variable 'sde' at daily resolution.
        '''
        
        # Group the DataFrame by date
        df['date_'] = pd.to_datetime(df['date'], format=ymd).copy()
        # Mean over the month. The 1-31 reprense 1-15, so we need to + 30 days if you are using 1-01 represent the montly mean.
        month_mean = self.ds.resample(time='1M').mean()
        df['date_offset'] = df['date_'] + pd.Timedelta(days=30)
        # Group the DataFrame by date
        grouped_df = df.groupby('date_offset')

        for date, subset_df in tqdm(grouped_df):
            # Interpolate the data spatially using xarray
            ds = month_mean.sel(time=date,method='nearest')

            # direct
            ds_values = ds.sel(x=subset_df.E.to_xarray(),y=subset_df.N.to_xarray(), method='nearest')
            df.loc[df['date_offset'] == date, 'sd_se'] = ds_values[target].values / 100
            
            # interpolated
            interpolated_values = ds.interp(x=subset_df.E.to_xarray(),y=subset_df.N.to_xarray(), method=interp_method,kwargs=None, **coords_kwargs)
            df.loc[df['date_offset'] == date, new_column] = interpolated_values[target].values / 100
        
        print('Monthly coupling - the last date:',date)
        return df
    
    def coupling_dataframe_sde_by_date(self, df, ymd='%Y%m%d',interp_method='linear',target='sde',new_column='sde_era', kwargs=None, **coords_kwargs):
        '''
        couple with dataframe by date. 
        ONLY one variable 'sde' at daily resolution.
        '''
        
        # Group the DataFrame by date
        df['date_'] = pd.to_datetime(df['date'], format=ymd).copy()
        grouped_df = df.groupby('date_')

        for date, subset_df in tqdm(grouped_df):
            # Interpolate the data spatially using xarray
            ds = self.ds.sel(time=date,expver=1,method='nearest')

            # direct
            ds_values = ds.sel(latitude=subset_df.latitude.to_xarray(),longitude=subset_df.longitude.to_xarray(), method='nearest')
            df.loc[df['date_'] == date, 'sd_era'] = ds_values[target].values
            
            # interpolated
            interpolated_values = ds.interp(latitude=subset_df.latitude.to_xarray(),longitude=subset_df.longitude.to_xarray(), method=interp_method,kwargs=None, **coords_kwargs)
            df.loc[df['date_'] == date, new_column] = interpolated_values[target].values
        
        print('Daily coupling - the last date:',date)
        return df

    def coupling_dataframe_other_by_date(self, df, ymd='%Y%m%d',target = None, new_column = None, offset_days=15):
        
        '''
        couple with dataframe by date. Without interpolation, At montly resolution.
        '''
        
        if target is None:
            target = ['sde','wf_positive','wf_negative','smlt_acc','sf_acc']
        if new_column is None:
            new_column = ['sd_era','wf_positive','wf_negative','smlt_acc','sf_acc']
        if self.wf:
            # Group the DataFrame by date. As monthly dataset annoted by the first day of the month, so we need to - 15 days.
            # 
            df['date_offset'] = pd.to_datetime(df['date'], format=ymd) - pd.Timedelta(days=offset_days)

            grouped_df = df.groupby('date_offset')

            for date, subset_df in tqdm(grouped_df):

                # directly get the data
                ds = self.ds.sel(time=date,method='nearest')
                ds_values = ds.sel(latitude=subset_df.latitude.to_xarray(),longitude=subset_df.longitude.to_xarray(),aspect=subset_df.aspect.to_xarray(), method='nearest')
                df.loc[df['date_offset'] == date, new_column] = ds_values[target].to_array().values.T
            
            print('Monthly coupling - the last date:',date)
            return df

    def coupling_dataframe(self, df,date_list=None):
        '''
        Not in use. for all dates.
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

    def coupling_dataframe_single_date(self, df, date, ymd='%Y%m%d', interp_method='linear',disable=False):
        '''
        produce pts. all in one.
        couple with dataframe, interpolate sde.
        return ['sd_era','wf_positive','wf_negative','smlt_acc','sf_acc']]

        disable = True, disable the tqdm, which is useful when you have tqdm outside.
        '''
        
        if self.wf:
            # Read era monthyly, and get the nearest time step
            ds = self.ds.sel(time=date, method='nearest')

            df['date_'] = pd.to_datetime(df['date'], format=ymd)

            grouped_df = df.groupby('date_')

            for date_, subset_df in tqdm(grouped_df, disable=disable):

                # get values directly
                ds_ = ds.sel(latitude=subset_df.latitude.to_xarray(), longitude=subset_df.longitude.to_xarray(),aspect=subset_df.aspect.to_xarray(),method='nearest')
                df.loc[df['date_'] == date_, ['sd_era','wf_positive','wf_negative','smlt_acc','sf_acc']] = ds_[['sde','wf_positive','wf_negative','smlt_acc','sf_acc']].to_array().values.T
                
                # get values interpolated
                interpolated_values = ds.isel(aspect=1)[['sde']].interp(latitude=subset_df.latitude.to_xarray(), longitude=subset_df.longitude.to_xarray(),method=interp_method)
                df.loc[df['date_'] == date_, ['sde_era']] = interpolated_values['sde'].values

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

                # Update the values in the dacoupling_dataframe_other_by_datetaset
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

from xrspatial import hillshade
import matplotlib.cm as cm
from datashader.transfer_functions import shade
from datashader.transfer_functions import stack
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import imageio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import imageio
import datashader.transfer_functions as tf
import xarray as xr
from skimage.measure import label, regionprops
from skimage.segmentation import expand_labels
from xrspatial import slope
import numpy as np
import datetime


class Snow_Distributor:
    def __init__(self, 
                 new_df, 
                 era,
                 senorge=None,
                 regressor_list=None,
                 features=None,
                 crop=None,
                 x='E',
                 y='N'):
        '''
        crop should be min_y,max_y,min_x,max_x.
        features should be a list of features, and exactly the same with the training dataset.
        '''
        self.df = new_df
        if crop:
            self.crop_df(crop[0],crop[1],crop[2],crop[3],df=new_df,x=x,y=y)

        self.timeseries = None
        self.era = era

        if senorge is not None:
            self.se = senorge
        else:
            self.se = None

        if regressor_list is None:
            self.regressor_list = ['sd_dtm10_abserror_250_10_qc_nve_.json',
                                    'sd_dtm1_abserror_250_10_qc_nve_.json',
                                    'sd_cop30_abserror_250_10_qc_nve_.json',
                                    'sd_fab_abserror_250_10_qc_nve_.json']
        else:
            self.regressor_list = regressor_list
        if features is None:
            self.features = ['E', 'N','h_te_best_fit',
                             'slope', 'aspect', 'planc',
                             'profc','tpi','tpi_9','tpi_27',
                             'curvature','sde_era','wf_positive', 
                             'wf_negative','month']
        else:
            self.features = features

    def crop_df(self,min_y,max_y,min_x,max_x,df=None,x='E',y='N'):
        '''
        crop the dataframe:
            min_y,max_y,min_x,max_x
        '''
        
        if df is None:
            df = self.df
        
        self.df = df[(df[y] >= min_y) & (df[y] <= max_y) & (df[x] >= min_x) & (df[x] <= max_x)].copy()

    def offset_adjustment(self,
                          df,
                          dems=['dtm1','dtm10','cop30','fab'],
                          slope = [0.49,0.51,0.49,0.48],
                          intercept = [0.70,0.71,0.77,0.76]):
        '''
        pts_nve
        '''
        # 
        #slope = [0.49,0.51,0.49,0.48]
        #intercept = [0.70,0.71,0.77,0.76]
    
        # Compute the mean of the grouped values
        for dem,s,i in zip(dems,slope,intercept):
            df[f'sd_predict_{dem}_'] = (df[f'sd_predict_{dem}'] - i) / s
            df.loc[df[f'sd_predict_{dem}_']< 0, f'sd_predict_{dem}_'] = 0
            df.loc[df[f'sd_predict_{dem}_']> 10, f'sd_predict_{dem}_'] = 10
        
        return df

    def offset_adjustment_quantile_mapping(self,
                                           df,
                                           correction,
                                           df_nve_08=None,
                                           adjust_factor=None,
                                           dems=['sd_predict_dtm1', 'sd_predict_dtm10', 'sd_predict_cop30', 'sd_predict_fab']):
        
        # calculate quantiles if needed.
        if adjust_factor is None:
            adjust_factor = {}
            # df (df_nve_08) should has dems, and 'sd_nve_10' (calibration dataset)
            for i in dems:
                adjust_factor[i] = quantile_calculate(df_nve_08, nve='sd_nve_10', dem=i, split=0)

        # usage:
        for dem, factor in adjust_factor.items():
            if correction == 'qm':
                # quantile mapping update the result of df
                quantile_mapping(df, factor['delta_lt1'],factor['delta_gt1'], dem=dem, split=0)
            elif correction == 'qm_original':
                quantile_mapping_original(df, factor['delta_gt1'],factor['dem_q_gt1'],dem=dem, split=0)

        return df


    def predict_snow_depth(self,
                           date,
                           df=None,
                           correction='qm',
                           regression='quantile_regression',
                           era='sde_era') -> pd.DataFrame:
        '''
        (1) Couple with ERA5 to get the neccesary features from ERA5 month. 
        (2) Predict snow depth for single date by pre-trained regressors.
            - quantile_regression
            - gamma_regression
            - others 
        (3) Do correction.
        
        '''

        # Preparetion
        if df is None:
            df = self.df.copy()
        # Using the data to couple with era5
        df['date']= pd.to_datetime(date)

        if 'h_te_best_fit' not in df.columns:
            df['h_te_best_fit'] = df['z']
        if 'z' not in df.columns:
            df['z'] = df['h_te_best_fit']

        # (1) Couple with ERA5, get sde_era, wf_positive, wf_negative, smlt_acc, sf_acc
        df = self.era.coupling_dataframe_single_date(df,date=date,disable=True)

        # get senorge snow depth
        if self.se is not None:
            df = self.se.coupling_dataframe_sde_by_month_senorge(df)

        # (2) use pre-trained regressior to predict
        df['month'] = df['date'].dt.month.copy()
        df = regression_task(df,
                             list_regressor = self.regressor_list,
                             v_x = self.features,
                             regression=regression,
                             era=era
                             )

        # corrections: if the predict snow depth is below 0, let it be 0
        df.loc[df['sd_predict_dtm1'] < 0, 'sd_predict_fab'] = 0
        df.loc[df['sd_predict_dtm10'] < 0, 'sd_predict_fab'] = 0
        df.loc[df['sd_predict_fab'] < 0, 'sd_predict_fab'] = 0
        df.loc[df['sd_predict_cop30'] < 0, 'sd_predict_cop30'] = 0

        if correction in ['qm','qm_original']:
            # open quantiles
            with open('adjust_factor_08_case6.pkl', 'rb') as f:
                adjust_factor = pickle.load(f)
            # correct it. Useful for distribution error.
            df = self.offset_adjustment_quantile_mapping(df,correction,adjust_factor=adjust_factor)
        
        elif correction == 'linear':
            df = self.offset_adjustment(df)

        else:
            print('No correction applied.')

        return df

    def product_time_series(self,
                            start_date=None, 
                            end_date=None, 
                            dates=None,
                            correction='qm',
                            regression='quantile_regression',
                            era='sde_era') -> xr.Dataset:
        '''
        For timeseries.
        '''

        # time dimension
        if dates is None:
            if start_date is None:
                start_date = pd.to_datetime('20080101')
            
            if end_date is None:
                end_date = pd.to_datetime('20220401')
            
            # MS means data in the first day of the month
            dates = pd.date_range(start_date, end_date, freq='MS')

        data_arrays = []

        for date in tqdm(dates):
            
            # Convert the string into datetime
            if not isinstance(date, str):
                date = date.strftime('%Y%m%d')
            
            # Predict snow depth for the date
            df = self.predict_snow_depth(date=date,correction=correction,regression=regression,era=era)

            df['time'] = pd.to_datetime(date)

            # Extract relevant columns
            desired_columns = ['N', 'E', 'z','time','sd_predict_dtm1_', 
                               'sd_predict_dtm1','sd_predict_dtm1_75','sd_predict_dtm1_25',
                               'sd_predict_dtm1_95','sd_predict_dtm1_05','sd_predict_dtm10', 
                               'sd_predict_cop30','sd_predict_fab','sde_era','sde_se','slope', 
                               'aspect', 'curvature']
            
            columns_to_select = list(set(desired_columns) & set(df.columns))

            df_subset = df[columns_to_select]

            # Group by (N, E) and aggregate the data using mean or any other desired aggregation function
            df_grouped = df_subset.groupby(['N', 'E','time']).mean().reset_index()

            # Set 'N' and 'E' columns as index
            df_grouped = df_grouped.set_index(['N','E','time'])

            # Create an xarray Dataset
            da = xr.Dataset.from_dataframe(df_grouped)

            # Add the DataArray to the list
            data_arrays.append(da)

        # Combine the list of DataArrays into a single xarray Dataset
        self.timeseries = xr.concat(data_arrays, dim='time')

        # Reshape the Dataset to remove the time dimension for specific variables
        variables_to_reshape = ['z', 'slope', 'aspect', 'curvature']

        # Remove the time dimension and drop it from the specified variables
        for variable in variables_to_reshape:
            self.timeseries[variable] = self.timeseries[variable].sel(time=self.timeseries['time'][0]).drop('time')

        return self.timeseries

    def product_time_series_df(self, df=None, start_date=None, end_date=None) -> xr.Dataset:
        '''
        For timeseries.
        '''

        if df is None:
            df = self.df

        if start_date is None:
            start_date = pd.to_datetime('20080101')
        
        if end_date is None:
            end_date = pd.to_datetime('20220401')

        dates = pd.date_range(start_date, end_date, freq='MS')
        data_arrays = []

        for date in tqdm(dates):
            date_str = date.strftime('%Y%m%d')
            df = self.predict_snow_depth(date=date_str)
            df['time'] = date

            # Extract relevant columns
            columns = ['N', 'E', 'z','time', 'sd_predict_dtm1_', 'sd_predict_dtm10_', 'sd_predict_cop30_', 'sd_predict_fab_', 'slope', 'aspect', 'planc', 'profc', 'curvature']
            df_subset = df[columns]

            # Add the DataArray to the list
            data_arrays.append(df_subset)

        # Combine the list of DataArrays into a single xarray Dataset
        self.timeseries = xr.concat(data_arrays, dim='time')

        # Reshape the Dataset to remove the time dimension for specific variables
        variables_to_reshape = ['z', 'slope', 'aspect', 'planc', 'profc', 'curvature']
        # Remove the time dimension and drop it from the specified variables
        for variable in variables_to_reshape:
            self.timeseries[variable] = self.timeseries[variable].sel(time=self.timeseries['time'][0]).drop('time')

        return self.timeseries

    def plot_snow_depth(self,date,df=None):

        if df is None:
            df = self.df

        date = pd.to_datetime(date) 

        terrain = df.sel(time=date).z
        illuminated = hillshade(terrain)
        snow_depth = df.sel(time=date).sd_predict_dtm1_

        # Create the plot figure
        fig = plt.figure(figsize=(10, 6))

        # Plot the terrain, illuminated layer, and snow depth using stack
        img = tf.stack(
            shade(terrain, cmap=["black", "white"], how="linear"),
            shade(illuminated, cmap=["black", "white"], how="linear", alpha=200),
            shade(snow_depth, cmap=cm.get_cmap('Spectral'), how="linear", alpha=150, span=[0, 5])
        )

        # Add gridlines
        ax = plt.axes()
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

        # date
        t_formatted = date.dt.strftime('%Y-%m').item()

        # Set the plot title
        plt.title(f'Monthly Snow Depth ({t_formatted})')
        # Show the plot
        image = plt.imshow(img)
        plt.colorbar(image, label='Snow Depth')
        plt.show()

    def plot_snow_depth_gif(self,ds_timeseries,variable = 'sd_predict_dtm1_'):

        # Assuming you have an xarray Dataset named 'ds' with dimensions 'N', 'E', and 'time'

        # Create an empty list to store the images
        images = []

        # Iterate over the time dimension
        for t in ds_timeseries.time:
            # Extract the data for the specific time slice
            data_slice = ds_timeseries[variable].sel(time=t)

            # Create a map plot
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(1, 1, 1)

            # Plot the data on the map
            im = ax.pcolormesh(ds_timeseries.E, ds_timeseries.N, data_slice, cmap='Spectral',vmin=0,vmax=5)

            # Format the time for the plot title
            t_formatted = t.dt.strftime('%Y-%m').item()

            # Set the plot title
            plt.title(f"Monthly snow depth {t_formatted}")

            # Add a colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(variable)

            # Save the figure as an image
            filename = f"{t_formatted}.png"
            plt.savefig(filename, dpi=100)
            plt.close(fig)

            # Append the image to the list
            images.append(imageio.imread(filename))

        # Save the list of images as a GIF
        imageio.mimsave('animation.gif', images, duration=0.8)


    def plot_snow_depth_gif_3d(self,ds_timeseries,variable = 'sd_predict_dtm1_'):

        # Assuming you have an xarray Dataset named 'ds' with dimensions 'N', 'E', and 'time'

        # Create an empty list to store the images
        images = []

        terrain = ds_timeseries.z
        illuminated = hillshade(terrain)

        # Iterate over the time dimension
        for t in ds_timeseries.time:
            # Extract the data for the specific time slice
            data_slice = ds_timeseries[variable].sel(time=t)
  
            # Create a map plot

            # Plot the terrain, illuminated layer, and snow depth using stack
            img = tf.stack(
                shade(terrain, cmap=["black", "white"], how="linear"),
                shade(illuminated, cmap=["black", "white"], how="linear", alpha=100),
                shade(data_slice, cmap=cm.get_cmap('Spectral'), how="linear", alpha=180, span=[0, 5])
            )

            # Format the time for the plot title
            t_formatted = t.dt.strftime('%Y-%m').item()

            # Save the figure as an image
            filename = f"{t_formatted}.png"
   
            ds.utils.export_image(img, t_formatted, background='white', export_path='.')

            # Append the image to the list
            images.append(imageio.imread(filename))

        # Save the list of images as a GIF
        imageio.mimsave('animation_3d.gif', images, duration=0.8)


    def plot_snow_depth_gif_chart(self, ds_timeseries, variable='sd_predict_dtm1_'):
        # Create an empty list to store the images
        images = []


        # Plot scatter for each time snapshot
        for t in ds_timeseries.time:
            # Plot chart for mean value
            fig, ax = plt.subplots(figsize=(10, 3))

            mean_series = ds_timeseries[variable].mean(dim=['N', 'E'])
            mean_series.plot(x='time', ax=ax)
            ax.set_ylabel('Mean Snow depth (m)')

            plt.scatter(t, mean_series.sel(time=t), c='red', marker='o')

            # Format the time for the plot title
            t_formatted = t.dt.strftime('%Y-%m').item()
            ax.set_title(f'Monthly Snow Depth - {t_formatted}')

            # Save the figure as an image
            filename = f"{t_formatted}.png"
            plt.savefig(filename, dpi=100)
            plt.close(fig)

            # Append the image to the list
            images.append(imageio.imread(filename))

        # Save the list of images as a GIF
        imageio.mimsave('animation_chart.gif', images, duration=1)

    def to_tiff(self,date,variable,path="__elevation.tif"):
        # Assuming you have an xarray dataset named 'xr_snowdepth' with dimensions 'N', 'E', 'time', and variables 'z' and 'sd_predict_dtm1_'

        # Convert the elevation data to a raster

        dem = self.df.sel(time=date)[variable].rio
        dem.set_spatial_dims('E', 'N', inplace=True)
        dem.write_crs("EPSG:4326", inplace=True)
        dem.to_raster(raster_path=path, driver="COG")

    def get_lake_mask(self,timeseries=None,threshold_slope=0.45,threshold_size=175,distance=4) -> xr.DataArray:

        '''
        threshold_size # Define the minimum area criteria for lakes
        '''

        # Assuming you have an xarray dataset named 'xr_snowdepth' with dimensions 'N', 'E', and variables 'z' and 'sd_predict_dtm1_'
        if timeseries is None:
            timeseries = self.timeseries

        if 'slope' in timeseries:
            slope_data = timeseries.slope
        else:
            # Extract the elevation data
            dem = timeseries.z
            # Calculate the slope using xrspatial
            slope_data = slope(dem)

        # Create a binary mask for potential lake areas
        lake_mask = slope_data < threshold_slope

        # Label connected regions in the lake mask
        labeled_mask = label(lake_mask)
        labeled_mask = expand_labels(labeled_mask,distance=4)

        # Compute properties of labeled regions
        region_props = regionprops(labeled_mask)

        # Filter out regions that do not meet the minimum area criteria
        lake_regions = [region for region in region_props if region.area > threshold_size]

        # Create an empty mask
        combined_mask = np.zeros_like(lake_mask, dtype=bool)

        # Combine the lake masks into a single mask
        for region in lake_regions:
            combined_mask |= labeled_mask == region.label

        # Convert the mask back to an xarray DataArray
        lake_mask_da = xr.DataArray(combined_mask, coords={"E": timeseries.E, "N": timeseries.N}, dims=["N", "E"])
        
        return lake_mask_da

    def mask_out_lake(self,mask=None,
                      is_plot=False,
                      threshold_slope=0.6,
                      threshold_size=100,
                      variable=None):
        '''
        mask: Xarray dataarray
        '''

        if mask is None:
            mask = self.get_lake_mask(threshold_slope=threshold_slope,threshold_size=threshold_size)
        
        # Merge the lake mask into the original xarray dataset
        self.timeseries = xr.merge([mask.rename('lake_mask'),self.timeseries], compat='override')
        
        if variable is None:
            variable = [var_name for var_name in self.timeseries if var_name.startswith('sd_predict')]

        if is_plot:
            # Extract the elevation data
            dem = self.timeseries.z
            # Calculate the slope using xrspatial
            slope_data = slope(dem)
            
            # Plot the lake mask overlaying the slope data
            fig, ax = plt.subplots(figsize=(10, 6))
            slope_data.plot(ax=ax, cmap='viridis', vmin=0, vmax=2)
            self.timeseries.lake_mask.plot(ax=ax, cmap='Blues', alpha=0.6)
            plt.show()

        self.timeseries[variable] = self.timeseries[variable].where(self.timeseries['lake_mask']==0)

        return self.timeseries