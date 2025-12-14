import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# To clear notebook outputs before committing to git:
# jupyter nbconvert --to notebook --clear-output --inplace feature_analysis_dashboard.ipynb

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

######################### 1. DATA LOADING #########################
def load_sensor_data_with_glucose(folder_path):
    '''
    Load sensor data with glucose column
    
    Inputs:
        folder_path: Path to the folder containing the sensor data
        
    Output:
        combined: Combined sensor and glucose data
    '''
    # Get the sensor data
    sensor_data = []
    sensor_dir = Path(folder_path)
    
    # Load the sensor data
    for file in sensor_dir.glob("gmData-*.csv"):
        df = pd.read_csv(file)
        df['DATE'] = pd.to_datetime(df['DATE'])
        df['source_file'] = file.name
        sensor_data.append(df)
    
    # If no sensor data, return an empty dataframe
    if len(sensor_data) == 0:
        return pd.DataFrame()
    
    # Combine the sensor data
    combined = pd.concat(sensor_data, ignore_index = True)
    return combined

def _ensure_float32(series):
    '''
    Helper function to convert series to float32 if numeric
    
    Inputs:
        series: Series to convert
        
    Output:
        series: Converted series
    '''

    # If the series is numeric, convert it to float32
    if pd.api.types.is_numeric_dtype(series):
        return series.astype('float32')
    return series

######################### 2. FEATURE ENGINEERING #########################
def create_advanced_features(df):
    '''
    Create advanced features for the sensor data
    
    Inputs:
        df: DataFrame containing the sensor data
        
    Output:
        features_df: DataFrame containing the advanced features
    '''
    features_df = df.copy()
    
    if 'N0' in features_df.columns:
        initial_count = len(features_df)
        n0_min = features_df['N0'].min()
        n0_max = features_df['N0'].max()
        print(f">>> N0 range before filtering: [{n0_min:.1f}, {n0_max:.1f}]")
        
        negative_count = (features_df['N0'] < 0).sum()
        if negative_count > 0:
            print(f">>> Filtering {negative_count:,} rows with negative N0 values (garbage data)")
            features_df = features_df[features_df['N0'] >= 0].copy()
        
        dead_sensor_count = (features_df['N0'] <= 1.0).sum()
        if dead_sensor_count > 0:
            print(f">>> Filtering {dead_sensor_count:,} rows with N0 <= 1.0 (dead sensor)")
            features_df = features_df[features_df['N0'] > 1.0].copy()
        
        filtered_count = initial_count - len(features_df)
        if filtered_count > 0:
            print(f">>> Removed {filtered_count:,} rows ({100*filtered_count/initial_count:.1f}%)")
            print(f">>> Remaining: {len(features_df):,} rows")
    
    numeric_cols_existing = features_df.select_dtypes(include = [np.number]).columns
    for col in numeric_cols_existing:
        if col != 'glucose':
            if features_df[col].dtype == 'float64':
                features_df[col] = features_df[col].astype('float32')
    
    if 'DATE' in features_df.columns:
        features_df = features_df.sort_values('DATE').reset_index(drop = True)
    
    temp_cols = ['T1', 'T2', 'T3', 'T4', 'T5']
    available_temp_cols = []
    for col in temp_cols:
        if col in features_df.columns:
            available_temp_cols.append(col)
    
    if len(available_temp_cols) >= 3:
        features_df['temp_mean'] = features_df[available_temp_cols].mean(axis = 1)
        features_df['temp_std'] = features_df[available_temp_cols].std(axis = 1)
        features_df['temp_min'] = features_df[available_temp_cols].min(axis = 1)
        features_df['temp_max'] = features_df[available_temp_cols].max(axis = 1)
        features_df['temp_range'] = features_df['temp_max'] - features_df['temp_min']
        features_df['temp_median'] = features_df[available_temp_cols].median(axis = 1)
        features_df['temp_cv'] = features_df['temp_std'] / (features_df['temp_mean'] + 1e-10)
        features_df['temp_iqr'] = features_df[available_temp_cols].quantile(0.75, axis = 1) - features_df[available_temp_cols].quantile(0.25, axis = 1)
    
    if 'T1' in features_df.columns and 'T3' in features_df.columns:
        features_df['gradient_T3_T1'] = features_df['T3'] - features_df['T1']
    if 'T1' in features_df.columns and 'T5' in features_df.columns:
        features_df['gradient_T5_T1'] = features_df['T5'] - features_df['T1']
    if 'T2' in features_df.columns and 'T4' in features_df.columns:
        features_df['gradient_T4_T2'] = features_df['T4'] - features_df['T2']
    if 'T2' in features_df.columns and 'T3' in features_df.columns:
        features_df['gradient_T3_T2'] = features_df['T3'] - features_df['T2']
    if 'T3' in features_df.columns and 'T5' in features_df.columns:
        features_df['gradient_T5_T3'] = features_df['T5'] - features_df['T3']
    
    if 'gradient_T5_T1' in features_df.columns and 'temp_mean' in features_df.columns:
        features_df['gradient_T5_T1_normalized'] = features_df['gradient_T5_T1'] / (features_df['temp_mean'] + 1e-10)
    
    if 'temp_mean' in features_df.columns:
        for window in [3, 5, 10, 20, 30]:
            features_df[f'temp_mean_rolling_{window}'] = features_df['temp_mean'].rolling(window = window, min_periods = 1, center = True).mean()
            features_df[f'temp_std_rolling_{window}'] = features_df['temp_mean'].rolling(window = window, min_periods = 1, center = True).std()
            features_df[f'temp_min_rolling_{window}'] = features_df['temp_mean'].rolling(window = window, min_periods = 1, center = True).min()
            features_df[f'temp_max_rolling_{window}'] = features_df['temp_mean'].rolling(window = window, min_periods = 1, center = True).max()
            features_df[f'temp_range_rolling_{window}'] = features_df[f'temp_max_rolling_{window}'] - features_df[f'temp_min_rolling_{window}']
        
        for span in [5, 10, 20]:
            features_df[f'temp_mean_ema_{span}'] = features_df['temp_mean'].ewm(span = span, adjust = False).mean()
            features_df[f'temp_std_ema_{span}'] = features_df['temp_mean'].ewm(span = span, adjust = False).std()
        
        features_df['temp_mean_diff'] = features_df['temp_mean'].diff().fillna(0)

        if len(features_df) >= 2:
            features_df['temp_mean_gradient'] = np.gradient(features_df['temp_mean'].bfill().values)
        else:
            features_df['temp_mean_gradient'] = np.zeros(len(features_df))
        
        features_df['temp_mean_pct_change'] = features_df['temp_mean'].pct_change().fillna(0)
        features_df['temp_mean_acceleration'] = features_df['temp_mean_diff'].diff().fillna(0)
    
    if 'DATE' in features_df.columns:
        features_df['hour'] = features_df['DATE'].dt.hour
        features_df['minute'] = features_df['DATE'].dt.minute
        features_df['day_of_week'] = features_df['DATE'].dt.dayofweek
        features_df['time_of_day'] = features_df['hour'] + features_df['minute'] / 60.0
        
        # Cyclical encoding for time features (preserves cyclical nature)
        features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
        features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
        features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
        features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
    
    if 'N0' in features_df.columns:
        # Use rolling normalization for drift correction
        print(f">>> Creating drift-robust N0 features (rolling normalization)...")
        
        # Rolling normalization windows
        for window in [10, 30, 60, 120]:
            rolling_mean = features_df['N0'].rolling(window = window, min_periods = 1, center = True).mean()
            rolling_std = features_df['N0'].rolling(window = window, min_periods = 1, center = True).std()
            
            # Normalized N0
            features_df[f'N0_normalized_{window}'] = ((features_df['N0'] - rolling_mean) / (rolling_std + 1e-8))
            
            # Ratio to rolling mean (percentage deviation from local baseline)
            features_df[f'N0_ratio_to_rolling_{window}'] = (features_df['N0'] / (rolling_mean + 1e-8))
            
            # Deviation from rolling mean (normalized)
            features_df[f'N0_deviation_from_rolling_{window}'] = ((features_df['N0'] - rolling_mean) / (rolling_mean + 1e-8))
        
        features_df['N0_diff'] = features_df['N0'].diff().fillna(0)
        features_df['N0_diff_abs'] = np.abs(features_df['N0_diff'])
        
        # Percentage change (normalized rate of change)
        features_df['N0_pct_change'] = features_df['N0'].pct_change().fillna(0)
        features_df['N0_pct_change_abs'] = np.abs(features_df['N0_pct_change'])
        
        # Second derivative (acceleration)
        features_df['N0_accel'] = features_df['N0_diff'].diff().fillna(0)
        features_df['N0_accel_abs'] = np.abs(features_df['N0_accel'])
        
        # Gradient (smooth derivative)
        n0_filled = features_df['N0'].bfill().ffill().fillna(0)

        # Calculate the gradient if we have enough data points
        if len(features_df) >= 2:
            features_df['N0_gradient'] = np.gradient(n0_filled.values)
        else:
            features_df['N0_gradient'] = np.zeros(len(features_df))
        
        # Multi-period changes (capture trends over different timeframes)
        for period in [2, 5, 10, 20]:
            features_df[f'N0_pct_change_{period}'] = (features_df['N0'].pct_change(periods = period).fillna(0))
            features_df[f'N0_diff_{period}'] = (features_df['N0'].diff(periods = period).fillna(0))
        
        # Rolling statistics of gradients (smoothness indicators)
        for window in [5, 10]:
            features_df[f'N0_diff_rolling_std_{window}'] = features_df['N0_diff'].rolling(window = window, min_periods = 1).std()
            features_df[f'N0_pct_change_rolling_std_{window}'] = features_df['N0_pct_change'].rolling(window = window, min_periods = 1).std()
        
        # Direction changes (trend reversals)
        features_df['N0_diff_sign'] = np.sign(features_df['N0_diff'])
        features_df['N0_direction_change'] = ((features_df['N0_diff_sign'] != features_df['N0_diff_sign'].shift(1)).astype(float))
        
        if 'Tmeas' in features_df.columns:
            # Rolling normalization for Tmeas
            for window in [10, 30]:
                rolling_tmeas = features_df['Tmeas'].rolling(window = window, min_periods = 1, center = True).mean()
                features_df[f'Tmeas_normalized_{window}'] = ((features_df['Tmeas'] - rolling_tmeas) / (rolling_tmeas + 1e-8))
            
            # Rate of change
            features_df['Tmeas_diff'] = features_df['Tmeas'].diff().fillna(0)
            features_df['Tmeas_pct_change'] = features_df['Tmeas'].pct_change().fillna(0)
            
            # N0/Tmeas ratio (using normalized versions for drift resistance)
            if 'N0_normalized_30' in features_df.columns and 'Tmeas_normalized_30' in features_df.columns:
                features_df['N0_Tmeas_ratio_normalized'] = features_df['N0_normalized_30'] / (np.abs(features_df['Tmeas_normalized_30']) + 1e-8)
            
            # Ratio of changes
            n0_pct = features_df['N0_pct_change']
            tmeas_pct = features_df['Tmeas_pct_change']
            features_df['N0_Tmeas_pct_ratio'] = np.where(
                np.abs(tmeas_pct) > 1e-6,
                n0_pct / (tmeas_pct + 1e-8),
                0
            )
        
        if 'temp_mean' in features_df.columns and 'N0_diff' in features_df.columns:
            # N0 change per degree
            temp_diff = features_df['temp_mean'].diff().fillna(0)
            n0_diff = features_df['N0_diff']
            features_df['N0_sensitivity_to_temp'] = np.where(
                np.abs(temp_diff) > 0.01,
                n0_diff / (temp_diff + 1e-8),
                0
            )
        
    
    if 'N0_normalized_30' in features_df.columns and 'temp_range' in features_df.columns:
        features_df['N0_norm_temp_range_interaction'] = features_df['N0_normalized_30'] * features_df['temp_range']
    
    if 'N0_pct_change' in features_df.columns and 'temp_pct_change' in features_df.columns:
        features_df['N0_temp_pct_change_interaction'] = features_df['N0_pct_change'] * features_df['temp_pct_change']
    
    if 'SIGN/REF' in features_df.columns:
        features_df['signal_ref_ratio'] = features_df['SIGN/REF']
        features_df['signal_ref_log'] = np.log1p(features_df['SIGN/REF'])
        # Use normalized N0 instead of raw N0
        if 'N0_normalized_30' in features_df.columns:
            features_df['N0_signal_ref'] = features_df['N0_normalized_30'] * features_df['SIGN/REF']
    
    if 'TEMstatus' in features_df.columns:
        features_df['temp_stable'] = (features_df['TEMstatus'] == 0).astype(int)
        features_df['temp_stable_interaction'] = features_df['temp_stable'] * features_df['temp_range']
    
    if 'N0_normalized_30' in features_df.columns:
        features_df['N0_normalized_30_squared'] = features_df['N0_normalized_30'] ** 2
    
    if 'N0_pct_change' in features_df.columns:
        features_df['N0_pct_change_squared'] = features_df['N0_pct_change'] ** 2
    
    if 'temp_range' in features_df.columns:
        features_df['temp_range_squared'] = features_df['temp_range'] ** 2
    
    if len(available_temp_cols) >= 3:
        temp_data = features_df[available_temp_cols].values
        # Calculate the skewness
        temp_mean = np.nanmean(temp_data, axis = 1, keepdims = True)
        temp_std = np.nanstd(temp_data, axis = 1, keepdims = True) + 1e-10
        centered = temp_data - temp_mean
        features_df['temp_skewness'] = np.nanmean((centered / temp_std) ** 3, axis = 1)
        
        # Calculate the kurtosis
        features_df['temp_kurtosis'] = np.nanmean((centered / temp_std) ** 4, axis = 1) - 3
    
    grad_cols = []
    for col in ['gradient_T3_T1', 'gradient_T5_T1']:
        if col in features_df.columns:
            grad_cols.append(col)
    
    if len(grad_cols) >= 2:
        grad_sum = 0
        for col in grad_cols:
            grad_sum += features_df[col]**2
        features_df['temp_gradient_magnitude'] = np.sqrt(grad_sum)
    elif len(grad_cols) == 1:
        features_df['temp_gradient_magnitude'] = np.abs(features_df[grad_cols[0]])
    
    if 'N0_rolling_std_10' in features_df.columns and 'N0' in features_df.columns:
        std_safe = features_df['N0_rolling_std_10'].clip(lower = 1e-6)
        features_df['N0_snr'] = features_df['N0'] / (std_safe + 1e-10)
    
    if 'temp_std' in features_df.columns and 'temp_mean' in features_df.columns:
        cv_safe = features_df['temp_cv'].clip(lower = 1e-10)
        features_df['temp_stability_index'] = 1.0 / (cv_safe + 1e-10)
    
    if 'N0_diff' in features_df.columns:
        features_df['N0_momentum'] = features_df['N0_diff'].diff().fillna(0)
        features_df['N0_momentum_abs'] = np.abs(features_df['N0_momentum'])
    
    if 'N0_normalized_30' in features_df.columns:
        n0_norm = features_df['N0_normalized_30']
        n0_q25 = n0_norm.quantile(0.25)
        n0_q50 = n0_norm.quantile(0.50)
        n0_q75 = n0_norm.quantile(0.75)
        features_df['N0_norm_quantile_bin'] = pd.cut(
            n0_norm, 
            bins = [-np.inf, n0_q25, n0_q50, n0_q75, np.inf],
            labels = [0, 1, 2, 3]
        ).astype(float)

        # Distance from quantiles
        features_df['N0_norm_dist_from_q25'] = np.abs(n0_norm - n0_q25)
        features_df['N0_norm_dist_from_q50'] = np.abs(n0_norm - n0_q50)
        features_df['N0_norm_dist_from_q75'] = np.abs(n0_norm - n0_q75)
    
    if 'temp_mean' in features_df.columns:
        temp_q25 = features_df['temp_mean'].quantile(0.25)
        temp_q50 = features_df['temp_mean'].quantile(0.50)
        temp_q75 = features_df['temp_mean'].quantile(0.75)
        features_df['temp_mean_dist_from_q50'] = np.abs(features_df['temp_mean'] - temp_q50)
    
    if 'DATE' in features_df.columns:
        # Time since start
        if len(features_df) > 0:
            start_time = features_df['DATE'].min()
            features_df['time_since_start'] = (features_df['DATE'] - start_time).dt.total_seconds() / 3600.0  # hours
            features_df['time_since_start_days'] = features_df['time_since_start'] / 24.0
            # Cyclical encoding for time since start (captures periodic patterns)
            max_time = features_df['time_since_start'].max()
            if max_time > 0:
                features_df['time_since_start_sin'] = np.sin(2 * np.pi * features_df['time_since_start'] / (max_time + 1e-10))
                features_df['time_since_start_cos'] = np.cos(2 * np.pi * features_df['time_since_start'] / (max_time + 1e-10))
        
        # Time-based interactions
        if 'hour_sin' in features_df.columns:
            if 'N0_normalized_30' in features_df.columns:
                features_df['N0_norm_hour_sin'] = features_df['N0_normalized_30'] * features_df['hour_sin']
                features_df['N0_norm_hour_cos'] = features_df['N0_normalized_30'] * features_df['hour_cos']
            if 'temp_mean' in features_df.columns:
                features_df['temp_mean_hour_sin'] = features_df['temp_mean'] * features_df['hour_sin']
                features_df['temp_mean_hour_cos'] = features_df['temp_mean'] * features_df['hour_cos']
        
        # Time of day bins (is_morning, etc.) - redundant with cyclical encoding and can cause overfitting. Cyclical encoding (hour_sin/hour_cos) is sufficient.
    
    numeric_cols = []
    for col in features_df.columns:
        if col != 'glucose':
            if pd.api.types.is_numeric_dtype(features_df[col]):
                numeric_cols.append(col)
    
    total_inf_replaced = 0
    total_extreme_clipped = 0
    
    batch_size = 50
    for i in range(0, len(numeric_cols), batch_size):
        batch_cols = numeric_cols[i:i+batch_size]
        
        for col in batch_cols:
            col_data = features_df[col].values
            inf_mask = np.isinf(col_data)
            if inf_mask.any():
                inf_count = inf_mask.sum()
                total_inf_replaced += inf_count
                col_data[inf_mask] = 0
                features_df[col] = col_data
            
            col_data = features_df[col].values
            if col_data.size > 0 and not np.all(np.isnan(col_data)):
                q001 = np.nanquantile(col_data, 0.001)
                q999 = np.nanquantile(col_data, 0.999)
                extreme_mask = (col_data < q001) | (col_data > q999)
                if extreme_mask.any():
                    total_extreme_clipped += extreme_mask.sum()
                    col_data = np.clip(col_data, q001, q999)
                    features_df[col] = col_data
            
            if features_df[col].isna().any():
                features_df[col] = features_df[col].fillna(0)
            
            if features_df[col].dtype == 'float64':
                features_df[col] = features_df[col].astype('float32')
    
    if total_inf_replaced > 0 or total_extreme_clipped > 0:
        print(f">>> Cleaned features: {total_inf_replaced:,} infinity values replaced, {total_extreme_clipped:,} extreme values clipped")
    
    return features_df

######################### 3. FEATURE ANALYSIS #########################
def analyze_feature_correlations(X, y, feature_names):
    '''
    Analyze correlations between features and target variable
    
    Inputs:
        X: DataFrame containing the features
        y: Series containing the target variable
        feature_names: List of feature names
        
    Output:
        corr_df: DataFrame with correlation metrics
    '''
    # Create DataFrame with features and target
    analysis_df = pd.DataFrame(X, columns = feature_names)
    analysis_df['glucose'] = y.values if hasattr(y, 'values') else y
    
    # Calculate Pearson correlations
    correlations = analysis_df[feature_names].corrwith(analysis_df['glucose']).abs().sort_values(ascending = False)
    
    # Calculate Spearman correlations (rank-based, captures non-linear relationships)
    spearman_corrs = analysis_df[feature_names].corrwith(
        analysis_df['glucose'], method = 'spearman'
    ).abs().sort_values(ascending = False)
    
    corr_df = pd.DataFrame({
        'pearson_correlation': correlations,
        'spearman_correlation': spearman_corrs,
        'abs_pearson': correlations,
        'abs_spearman': spearman_corrs
    }).sort_values('abs_pearson', ascending = False)
    
    return corr_df

def perform_feature_selection(X, y, feature_names, corr_df, k_best = 25):
    '''
    Perform feature selection
    
    Inputs:
        X: DataFrame containing the features
        y: Series containing the target variable
        feature_names: List of feature names
        corr_df: DataFrame with correlation metrics
        k_best: Number of best features to select
        
    Output:
        top_features: List of the top k best features
        importance_df: DataFrame with the feature importance scores
    '''
    # Cleaning for Selection
    X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
    y_clean = y.fillna(y.mean()) # Fallback for target
    
    selector = SelectKBest(score_func = f_regression, k = 'all')
    selector.fit(X_clean, y_clean)

    f_scores = selector.scores_
    # Handle potential division by zero if scores are all 0
    denom = (f_scores.max() - f_scores.min())
    if denom == 0:
        denom = 1.0

    f_norm = (f_scores - f_scores.min()) / denom

    # Create the importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'f_score': f_scores,
        'f_score_norm': f_norm,
        'abs_pearson_corr': corr_df.loc[feature_names, 'abs_pearson']
    }).sort_values('f_score', ascending = False)

    top_features = []
    top_features_df = importance_df.head(k_best)
    for feature in top_features_df['feature']:
        top_features.append(feature)
    return top_features, importance_df