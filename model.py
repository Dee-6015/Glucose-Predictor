#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

import feature_engineering as fe

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
# Enable deterministic operations for TensorFlow (slower but fully reproducible)
tf.config.experimental.enable_op_determinism()


##### 1. DATA LOADING #####

def load_and_patch_data(base_path = "."):
    '''
    Load sensor data from multiple sessions and merge with glucose reference data.
    Handles data cleaning and interpolation for missing glucose values.
    
    Inputs:
        base_path: String path to the base directory containing sensor and glucose data folders
        
    Output:
        merged_df: DataFrame with merged sensor and glucose data, sorted by date
    '''
    base_dir = Path(base_path)
    glucose_dir = base_dir / "Glucose Data"
    
    print(">>> Loading Sensor Data...")
    df1 = fe.load_sensor_data_with_glucose(base_dir / "Sensor Data 1_2")
    df2 = fe.load_sensor_data_with_glucose(base_dir / "Sensor Data 2_2")
    sensor_df = pd.concat([df1, df2], ignore_index = True).sort_values('DATE').reset_index(drop = True)
    
    print(">>> Merging Glucose Data...")
    glucose_data = []
    
    if glucose_dir.exists():
        for f in glucose_dir.glob("libra-*.csv"):
            g_df = pd.read_csv(f)
            if 'Date' in g_df.columns:
                g_df['DATE'] = pd.to_datetime(g_df['Date'])
            if 'glucos' in g_df.columns:
                g_df = g_df.rename(columns = {'glucos': 'glucose'})
            g_df = g_df.dropna(subset = ['DATE', 'glucose'])
            glucose_data.append(g_df[['DATE', 'glucose']])
    
    if not glucose_data:
        return sensor_df

    glucose_df = pd.concat(glucose_data, ignore_index = True).sort_values('DATE').reset_index(drop = True)
    merged_df = pd.merge_asof(sensor_df, glucose_df, on = 'DATE', direction = 'nearest', tolerance = pd.Timedelta(minutes = 15))

    if 'glucose' in merged_df.columns:
        merged_df = merged_df.set_index('DATE')
        merged_df['glucose'] = merged_df['glucose'].interpolate(method = 'time', limit = 15)
        merged_df = merged_df.reset_index()

    return merged_df

def calculate_mard(y_true, y_pred):
    '''
    Calculate Mean Absolute Relative Difference (MARD), a common metric for glucose prediction accuracy.
    Lower values indicate better performance.
    
    Inputs:
        y_true: Array of true/actual glucose values
        y_pred: Array of predicted glucose values
        
    Output:
        mard: Mean Absolute Relative Difference as a percentage
    '''
    relative_errors = np.abs((y_true - y_pred) / y_true)
    mard = np.mean(relative_errors) * 100
    return mard


def clarke_error_grid_zones(ref_values, pred_values):
    '''
    Calculate Clarke Error Grid zone percentages for clinical accuracy assessment.
    Zone A represents clinically accurate predictions, Zone B is acceptable.
    
    Inputs:
        ref_values: Array of reference/true glucose values
        pred_values: Array of predicted glucose values
        
    Output:
        zone_a: Percentage of predictions in Zone A (clinically accurate)
        zone_b: Percentage of predictions in Zone B (acceptable)
    '''
    ref_values = np.array(ref_values)
    pred_values = np.array(pred_values)
    diff = np.abs(ref_values - pred_values)
    
    zone_a = (diff <= 0.2 * ref_values) | ((ref_values < 70) & (diff <= 14))
    zone_b = (~zone_a) & (np.abs(ref_values - pred_values) < 50)
    
    total = len(ref_values)
    zone_a_pct = (np.sum(zone_a) / total) * 100
    zone_b_pct = (np.sum(zone_b) / total) * 100
    return zone_a_pct, zone_b_pct

##### 2. CHUNK SPLITTING #####

def create_time_chunks(df, chunk_hours = 1):
    '''
    Split the dataset into time-based chunks for proper train/test splitting.
    This ensures we don't mix training and testing data from the same time periods.
    
    Inputs:
        df: DataFrame with a 'DATE' column
        chunk_hours: Number of hours per chunk (default 1 hour)
        
    Output:
        df: DataFrame with added 'chunk_id' column indicating which time chunk each row belongs to
    '''
    print(f">>> Creating {chunk_hours}-hour chunks...")
    df = df.copy().sort_values('DATE')
    start = df['DATE'].min()
    df['chunk_id'] = ((df['DATE'] - start).dt.total_seconds() / 3600 // chunk_hours).astype(int)
    return df

##### 3. DATA PREP (FIXED: UNIVERSAL INPUTS + TARGET SCALING) #####

def create_universal_inputs(df, sequence_length = 30, scalers = None):
    """
    Creates inputs for both tree-based models (2D tabular) and deep learning models (3D sequences).
    Handles feature scaling and target normalization automatically.
    
    Inputs:
        df: DataFrame containing features and glucose target
        sequence_length: Length of sequences for deep learning models (default 30)
        scalers: Dictionary of fitted scalers (None for training, provided for testing)
        
    Output:
        X_seq: 3D array of sequences for deep learning (samples, sequence_length, features)
        X_tab: 2D array of tabular features for tree models
        y_out_scaled: Scaled target values for deep learning training
        y_out_raw: Raw target values for tree model training
        n_feats: Number of features used for deep learning
        scalers: Dictionary containing fitted scalers (tree, dl, y)
    """
    if scalers is None:
        mode = "Training"
    else:
        mode = "Testing"
    print(f">>> Generating Inputs ({mode}, Seq Len: {sequence_length})...")
    
    numeric_df = df.select_dtypes(include = [np.number])
    metadata = ['glucose', 'chunk_id', 'hours_elapsed', 'session_id']
    
    feature_cols = []
    for c in numeric_df.columns:
        if c not in metadata:
            feature_cols.append(c)
    
    dynamic_cols = []
    for c in feature_cols:
        if 'N0' in c or 'grad' in c or 'Tmeas' in c:
            dynamic_cols.append(c)

    if scalers is None:
        tree_scaler = RobustScaler()
        X_tree = tree_scaler.fit_transform(numeric_df[feature_cols].values.astype('float32'))
        
        dl_scaler = StandardScaler()
        X_dl = dl_scaler.fit_transform(numeric_df[dynamic_cols].values.astype('float32'))
        X_dl = np.clip(X_dl, -5.0, 5.0)

        scalers = {'tree': tree_scaler, 'dl': dl_scaler}
    else:
        X_tree = scalers['tree'].transform(numeric_df[feature_cols].values.astype('float32'))
        X_dl = scalers['dl'].transform(numeric_df[dynamic_cols].values.astype('float32'))
        X_dl = np.clip(X_dl, -5.0, 5.0)

    y_raw = numeric_df['glucose'].values.astype('float32').reshape(-1, 1)
    
    if 'y' not in scalers and mode == "Training":
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y_raw).flatten()
        scalers['y'] = y_scaler
    elif 'y' in scalers:
        y_scaled = scalers['y'].transform(y_raw).flatten()
    else:
        y_scaled = y_raw.flatten()

    n_samples = len(df) - sequence_length
    X_seq = np.zeros((n_samples, sequence_length, X_dl.shape[1]), dtype = 'float32')
    X_tab = X_tree[sequence_length:]
    y_out_scaled = y_scaled[sequence_length:]
    y_out_raw = y_raw[sequence_length:].flatten()
    
    chunk = 50000
    for i in range(0, n_samples, chunk):
        end = min(i + chunk, n_samples)
        for j in range(i, end):
            X_seq[j] = X_dl[j:j+sequence_length]

    return X_seq, X_tab, y_out_scaled, y_out_raw, X_dl.shape[1], scalers

##### 4. BASE MODELS #####

def build_transformer(input_shape):
    '''
    Build a transformer model with multi-head attention for sequence learning.
    Uses attention mechanism to learn relationships across time steps.
    
    Inputs:
        input_shape: Tuple of (sequence_length, num_features)
        
    Output:
        model: Compiled Keras transformer model ready for training
    '''
    inputs = keras.Input(shape = input_shape)
    x = layers.Dense(64)(inputs)
    x = layers.LayerNormalization(epsilon = 1e-6)(x)
    
    attn = layers.MultiHeadAttention(num_heads = 4, key_dim = 32)(x, x)
    attn = layers.Dropout(0.2)(attn)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization(epsilon = 1e-6)(x)
    
    ff = layers.Conv1D(filters = 64, kernel_size = 1, activation = "relu")(x)
    ff = layers.Dropout(0.2)(ff)
    x = layers.Add()([x, ff])
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation = "relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs, outputs, name = "Transformer")
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001, clipnorm = 1.0), loss = 'mse')
    return model


def build_lstm(input_shape):
    '''
    Build a bidirectional LSTM model for sequence learning.
    Bidirectional allows the model to see both past and future context.
    
    Inputs:
        input_shape: Tuple of (sequence_length, num_features)
        
    Output:
        model: Compiled Keras LSTM model ready for training
    '''
    inputs = keras.Input(shape = input_shape)
    x = layers.BatchNormalization()(inputs)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences = False, dropout = 0.2))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation = "relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs, outputs, name = "LSTM")
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001, clipnorm = 1.0), loss = 'mse')
    return model


def build_cnn(input_shape):
    '''
    Build a CNN model using 1D convolutions for sequence learning.
    Convolutions can detect local patterns in the time series.
    
    Inputs:
        input_shape: Tuple of (sequence_length, num_features)
        
    Output:
        model: Compiled Keras CNN model ready for training
    '''
    inputs = keras.Input(shape = input_shape)
    x = layers.BatchNormalization()(inputs)
    x = layers.Conv1D(64, 3, activation = 'relu', padding = 'same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation = 'relu', padding = 'same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation = "relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs, outputs, name = "CNN")
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.001, clipnorm = 1.0), loss = 'mse')
    return model

##### 5. MAIN PIPELINE #####

def train_and_evaluate(base_path = "."):
    '''
    Main training and evaluation pipeline for glucose prediction models.
    Trains multiple models, evaluates them, creates ensembles, and generates visualizations.
    
    Inputs:
        base_path: String path to base directory containing data folders
        
    Output:
        None (saves plots and prints results to console)
    '''
    print(f"\n{'#'*67}\n1. PREPARING DATA\n{'#'*67}")
    raw_df = load_and_patch_data(base_path)
    if raw_df.empty:
        return

    # Create images directory if it doesn't exist
    images_dir = Path(base_path) / "images"
    images_dir.mkdir(exist_ok = True)

    df = fe.create_advanced_features(raw_df)
    df = df.dropna(subset = ['glucose'])
    df = df[(df['glucose'] >= 30) & (df['glucose'] <= 500)]

    if 'N0' in df.columns:
        df = df[df['N0'] > 0]

    df = create_time_chunks(df, chunk_hours = 1)
    splitter = GroupShuffleSplit(n_splits = 1, train_size = 0.8, random_state = 42)
    train_idx, test_idx = next(splitter.split(df, groups = df['chunk_id']))
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

    # Sort test data by time for proper temporal ordering
    df_test = df_test.sort_values('DATE').reset_index(drop=True)
    print(f">>> Train: {len(df_train):,} | Test: {len(df_test):,}")

    print(f"\n{'#'*67}\n2. UNIVERSAL INPUTS (FIXED)\n{'#'*67}")
    X_seq_tr, X_tab_tr, y_tr_scaled, y_tr_raw, n_feats, scalers = create_universal_inputs(df_train)

    numeric_df_train = df_train.select_dtypes(include = [np.number])
    metadata = ['glucose', 'chunk_id', 'hours_elapsed', 'session_id']
    feature_cols = []
    for c in numeric_df_train.columns:
        if c not in metadata:
            feature_cols.append(c)

    X_seq_te, X_tab_te, _, y_te_raw, _, _ = create_universal_inputs(df_test, scalers = scalers)

    print(f"\n{'#'*67}\n3. TRAINING MODELS\n{'#'*67}")
    preds = {}
    es = callbacks.EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)

    print(f"\n{'#'*67}\n4. INDIVIDUAL MODEL RESULTS\n{'#'*67}")
    print(f"{'Model':<20} {'MAE':<10} {'R2':<10} {'MARD':<10} {'Status':<10}")
    print("-" * 70)
    
    print(">>> A. Training Gradient Boosting...")
    gb = HistGradientBoostingRegressor(max_iter = 1000, learning_rate = 0.05, max_depth = 8, l2_regularization = 5.0, random_state = 42)
    gb.fit(X_tab_tr, y_tr_raw)
    preds['GB'] = gb.predict(X_tab_te)
    
    min_len = min(len(preds['GB']), len(y_te_raw))
    gb_pred_aligned = preds['GB'][-min_len:]
    y_eval_aligned = y_te_raw[-min_len:]
    gb_mae = mean_absolute_error(y_eval_aligned, gb_pred_aligned)
    gb_r2 = r2_score(y_eval_aligned, gb_pred_aligned)
    gb_mard = calculate_mard(y_eval_aligned, gb_pred_aligned)
    print(f"{'GB':<20} {gb_mae:<10.2f} {gb_r2:<10.4f} {gb_mard:<10.2f}% {'Done':<10}")

    print(">>> B. Training Extra Trees...")
    et = ExtraTreesRegressor(n_estimators = 50, min_samples_leaf = 50, n_jobs = 1, random_state = 42)
    et.fit(X_tab_tr, y_tr_raw)
    preds['ET'] = et.predict(X_tab_te)
    
    min_len = min(len(preds['ET']), len(y_te_raw))
    et_pred_aligned = preds['ET'][-min_len:]
    y_eval_aligned = y_te_raw[-min_len:]
    et_mae = mean_absolute_error(y_eval_aligned, et_pred_aligned)
    et_r2 = r2_score(y_eval_aligned, et_pred_aligned)
    et_mard = calculate_mard(y_eval_aligned, et_pred_aligned)
    print(f"{'ET':<20} {et_mae:<10.2f} {et_r2:<10.4f} {et_mard:<10.2f}% {'Done':<10}")

    dl_models = [('Transformer', build_transformer), ('LSTM', build_lstm), ('CNN', build_cnn)]
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 2, min_lr = 1e-6, verbose = 0)
    
    for name, builder in dl_models:
        print(f">>> C. Training {name}...")
        model = builder((30, n_feats))
        model.fit(X_seq_tr, y_tr_scaled, validation_split = 0.1, epochs = 15, batch_size = 256, callbacks = [es, lr_scheduler], verbose = 0)

        scaled_pred = model.predict(X_seq_te, verbose = 0)
        preds[name] = scalers['y'].inverse_transform(scaled_pred).flatten()

        min_len = min(len(preds[name]), len(y_te_raw))
        dl_pred_aligned = preds[name][-min_len:]
        y_eval_aligned = y_te_raw[-min_len:]
        dl_mae = mean_absolute_error(y_eval_aligned, dl_pred_aligned)
        dl_r2 = r2_score(y_eval_aligned, dl_pred_aligned)
        dl_mard = calculate_mard(y_eval_aligned, dl_pred_aligned)
        print(f"{name:<20} {dl_mae:<10.2f} {dl_r2:<10.4f} {dl_mard:<10.2f}% {'Done':<10}")

    print(f"\n{'#'*67}\n5. INDIVIDUAL RESULTS SUMMARY\n{'#'*67}")
    print(f"{'Model':<20} {'MAE':<10} {'R2':<10} {'MARD':<10}")
    print("-" * 55)
    
    pred_lengths = []
    for p in preds.values():
        pred_lengths.append(len(p))
    min_len = min(pred_lengths)
    y_eval = y_te_raw[-min_len:]
    
    aligned = {}
    for k, v in preds.items():
        aligned[k] = v[-min_len:]
    
    for name, p in aligned.items():
        mae = mean_absolute_error(y_eval, p)
        r2 = r2_score(y_eval, p)
        mard = calculate_mard(y_eval, p)
        print(f"{name:<20} {mae:<10.2f} {r2:<10.4f} {mard:<10.2f}%")

    print(f"\n{'#'*67}\n6. ENSEMBLE COMBINATIONS\n{'#'*67}")
    combos = {}
    combos["1. Gradient Boosting + Extra Trees"] = (aligned['GB'] + aligned['ET']) /2
    combos["2. Gradient Boosting + LSTM + Transformer"] = (aligned['GB'] + aligned['LSTM'] + aligned['Transformer']) / 3
    combos["3. Extra Trees + CNN + Gradient Boosting"] = (aligned['ET'] + aligned['CNN'] + aligned['GB']) / 3
    combos["4. LSTM + Transformer + CNN"] = (aligned['LSTM'] + aligned['Transformer'] + aligned['CNN']) / 3
    combos["5. Gradient Boosting + Extra Trees + Transformer"] = (aligned['GB'] + aligned['ET'] + aligned['Transformer']) / 3

    best_score = -float('inf')
    best_name = ""
    best_preds = None
    print(f"{'Combination':<35} {'MAE':<10} {'R2':<10} {'MARD':<10}")
    print("-" * 70)
    for name, p in combos.items():
        r2 = r2_score(y_eval, p)
        mae = mean_absolute_error(y_eval, p)
        mard = calculate_mard(y_eval, p)
        print(f"{name:<35} {mae:<10.2f} {r2:<10.4f} {mard:<10.2f}%")
        if r2 > best_score:
            best_score = r2
            best_name = name
            best_preds = p

    print("-" * 70)
    print(f"WINNER: {best_name}")

    print(">>> Creating prediction error plot...")
    _, axes = plt.subplots(1, 2, figsize = (14, 6))

    ax1 = axes[0]
    errors = best_preds - y_eval
    abs_errors = np.abs(errors)

    sample_size = min(5000, len(y_eval))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(y_eval), sample_size, replace = False)
    y_sample = y_eval[sample_idx]
    pred_sample = best_preds[sample_idx]
    error_sample = errors[sample_idx]
    
    scatter = ax1.scatter(y_sample, pred_sample, c = abs_errors[sample_idx], cmap = 'RdYlGn_r', alpha = 0.6, s = 20, vmin = 0, vmax = 50)
    ax1.plot([y_eval.min(), y_eval.max()], [y_eval.min(), y_eval.max()], 'k--', lw = 2, label = 'Perfect Prediction')
    ax1.set_xlabel('Reference Glucose (mg/dL)', fontsize = 11)
    ax1.set_ylabel('Predicted Glucose (mg/dL)', fontsize = 11)
    ax1.set_title(f'Prediction Accuracy\n(MARD: {calculate_mard(y_eval, best_preds):.2f}%)', fontsize = 12, fontweight = 'bold')
    ax1.legend()
    ax1.grid(True, alpha = 0.3)
    plt.colorbar(scatter, ax = ax1, label = 'Absolute Error (mg/dL)')
    
    ax2 = axes[1]
    ax2.scatter(y_sample, error_sample, alpha = 0.5, s = 20, c = 'steelblue')
    ax2.axhline(y = 0, color = 'k', linestyle = '--', lw = 2)
    ax2.axhline(y = 20, color = 'r', linestyle = ':', lw = 1.5, label = '±20 mg/dL')
    ax2.axhline(y = -20, color = 'r', linestyle = ':', lw = 1.5)
    ax2.set_xlabel('Reference Glucose (mg/dL)', fontsize = 11)
    ax2.set_ylabel('Prediction Error (mg/dL)', fontsize = 11)
    ax2.set_title('Residual Analysis', fontsize = 12, fontweight = 'bold')
    ax2.legend()
    ax2.grid(True, alpha = 0.3)
    
    plt.tight_layout()
    plt.savefig(images_dir / 'clarke_grid.png', dpi = 300, bbox_inches = 'tight')
    plt.close()
    print(">>> Saved images/clarke_grid.png")

    print(">>> Creating model comparison plot...")
    model_names = list(aligned.keys())
    
    model_maes = []
    for name in model_names:
        model_maes.append(mean_absolute_error(y_eval, aligned[name]))
    model_r2s = []
    for name in model_names:
        model_r2s.append(r2_score(y_eval, aligned[name]))
    model_mards = []
    for name in model_names:
        model_mards.append(calculate_mard(y_eval, aligned[name]))
    
    fig, axes = plt.subplots(1, 3, figsize = (16, 5))
    
    ax1 = axes[0]
    bars1 = ax1.bar(model_names, model_maes, color = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
    ax1.set_ylabel('Mean Absolute Error (mg/dL)', fontsize = 11)
    ax1.set_title('MAE Comparison', fontsize = 12, fontweight = 'bold')
    ax1.tick_params(axis = 'x', rotation = 45)
    ax1.grid(True, alpha = 0.3, axis = 'y')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha = 'center', va = 'bottom', fontsize = 9)
    
    ax2 = axes[1]
    bars2 = ax2.bar(model_names, model_r2s, color = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
    ax2.set_ylabel('R2 Score', fontsize = 11)
    ax2.set_title('R2 Comparison', fontsize = 12, fontweight = 'bold')
    ax2.tick_params(axis = 'x', rotation = 45)
    ax2.grid(True, alpha = 0.3, axis = 'y')
    ax2.set_ylim([0, 1])
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha = 'center', va = 'bottom', fontsize = 9)
    
    ax3 = axes[2]
    bars3 = ax3.bar(model_names, model_mards, color = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
    ax3.set_ylabel('MARD (%)', fontsize = 11)
    ax3.set_title('MARD Comparison', fontsize = 12, fontweight = 'bold')
    ax3.tick_params(axis = 'x', rotation = 45)
    ax3.grid(True, alpha = 0.3, axis = 'y')
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha = 'center', va = 'bottom', fontsize = 9)
    
    plt.tight_layout()
    plt.savefig(images_dir / 'model_comparison.png', dpi = 300, bbox_inches = 'tight')
    plt.close()
    print(">>> Saved images/model_comparison.png")
    
    # Winner time series plot
    plt.figure(figsize = (14, 6))

    # Calculate local variance to find an interesting segment
    window_size = 100
    n_samples = min(1000, len(y_eval))

    # Try to find a segment with more glucose variation
    best_start = 0
    best_variance = 0
    for start in range(0, len(y_eval) - n_samples, 100):
        segment_var = np.var(y_eval[start:start + n_samples])
        if segment_var > best_variance:
            best_variance = segment_var
            best_start = start

    # Use the segment with most variation
    if best_variance < 100:
        best_start = max(0, len(y_eval) // 2 - n_samples // 2)

    plot_slice = slice(best_start, best_start + n_samples)
    x_axis = np.arange(n_samples)

    plt.plot(x_axis, y_eval[plot_slice], label = 'Reference', color = 'black', alpha = 0.7, linewidth = 2)
    plt.plot(x_axis, best_preds[plot_slice], label = f'Winner ({best_name})', color = 'blue', alpha = 0.8, linewidth = 1.5)

    # Add error band
    plt.fill_between(x_axis, y_eval[plot_slice], best_preds[plot_slice], alpha = 0.2, color = 'red', label = 'Error')
    plt.title(f"Tournament Winner: {best_name}\n(MARD: {calculate_mard(y_eval, best_preds):.2f}%, "f"MAE: {mean_absolute_error(y_eval, best_preds):.1f} mg/dL)", fontsize=13, fontweight='bold')
    plt.xlabel('Sample Index', fontsize = 11)
    plt.ylabel('Glucose (mg/dL)', fontsize = 11)
    plt.legend(fontsize = 10, loc = 'best')
    plt.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(images_dir / 'tournament_winner.png', dpi = 300, bbox_inches = 'tight')
    plt.close()
    
    zone_a, zone_b = clarke_error_grid_zones(y_eval, best_preds)
    print(f">>> Zone A: {zone_a:.1f}% | Zone A+B: {zone_a+zone_b:.1f}%")
    print("\n>>> Saved images/tournament_winner.png")

if __name__ == "__main__":
    train_and_evaluate(".")
