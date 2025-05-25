import os
import pandas as pd
import numpy as np
import logging
import optuna
import pickle
from datetime import datetime
import warnings
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from catboost import CatBoostRegressor, Pool
import smartsheet
import shap
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get environment variables or use defaults
SMARTSHEET_API_TOKEN = "wvPWOqoyipPIUdNIZD0IjyiERIRy6G57jxFd3"
SHEET1_ID = "53C423QHGgRhHHVc4MJ5vM3wRmr36CGF5jH9XQW1"
SHEET2_ID = "3c2cMpgC6cF3P2PGwc9Qpc4WH79Xr96P9jh4jmJ1"
SHEET3_ID = "9hX8fHWcM8wMG86q5rvXPrHX64m7f5WrHmGP5Hf1"
SHEET4_ID = "Cchp54454CVwQP3cvwmRGf9WmMpQCCCPVpFw4F71"
TASK_SHEET_ID = "v28pj8J2Wm7hMJxC6fPMCc7Xfgxv3CCRHq9cf681"
SKILLS_SHEET_ID = "3F8rgj3vpQ83p4jxwRfHr66Jjh44F3CG637grJ91"
METRICS_SHEET_ID = os.getenv('METRICS_SHEET_ID', None)  # Optional metrics sheet

#--------------------------------------------------------------------------
# Custom Transformer Classes
#--------------------------------------------------------------------------

class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Transformer to drop specified columns from a DataFrame
    """
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop if columns_to_drop is not None else []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        cols_to_drop = [col for col in self.columns_to_drop if col in X_transformed.columns]
        return X_transformed.drop(columns=cols_to_drop)


class DateTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, date_column='Date', n_splits=5, max_train_size=None, gap=0, window_type='expanding'):
        """
        Parameters:
            date_column (str): Name of the datetime column in X.
            n_splits (int): Number of splits.
            max_train_size (int or None): Max size of training window (only used in rolling window).
            gap (int): Number of days to exclude between train and test.
            window_type (str): 'expanding' (default) or 'rolling'
        """
        if window_type not in ['expanding', 'rolling']:
            raise ValueError("window_type must be 'expanding' or 'rolling'")
        self.date_column = date_column
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.gap = gap
        self.window_type = window_type

    def split(self, X, y=None, groups=None):
        if self.date_column not in X.columns:
            raise ValueError(f"{self.date_column} not in DataFrame columns")

        X_sorted = X.sort_values(self.date_column).reset_index()
        date_vals = X_sorted[self.date_column].drop_duplicates().values
        n_dates = len(date_vals)
        split_size = n_dates // (self.n_splits + 1)

        for i in range(1, self.n_splits + 1):
            test_start = i * split_size
            test_end = test_start + split_size
            if test_end > n_dates:
                break

            test_dates = date_vals[test_start:test_end]
            test_start_date = test_dates[0]
            test_end_date = test_dates[-1]

            # Test set mask
            test_mask = (
                (X_sorted[self.date_column] >= test_start_date) &
                (X_sorted[self.date_column] <= test_end_date)
            )
            test_indices = X_sorted[test_mask].index.values

            # Training end is before test_start - gap
            train_end_date = test_start_date - np.timedelta64(self.gap, 'D')
            train_mask = X_sorted[self.date_column] < train_end_date
            train_df = X_sorted[train_mask]

            if self.window_type == 'rolling' and self.max_train_size is not None:
                train_df = train_df.tail(self.max_train_size)

            train_indices = train_df.index.values

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class BaselineModel(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True, task_history=None, default_takt=None):
        self.fit_intercept = fit_intercept
        self.task_history = task_history
        self.default_takt = default_takt

    def fit(self, X, y=None):
        # Model doesn't learn anything from training data
        return self

    def predict_takt(self, task_details):
        product_assytype = task_details['PRODUCT-ASSY TYPE']
        task_date = task_details['Date']
        task = task_details['Task']

        df_history = self.task_history[
            (self.task_history['PRODUCT-ASSY TYPE'] == product_assytype) & 
            (self.task_history['Date'] < task_date) & 
            (self.task_history['Task'] == task)
        ]

        if len(df_history) > 0:
            return df_history['Total Adj Time Spent'].mean()

        df_default = self.default_takt[
            (self.default_takt['PRODUCT-ASSY TYPE'] == product_assytype) & 
            (self.default_takt['Task'] == task)
        ]

        if len(df_default) > 0:
            return df_default['Takt'].mean()

        return 0  # fallback if no historical or default data

    def predict(self, X):
        return np.array([self.predict_takt(row) for _, row in X.iterrows()])

#--------------------------------------------------------------------------
# Smartsheet Data Retrieval Functions
#--------------------------------------------------------------------------

def get_smartsheet_data(sheet_id: str, api_token: str) -> pd.DataFrame:
    """
    Retrieve data from a Smartsheet and convert to DataFrame
    """
    try:
        # Initialize client with API access token
        smart = smartsheet.Smartsheet(api_token)
        
        # Load the sheet
        sheet = smart.Sheets.get_sheet(sheet_id)
        
        # Extract column names
        columns = [col.title for col in sheet.columns]
        
        # Process rows
        rows = []
        for row in sheet.rows:
            cells = [
                cell.value if hasattr(cell, 'value') and cell.value is not None 
                else None 
                for cell in row.cells
            ]
            
            # Only append non-empty rows
            if any(cell is not None for cell in cells):
                rows.append(cells)
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=columns)
        
        logger.info(f"Successfully retrieved {len(df)} rows from sheet {sheet_id}")
        return df
    
    except Exception as e:
        logger.error(f"Error retrieving sheet {sheet_id}: {e}")
        raise

def melt_default_task(sheet_id: str, api_token: str) -> pd.DataFrame:
    """
    Melt Smartsheet data
    """
    # Retrieve sheet data
    df1 = get_smartsheet_data(sheet_id, api_token)
    
    # Identify value columns (exclude ID column)
    value_columns = [col for col in df1.columns if col != 'PRODUCT-ASSY TYPE']
    
    # Melt the DataFrame
    df_melted = df1.melt(
        id_vars='PRODUCT-ASSY TYPE',
        var_name='Task',
        value_name='Takt',
        value_vars=value_columns
    )
    
    # Clean and standardize data
    df_melted = df_melted.dropna()
    df_melted['PRODUCT-ASSY TYPE'] = df_melted['PRODUCT-ASSY TYPE'].str.upper().str.strip()
    df_melted['Task'] = df_melted['Task'].str.upper().str.strip()
    
    return df_melted

def process_smartsheet_data(
    api_token: str, 
    sheet1_id: str, 
    sheet2_id: str, 
    sheet3_id: str, 
    sheet4_id: str, 
    ml_sheet_id: str, 
    skills_sheet_id: str
):
    """
    Main function to process Smartsheet data
    """
    # Retrieve ML and Skills data
    df_ml = get_smartsheet_data(ml_sheet_id, api_token)
    df_skills = get_smartsheet_data(skills_sheet_id, api_token)
    
    # Merge and process ML data
    df_ml = df_ml.merge(df_skills, on='PIC', how='left')
    df_ml = df_ml.fillna(0)
    
    # Convert date
    df_ml['Date'] = pd.to_datetime(df_ml['Date'])
    
    # Melt multiple sheets
    df_melted = melt_default_task(sheet1_id, api_token)
    df_melted2 = melt_default_task(sheet2_id, api_token)
    df_melted3 = melt_default_task(sheet3_id, api_token)
    df_melted4 = melt_default_task(sheet4_id, api_token)
    
    # Concatenate melted dataframes
    df_all_melted = pd.concat([df_melted, df_melted2, df_melted3, df_melted4], ignore_index=True)
    
    return df_all_melted, df_ml, df_skills

#--------------------------------------------------------------------------
# Feature Engineering Functions
#--------------------------------------------------------------------------

def add_historical_performance(df_ml, months_lag=6):
    """
    Add historical performance metrics based on past data
    """
    df_ml['Date'] = pd.to_datetime(df_ml['Date'])
    df_ml = df_ml.sort_values('Date').reset_index(drop=True)
    
    # Create a copy to avoid modifying the original directly
    df_stats = df_ml.copy()
    
    # Sort by group and date
    if 'PIC' in df_stats.columns and 'ID' in df_stats.columns:
        df_stats = df_stats.sort_values(['ID', 'Task', 'PRODUCT-ASSY TYPE', 'Date', 'PIC'])
        group_cols = ['ID', 'Task', 'PRODUCT-ASSY TYPE', 'PIC']
    elif 'PIC' in df_stats.columns:
        df_stats = df_stats.sort_values(['Task', 'PRODUCT-ASSY TYPE', 'Date', 'PIC'])
        group_cols = ['Task', 'PRODUCT-ASSY TYPE', 'PIC']
    else:
        df_stats = df_stats.sort_values(['Task', 'PRODUCT-ASSY TYPE', 'Date'])
        group_cols = ['Task', 'PRODUCT-ASSY TYPE']
    
    # Initialize result columns
    df_stats['jobs_last_6m'] = np.nan
    df_stats['avg_takt_6m'] = np.nan
    df_stats['std_takt_6m'] = np.nan
    
    # Process group by group
    for group_key, group in df_stats.groupby(group_cols):
        for idx, row in group.iterrows():
            cutoff_date = row['Date'] - pd.DateOffset(months=months_lag)
            past_rows = group[(group['Date'] < row['Date']) & (group['Date'] >= cutoff_date)]
    
            df_stats.at[idx, 'jobs_last_6m'] = past_rows.shape[0]
            df_stats.at[idx, 'avg_takt_6m'] = past_rows['Total Adj Time Spent'].mean() if len(past_rows) > 0 else np.nan
            df_stats.at[idx, 'std_takt_6m'] = past_rows['Total Adj Time Spent'].std() if len(past_rows) > 1 else np.nan
    
    # Merge new features into the main dataset
    join_cols = group_cols.copy()
    if 'Date' not in join_cols:
        join_cols.append('Date')
        
    df_result = df_ml.merge(
        df_stats[join_cols + ['jobs_last_6m', 'avg_takt_6m', 'std_takt_6m']],
        on=join_cols,
        how='left'
    )
    
    # Create flag for rows with no history
    df_result['no_history_flag'] = df_result['jobs_last_6m'].isna().astype(int)
    
    # Fill NaNs in stats with 0
    df_result['jobs_last_6m'] = df_result['jobs_last_6m'].fillna(0)
    df_result['avg_takt_6m'] = df_result['avg_takt_6m'].fillna(0)
    df_result['std_takt_6m'] = df_result['std_takt_6m'].fillna(0)
    
    return df_result

def expand_ml_dataset(df_ml):
    """
    Expand the ML dataset with additional features
    """
    # Create a copy of the ML DataFrame
    df_ml_expanded = df_ml.copy()
    
    # Extract 'ASSY TYPE' and 'PARTS TYPE' from 'PRODUCT-ASSY TYPE'
    if 'PRODUCT-ASSY TYPE' in df_ml_expanded.columns:
        if not 'ASSY TYPE' in df_ml_expanded.columns:
            df_ml_expanded['ASSY TYPE'] = df_ml_expanded['PRODUCT-ASSY TYPE'].str.split('/').str[0].str.strip()
        
        if not 'PARTS TYPE' in df_ml_expanded.columns and df_ml_expanded['PRODUCT-ASSY TYPE'].str.contains('/').any():
            df_ml_expanded['PARTS TYPE'] = df_ml_expanded['PRODUCT-ASSY TYPE'].str.split('/').str[1].str.strip()
        elif not 'PARTS TYPE' in df_ml_expanded.columns:
            df_ml_expanded['PARTS TYPE'] = 'Unknown'
    
    # Feature engineering: Count number of sites and strips if applicable
    if 'SITE_TYPE' in df_ml_expanded.columns:
        df_ml_expanded['NUM_SITE'] = df_ml_expanded['SITE_TYPE'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) else 0
        )
    
    if 'STRIP' in df_ml_expanded.columns:
        df_ml_expanded['NUM_STRIP'] = df_ml_expanded['STRIP'].apply(
            lambda x: 1 if pd.notna(x) and str(x).upper() in ['YES', 'Y', 'TRUE', '1'] else 0
        )
    
    # Extract PN_PREFIX if not already present
    if 'PN' in df_ml_expanded.columns and 'PN_PREFIX' not in df_ml_expanded.columns:
        df_ml_expanded['PN_PREFIX'] = df_ml_expanded['PN'].astype(str).str[:3]
    
    return df_ml_expanded

#--------------------------------------------------------------------------
# Model Training Functions
#--------------------------------------------------------------------------

def root_mean_squared_error(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def prepare_data(df_ml, target_col, cat_cols, date_cutoff, drop_cols=None):
    """
    Prepare data for model training
    """
    if drop_cols is None:
        drop_cols = ['Date']
        
    # Sort and split data
    df = df_ml.sort_values('Date').reset_index(drop=True)
    df_train = df[df['Date'] < date_cutoff].copy()
    df_valid = df[df['Date'] >= date_cutoff].copy()
    
    # Process categorical columns
    category_mappings = {}
    for col in cat_cols:
        if col in df_train.columns:
            df_train[col] = df_train[col].astype('category')
            df_valid[col] = df_valid[col].astype('category')
            category_mappings[col] = dict(enumerate(df_train[col].cat.categories))
    
    # Extract target and features
    y_train = df_train[target_col]
    y_valid = df_valid[target_col]
    X_train = df_train.drop(columns=[target_col])
    X_valid = df_valid.drop(columns=[target_col])
    
    # Drop unwanted columns
    dropper = ColumnDropper(columns_to_drop=drop_cols)
    X_train_dropped = dropper.fit_transform(X_train)
    X_valid_dropped = dropper.transform(X_valid)
    
    # Get categorical feature names
    cat_features = [col for col in cat_cols if col in X_train_dropped.columns]
    
    return X_train_dropped, X_valid_dropped, y_train, y_valid, cat_features, dropper

def show_model_interpretation(model, X_valid, show_plots=False):
    """
    Show model interpretation using SHAP values
    """
    # Prepare data for SHAP
    X_display = X_valid.copy()
    
    # Encode categorical features for SHAP
    X_encoded = X_valid.copy()
    for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
        X_encoded[col] = X_encoded[col].astype('category').cat.codes
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_encoded)
    
    # Create feature importance DataFrame
    mean_abs_shap = pd.DataFrame({
        'Feature': X_display.columns,
        'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0)
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    
    # Only create plots if requested
    if show_plots:
        # SHAP summary plots
        shap.summary_plot(shap_values, X_display, plot_type='bar')
        shap.summary_plot(shap_values, X_display)
    
    return mean_abs_shap

def ml_hyperparameter_tuning(df_ml, target_col, cat_cols,
                           date_cutoff, n_splits=3, gap=0, window_type='expanding',
                           drop_cols=None, n_trials=50, show_plots=False):
    """
    Train CatBoost model with hyperparameter tuning
    """
    logger.info("Preparing data for model training")
    X_train_dropped, X_valid_dropped, y_train, y_valid, cat_features, dropper = prepare_data(
        df_ml, target_col, cat_cols, date_cutoff, drop_cols
    )
    
    # Create cross-validation splitter
    splitter = DateTimeSeriesSplit(
        date_column='Date',
        n_splits=n_splits,
        gap=gap,
        window_type=window_type
    )
    
    # Define Optuna objective
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "verbose": 0,
            "early_stopping_rounds": 20,
            "task_type": "CPU"
        }
        
        scores = []
        for train_idx, val_idx in splitter.split(df_ml[df_ml['Date'] < date_cutoff]):
            X_tr = X_train_dropped.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_te = X_train_dropped.iloc[val_idx]
            y_te = y_train.iloc[val_idx]
            
            train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
            valid_pool = Pool(X_te, y_te, cat_features=cat_features)
            
            model = CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            
            preds = model.predict(X_te)
            scores.append(r2_score(y_te, preds))
        
        return np.mean(scores)
    
    # Run Optuna search
    logger.info(f"Starting hyperparameter tuning with {n_trials} trials")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Best CV R2: {study.best_value}")
    cv_r2 = study.best_value
    
    # Train final model on full training data
    best_params = study.best_params
    best_params.update({
        "verbose": 0,
        "loss_function": "MAE",
        "early_stopping_rounds": 20,
        "task_type": "CPU"
    })
    
    logger.info("Training final model with best parameters")
    final_model = CatBoostRegressor(**best_params)
    train_pool = Pool(X_train_dropped, y_train, cat_features=cat_features)
    final_model.fit(train_pool, use_best_model=True)
    
    # Evaluate on validation set
    y_pred = final_model.predict(X_valid_dropped)
    rmse = root_mean_squared_error(y_valid, y_pred)
    mae = mean_absolute_error(y_valid, y_pred)
    mape = mean_absolute_percentage_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    
    logger.info(f"Validation RMSE: {rmse}")
    logger.info(f"Validation MAE: {mae}")
    logger.info(f"Validation MAPE: {mape}")
    logger.info(f"Validation R2: {r2}")
    
    # Generate model interpretation
    feature_importance = show_model_interpretation(final_model, X_valid_dropped, show_plots=show_plots)
    
    # Return results
    metrics_dict = {
        "cv_r2": cv_r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "feature_importance": feature_importance
    }
    
    return final_model, metrics_dict

#--------------------------------------------------------------------------
# Main Function
#--------------------------------------------------------------------------

def main():
    """
    Main function to run the model training pipeline
    """
    try:
        # Process Smartsheet data
        logger.info("Retrieving data from Smartsheet")
        df_all_melted, df_ml, df_skills = process_smartsheet_data(
            SMARTSHEET_API_TOKEN, 
            SHEET1_ID, 
            SHEET2_ID, 
            SHEET3_ID, 
            SHEET4_ID, 
            TASK_SHEET_ID, 
            SKILLS_SHEET_ID
        )
        
        # Expand the ML dataset
        logger.info("Expanding ML dataset")
        df_ml_expanded = expand_ml_dataset(df_ml)
        
        # Add historical performance features
        logger.info("Adding historical performance features")
        df_ml_expanded = add_historical_performance(df_ml_expanded, months_lag=6)
        
        # Prepare final dataset
        logger.info("Preparing final dataset")
        df_ml_dataset = df_ml_expanded[['Date', 'Task', 'ASSY TYPE', 'PARTS TYPE',
                                        'ORDER TYPE', 'SITE_TYPE', 'PN_PREFIX',
                                        'JAGUAR', 'WCSP', 'STRIP', 'BI', 'TI', 'NON TI', 
                                        'Designation',
                                        'CadExp_Enplas', 'Enplas_Years', 'CADWork_NonEnplas',
                                        'NonCadWork', 'Total',
                                        'NUM_SITE', 'NUM_STRIP',
                                        'jobs_last_6m', 'avg_takt_6m', 'std_takt_6m', 'no_history_flag',
                                        'Total Adj Time Spent']]
        
        # Define categorical columns
        cat_cols = ['Task', 'ASSY TYPE', 'PARTS TYPE',
                    'ORDER TYPE', 'SITE_TYPE', 'PN_PREFIX', 'Designation']
        
        # Handle missing values
        for col in df_ml_dataset.columns:
            if col in cat_cols:
                df_ml_dataset[col] = df_ml_dataset[col].fillna('Unknown').astype(str)
            elif col not in ['Date']:
                df_ml_dataset[col] = pd.to_numeric(df_ml_dataset[col], errors='coerce').fillna(0)
        
        # Define a date cutoff for validation (last 15% of data)
        dates = sorted(df_ml_dataset['Date'].unique())
        cut_idx = int(len(dates) * 0.85)
        date_cutoff = dates[cut_idx]
        logger.info(f"Using {date_cutoff} as cutoff date for validation")
        
        # Train model
        logger.info("Training CatBoost model")
        model, metrics = ml_hyperparameter_tuning(
            df_ml_dataset, 
            'Total Adj Time Spent', 
            cat_cols,
            date_cutoff, 
            n_splits=3, 
            gap=0, 
            window_type='expanding',
            n_trials=50,
            show_plots=False
        )
        
        # Save the model
        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"models/CatBoost_{timestamp}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved to {model_path}")
        
        # Print metrics
        logger.info("Training completed. Model metrics:")
        logger.info(f"  CV R2: {metrics['cv_r2']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.4f}")
        logger.info(f"  R2: {metrics['r2']:.4f}")
        
        # Top feature importance
        logger.info("Top 10 features by importance:")
        for i, (feature, importance) in enumerate(zip(metrics['feature_importance']['Feature'].head(10), 
                                                      metrics['feature_importance']['Mean_Abs_SHAP'].head(10))):
            logger.info(f"  {i+1}. {feature}: {importance:.4f}")
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()