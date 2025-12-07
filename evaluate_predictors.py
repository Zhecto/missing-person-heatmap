"""
Comprehensive Prediction Model Evaluation with Hyperparameter Tuning
Evaluates Random Forest, Gradient Boosting, and Poisson Regression models.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    classification_report, accuracy_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from core.ingestion.data_loader import DataLoader
from core.preprocessing.data_cleaner import DataCleaner

# Configuration
DATA_PATH = "data/sample_data.csv"
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("🔬 PREDICTION MODEL EVALUATION WITH HYPERPARAMETER TUNING")
print("=" * 80)
print(f"Dataset: {DATA_PATH}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 80)

# ==============================================================================
# PART 1: DATA LOADING & PREPROCESSING
# ==============================================================================
print("\n📂 PART 1: Loading and preparing data...")
print("-" * 80)

loader = DataLoader()
cleaner = DataCleaner()

# Load data
df = loader.load_csv(DATA_PATH)
print(f"✓ Loaded {len(df)} records")

# Clean data
df = cleaner.handle_missing_values(df, strategy='smart')
print(f"✓ Cleaned: {len(df)} records remain")

# Prepare temporal features
df['Date Reported Missing'] = pd.to_datetime(df['Date Reported Missing'], errors='coerce')
df['Year'] = df['Date Reported Missing'].dt.year
df['Month'] = df['Date Reported Missing'].dt.month
df['Quarter'] = df['Date Reported Missing'].dt.quarter
df['DayOfWeek'] = df['Date Reported Missing'].dt.dayofweek

# Cyclical encoding
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

# Encode gender
df['Gender_Encoded'] = LabelEncoder().fit_transform(df['Gender'])

print(f"✓ Temporal features created")
print(f"  Date range: {df['Year'].min()} - {df['Year'].max()}")
print(f"  Features: Month, Quarter, DayOfWeek, Cyclical encodings")

# ==============================================================================
# PART 2: TASK 1 - LOCATION PREDICTION (CLASSIFICATION)
# ==============================================================================
print("\n\n" + "=" * 80)
print("🎯 PART 2: LOCATION PREDICTION (Classification Task)")
print("=" * 80)
print("Goal: Predict which barangay a missing person case will occur in")
print("-" * 80)

# Prepare features for classification
location_features = [
    'Latitude', 'Longitude', 'Month', 'Quarter', 'DayOfWeek',
    'Month_Sin', 'Month_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos',
    'Age', 'Gender_Encoded'
]

X_loc = df[location_features].dropna()
y_loc = df.loc[X_loc.index, 'Barangay District']

# Filter out rare locations (< 5 occurrences)
location_counts = y_loc.value_counts()
valid_locations = location_counts[location_counts >= 5].index
mask = y_loc.isin(valid_locations)
X_loc = X_loc[mask]
y_loc = y_loc[mask]

print(f"✓ Dataset: {len(X_loc)} samples, {len(valid_locations)} locations")
print(f"  Features: {len(location_features)}")

# Train-test split
X_train_loc, X_test_loc, y_train_loc, y_test_loc = train_test_split(
    X_loc, y_loc, test_size=0.2, random_state=42, stratify=y_loc
)

# Scale features
scaler_loc = StandardScaler()
X_train_loc_scaled = scaler_loc.fit_transform(X_train_loc)
X_test_loc_scaled = scaler_loc.transform(X_test_loc)

# --- Random Forest Classifier with Hyperparameter Tuning ---
print("\n🌲 Random Forest Classifier - Hyperparameter Tuning...")
print("-" * 80)

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)

rf_grid_search = GridSearchCV(
    rf_classifier,
    rf_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print(f"Testing {len(rf_param_grid['n_estimators']) * len(rf_param_grid['max_depth']) * len(rf_param_grid['min_samples_split']) * len(rf_param_grid['min_samples_leaf']) * len(rf_param_grid['max_features'])} combinations...")

rf_grid_search.fit(X_train_loc_scaled, y_train_loc)

print(f"\n✓ Best parameters:")
for param, value in rf_grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Evaluate best model
best_rf_classifier = rf_grid_search.best_estimator_
rf_train_acc = best_rf_classifier.score(X_train_loc_scaled, y_train_loc)
rf_test_acc = best_rf_classifier.score(X_test_loc_scaled, y_test_loc)
rf_cv_scores = cross_val_score(best_rf_classifier, X_train_loc_scaled, y_train_loc, cv=5)

y_pred_rf = best_rf_classifier.predict(X_test_loc_scaled)
rf_f1 = f1_score(y_test_loc, y_pred_rf, average='weighted')

print(f"\n📊 Random Forest Classifier Results:")
print(f"  Train Accuracy: {rf_train_acc:.4f}")
print(f"  Test Accuracy:  {rf_test_acc:.4f}")
print(f"  CV Accuracy:    {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
print(f"  F1-Score:       {rf_f1:.4f}")
print(f"  Best CV Score:  {rf_grid_search.best_score_:.4f}")

# Feature importance
rf_feature_importance = pd.DataFrame({
    'Feature': location_features,
    'Importance': best_rf_classifier.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n🔝 Top 5 Features:")
for idx, row in rf_feature_importance.head().iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# Save RF classification results
rf_classification_results = {
    'model': 'Random Forest Classifier',
    'task': 'Location Prediction',
    'best_params': rf_grid_search.best_params_,
    'train_accuracy': float(rf_train_acc),
    'test_accuracy': float(rf_test_acc),
    'cv_accuracy': float(rf_cv_scores.mean()),
    'cv_std': float(rf_cv_scores.std()),
    'f1_score': float(rf_f1),
    'n_locations': len(valid_locations),
    'n_samples': len(X_loc),
    'feature_importance': rf_feature_importance.to_dict('records')
}

# ==============================================================================
# PART 3: TASK 2 - HOTSPOT INTENSITY PREDICTION (REGRESSION)
# ==============================================================================
print("\n\n" + "=" * 80)
print("📈 PART 3: HOTSPOT INTENSITY PREDICTION (Regression Task)")
print("=" * 80)
print("Goal: Predict the number of cases per location")
print("-" * 80)

# Aggregate by location and year
agg_df = df.groupby(['Barangay District', 'Year']).agg({
    'Person ID': 'count',
    'Latitude': 'mean',
    'Longitude': 'mean',
    'Age': 'mean'
}).reset_index()
agg_df.rename(columns={'Person ID': 'Case_Count'}, inplace=True)

# Create lag features (previous year's count)
agg_df = agg_df.sort_values(['Barangay District', 'Year'])
agg_df['Prev_Year_Count'] = agg_df.groupby('Barangay District')['Case_Count'].shift(1)
agg_df = agg_df.dropna(subset=['Prev_Year_Count'])

# Prepare features for regression
regression_features = ['Latitude', 'Longitude', 'Year', 'Prev_Year_Count', 'Age']
X_reg = agg_df[regression_features]
y_reg = agg_df['Case_Count']

print(f"✓ Dataset: {len(X_reg)} samples")
print(f"  Features: {regression_features}")
print(f"  Target range: {y_reg.min():.0f} - {y_reg.max():.0f} cases")

# Train-test split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale features
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# --- Model 1: Random Forest Regressor ---
print("\n🌲 Random Forest Regressor - Hyperparameter Tuning...")
print("-" * 80)

rfr_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rfr = RandomForestRegressor(random_state=42, n_jobs=-1)

rfr_grid_search = GridSearchCV(
    rfr,
    rfr_param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

print(f"Testing {len(rfr_param_grid['n_estimators']) * len(rfr_param_grid['max_depth']) * len(rfr_param_grid['min_samples_split']) * len(rfr_param_grid['min_samples_leaf']) * len(rfr_param_grid['max_features'])} combinations...")

rfr_grid_search.fit(X_train_reg_scaled, y_train_reg)

print(f"\n✓ Best parameters:")
for param, value in rfr_grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Evaluate
best_rfr = rfr_grid_search.best_estimator_
y_pred_rfr_train = best_rfr.predict(X_train_reg_scaled)
y_pred_rfr_test = best_rfr.predict(X_test_reg_scaled)

rfr_train_r2 = r2_score(y_train_reg, y_pred_rfr_train)
rfr_test_r2 = r2_score(y_test_reg, y_pred_rfr_test)
rfr_train_rmse = np.sqrt(mean_squared_error(y_train_reg, y_pred_rfr_train))
rfr_test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_rfr_test))
rfr_test_mae = mean_absolute_error(y_test_reg, y_pred_rfr_test)

print(f"\n📊 Random Forest Regressor Results:")
print(f"  Train R²:   {rfr_train_r2:.4f}")
print(f"  Test R²:    {rfr_test_r2:.4f}")
print(f"  Train RMSE: {rfr_train_rmse:.4f}")
print(f"  Test RMSE:  {rfr_test_rmse:.4f}")
print(f"  Test MAE:   {rfr_test_mae:.4f}")

# --- Model 2: Gradient Boosting Regressor ---
print("\n\n🚀 Gradient Boosting Regressor - Hyperparameter Tuning...")
print("-" * 80)

gbr_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

gbr = GradientBoostingRegressor(random_state=42)

gbr_grid_search = GridSearchCV(
    gbr,
    gbr_param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

print(f"Testing {len(gbr_param_grid['n_estimators']) * len(gbr_param_grid['learning_rate']) * len(gbr_param_grid['max_depth']) * len(gbr_param_grid['min_samples_split']) * len(gbr_param_grid['min_samples_leaf']) * len(gbr_param_grid['subsample'])} combinations...")

gbr_grid_search.fit(X_train_reg_scaled, y_train_reg)

print(f"\n✓ Best parameters:")
for param, value in gbr_grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Evaluate
best_gbr = gbr_grid_search.best_estimator_
y_pred_gbr_train = best_gbr.predict(X_train_reg_scaled)
y_pred_gbr_test = best_gbr.predict(X_test_reg_scaled)

gbr_train_r2 = r2_score(y_train_reg, y_pred_gbr_train)
gbr_test_r2 = r2_score(y_test_reg, y_pred_gbr_test)
gbr_train_rmse = np.sqrt(mean_squared_error(y_train_reg, y_pred_gbr_train))
gbr_test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_gbr_test))
gbr_test_mae = mean_absolute_error(y_test_reg, y_pred_gbr_test)

print(f"\n📊 Gradient Boosting Regressor Results:")
print(f"  Train R²:   {gbr_train_r2:.4f}")
print(f"  Test R²:    {gbr_test_r2:.4f}")
print(f"  Train RMSE: {gbr_train_rmse:.4f}")
print(f"  Test RMSE:  {gbr_test_rmse:.4f}")
print(f"  Test MAE:   {gbr_test_mae:.4f}")

# --- Model 3: Poisson Regressor ---
print("\n\n📊 Poisson Regressor - Hyperparameter Tuning...")
print("-" * 80)

poisson_param_grid = {
    'alpha': [0.0, 0.1, 0.5, 1.0, 2.0],
    'max_iter': [100, 200, 500]
}

poisson = PoissonRegressor()

poisson_grid_search = GridSearchCV(
    poisson,
    poisson_param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

print(f"Testing {len(poisson_param_grid['alpha']) * len(poisson_param_grid['max_iter'])} combinations...")

poisson_grid_search.fit(X_train_reg_scaled, y_train_reg)

print(f"\n✓ Best parameters:")
for param, value in poisson_grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Evaluate
best_poisson = poisson_grid_search.best_estimator_
y_pred_poisson_train = best_poisson.predict(X_train_reg_scaled)
y_pred_poisson_test = best_poisson.predict(X_test_reg_scaled)

poisson_train_r2 = r2_score(y_train_reg, y_pred_poisson_train)
poisson_test_r2 = r2_score(y_test_reg, y_pred_poisson_test)
poisson_train_rmse = np.sqrt(mean_squared_error(y_train_reg, y_pred_poisson_train))
poisson_test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_poisson_test))
poisson_test_mae = mean_absolute_error(y_test_reg, y_pred_poisson_test)

print(f"\n📊 Poisson Regressor Results:")
print(f"  Train R²:   {poisson_train_r2:.4f}")
print(f"  Test R²:    {poisson_test_r2:.4f}")
print(f"  Train RMSE: {poisson_train_rmse:.4f}")
print(f"  Test RMSE:  {poisson_test_rmse:.4f}")
print(f"  Test MAE:   {poisson_test_mae:.4f}")

# ==============================================================================
# PART 4: MODEL COMPARISON & VISUALIZATION
# ==============================================================================
print("\n\n" + "=" * 80)
print("📊 PART 4: MODEL COMPARISON")
print("=" * 80)

# Regression models comparison
regression_comparison = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting', 'Poisson'],
    'Test_R2': [rfr_test_r2, gbr_test_r2, poisson_test_r2],
    'Test_RMSE': [rfr_test_rmse, gbr_test_rmse, poisson_test_rmse],
    'Test_MAE': [rfr_test_mae, gbr_test_mae, poisson_test_mae],
    'Train_R2': [rfr_train_r2, gbr_train_r2, poisson_train_r2],
    'Overfit_Gap': [
        rfr_train_r2 - rfr_test_r2,
        gbr_train_r2 - gbr_test_r2,
        poisson_train_r2 - poisson_test_r2
    ]
})

print("\n🏆 REGRESSION MODEL COMPARISON:")
print(regression_comparison.to_string(index=False))

# Determine best model
best_model_idx = regression_comparison['Test_R2'].idxmax()
best_model = regression_comparison.loc[best_model_idx, 'Model']
print(f"\n✅ BEST MODEL: {best_model} (Test R² = {regression_comparison.loc[best_model_idx, 'Test_R2']:.4f})")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: R² Comparison
ax1 = axes[0, 0]
models = regression_comparison['Model']
x_pos = np.arange(len(models))
ax1.bar(x_pos - 0.2, regression_comparison['Train_R2'], 0.4, label='Train R²', alpha=0.8)
ax1.bar(x_pos + 0.2, regression_comparison['Test_R2'], 0.4, label='Test R²', alpha=0.8)
ax1.set_xlabel('Model')
ax1.set_ylabel('R² Score')
ax1.set_title('R² Score Comparison')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: RMSE Comparison
ax2 = axes[0, 1]
ax2.bar(models, regression_comparison['Test_RMSE'], color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
ax2.set_xlabel('Model')
ax2.set_ylabel('RMSE')
ax2.set_title('Test RMSE Comparison (Lower is Better)')
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Overfit Analysis
ax3 = axes[1, 0]
ax3.bar(models, regression_comparison['Overfit_Gap'], color=['#d62728', '#9467bd', '#8c564b'], alpha=0.8)
ax3.set_xlabel('Model')
ax3.set_ylabel('Overfit Gap (Train R² - Test R²)')
ax3.set_title('Overfitting Analysis (Lower is Better)')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Actual vs Predicted (Best Model)
ax4 = axes[1, 1]
if best_model == 'Random Forest':
    y_pred_best = y_pred_rfr_test
elif best_model == 'Gradient Boosting':
    y_pred_best = y_pred_gbr_test
else:
    y_pred_best = y_pred_poisson_test

ax4.scatter(y_test_reg, y_pred_best, alpha=0.5)
ax4.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
ax4.set_xlabel('Actual Cases')
ax4.set_ylabel('Predicted Cases')
ax4.set_title(f'Actual vs Predicted ({best_model})')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'predictor_evaluation.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization: {OUTPUT_DIR / 'predictor_evaluation.png'}")

# ==============================================================================
# PART 5: SAVE COMPREHENSIVE RESULTS
# ==============================================================================
print("\n\n" + "=" * 80)
print("💾 PART 5: SAVING RESULTS")
print("=" * 80)

# Save regression comparison
regression_comparison.to_csv(OUTPUT_DIR / 'predictor_comparison.csv', index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'predictor_comparison.csv'}")

# Save comprehensive evaluation summary
evaluation_summary = {
    'evaluation_date': pd.Timestamp.now().isoformat(),
    'dataset': {
        'path': DATA_PATH,
        'total_records': len(df),
        'date_range': f"{df['Year'].min()} - {df['Year'].max()}"
    },
    'location_prediction': rf_classification_results,
    'intensity_prediction': {
        'random_forest': {
            'best_params': rfr_grid_search.best_params_,
            'test_r2': float(rfr_test_r2),
            'test_rmse': float(rfr_test_rmse),
            'test_mae': float(rfr_test_mae),
            'overfit_gap': float(rfr_train_r2 - rfr_test_r2)
        },
        'gradient_boosting': {
            'best_params': gbr_grid_search.best_params_,
            'test_r2': float(gbr_test_r2),
            'test_rmse': float(gbr_test_rmse),
            'test_mae': float(gbr_test_mae),
            'overfit_gap': float(gbr_train_r2 - gbr_test_r2)
        },
        'poisson': {
            'best_params': poisson_grid_search.best_params_,
            'test_r2': float(poisson_test_r2),
            'test_rmse': float(poisson_test_rmse),
            'test_mae': float(poisson_test_mae),
            'overfit_gap': float(poisson_train_r2 - poisson_test_r2)
        },
        'best_model': best_model,
        'best_test_r2': float(regression_comparison.loc[best_model_idx, 'Test_R2'])
    }
}

with open(OUTPUT_DIR / 'predictor_evaluation_summary.json', 'w') as f:
    json.dump(evaluation_summary, f, indent=2)
print(f"✓ Saved: {OUTPUT_DIR / 'predictor_evaluation_summary.json'}")

# Save feature importance
rf_feature_importance.to_csv(OUTPUT_DIR / 'rf_feature_importance.csv', index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'rf_feature_importance.csv'}")

# ==============================================================================
# PART 6: FINAL RECOMMENDATIONS
# ==============================================================================
print("\n\n" + "=" * 80)
print("🎯 PART 6: RECOMMENDATIONS")
print("=" * 80)

print(f"\n📍 LOCATION PREDICTION (Classification):")
print(f"  ✅ Use: Random Forest Classifier")
print(f"  ✅ Test Accuracy: {rf_test_acc:.4f}")
print(f"  ✅ Best for: Identifying which barangay will have cases")

print(f"\n📈 INTENSITY PREDICTION (Regression):")
print(f"  ✅ Use: {best_model}")
print(f"  ✅ Test R²: {regression_comparison.loc[best_model_idx, 'Test_R2']:.4f}")
print(f"  ✅ Best for: Forecasting number of cases per location")

print("\n💡 THESIS RECOMMENDATIONS:")
print("  1. Use Random Forest Classifier for location hotspot prediction")
print(f"  2. Use {best_model} for intensity forecasting")
print("  3. Feature importance: Latitude, Longitude are top predictors")
print("  4. Temporal patterns (Month, DayOfWeek) have moderate importance")

print("\n" + "=" * 80)
print("✅ EVALUATION COMPLETE!")
print("=" * 80)
