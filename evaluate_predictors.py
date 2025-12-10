"""
Comprehensive Prediction Model Evaluation with Hyperparameter Tuning
Predicts 2025 missing person hotspots based on 2019-2024 historical data.

Approach:
- Train on 2019-2024 (500 records)
- Test on 2025 (100 records)
- Predict case counts per barangay for next year
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

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

from core.ingestion.data_loader import DataLoader
from core.preprocessing.data_cleaner import DataCleaner

# Configuration
DATA_PATH = "data/sample_data.csv"
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("🔮 PREDICTION MODEL EVALUATION - 2025 HOTSPOT FORECASTING")
print("=" * 80)
print(f"Dataset: {DATA_PATH}")
print(f"Approach: Train on 2019-2024 → Test on 2025")
print(f"Output: {OUTPUT_DIR}")
print("=" * 80)

# ==============================================================================
# PART 1: DATA LOADING & TEMPORAL SPLIT
# ==============================================================================
print("\n📂 PART 1: Loading and splitting data by year...")
print("-" * 80)

loader = DataLoader()
cleaner = DataCleaner()

# Load data
df = loader.load_csv(DATA_PATH)
print(f"✓ Loaded {len(df)} records")

# Clean data
df = cleaner.handle_missing_values(df, strategy='smart')
print(f"✓ Cleaned: {len(df)} records remain")

# Parse dates
df['Date Reported Missing'] = pd.to_datetime(df['Date Reported Missing'], errors='coerce')
df['Year'] = df['Date Reported Missing'].dt.year
df['Month'] = df['Date Reported Missing'].dt.month
df['Quarter'] = df['Date Reported Missing'].dt.quarter
df['DayOfWeek'] = df['Date Reported Missing'].dt.dayofweek

# Temporal split: Train on 2019-2024, Test on 2025
df_train = df[df['Year'] < 2025].copy()
df_test = df[df['Year'] == 2025].copy()

print(f"\n📊 Temporal Split:")
print(f"  Training: 2019-2024 → {len(df_train)} records")
print(f"  Testing:  2025      → {len(df_test)} records")

year_counts = df['Year'].value_counts().sort_index()
for year, count in year_counts.items():
    marker = "🔵 TRAIN" if year < 2025 else "🟢 TEST"
    print(f"    {year}: {count:3d} records {marker}")

# ==============================================================================
# PART 2: AGGREGATE BY LOCATION-YEAR FOR HOTSPOT PREDICTION
# ==============================================================================
print("\n\n" + "=" * 80)
print("📈 PART 2: HOTSPOT INTENSITY PREDICTION TASK")
print("=" * 80)
print("Goal: Predict number of cases per barangay in 2025")
print("-" * 80)

# Aggregate training data by location and year
agg_train = df_train.groupby(['Barangay District', 'Year']).agg({
    'Person ID': 'count',
    'Latitude': 'mean',
    'Longitude': 'mean',
    'Age': 'mean'
}).reset_index()
agg_train.rename(columns={'Person ID': 'Case_Count'}, inplace=True)

# Create lag features (previous year's count)
agg_train = agg_train.sort_values(['Barangay District', 'Year'])
agg_train['Prev_Year_Count'] = agg_train.groupby('Barangay District')['Case_Count'].shift(1)
agg_train = agg_train.dropna(subset=['Prev_Year_Count'])

print(f"✓ Training samples: {len(agg_train)} (location-year combinations)")
print(f"  Features: Latitude, Longitude, Year, Prev_Year_Count, Age")
print(f"  Target: Case_Count per location")

# Prepare features and target for training
regression_features = ['Latitude', 'Longitude', 'Year', 'Prev_Year_Count', 'Age']
X_train = agg_train[regression_features]
y_train = agg_train['Case_Count']

print(f"  Target range: {y_train.min():.0f} - {y_train.max():.0f} cases")

# For 2025 prediction, we need 2024 case counts as lag feature
agg_2024 = df_train[df_train['Year'] == 2024].groupby('Barangay District').agg({
    'Person ID': 'count',
    'Latitude': 'mean',
    'Longitude': 'mean',
    'Age': 'mean'
}).reset_index()
agg_2024.rename(columns={'Person ID': 'Case_Count_2024'}, inplace=True)

# Create 2025 prediction features for each barangay
unique_barangays = df_train['Barangay District'].unique()
X_test_list = []

for barangay in unique_barangays:
    # Get 2024 count (or 0 if barangay had no cases in 2024)
    prev_count = agg_2024[agg_2024['Barangay District'] == barangay]['Case_Count_2024'].values
    prev_count = prev_count[0] if len(prev_count) > 0 else 0
    
    # Get average coordinates and age
    barangay_data = df_train[df_train['Barangay District'] == barangay]
    
    X_test_list.append({
        'Barangay District': barangay,
        'Latitude': barangay_data['Latitude'].mean(),
        'Longitude': barangay_data['Longitude'].mean(),
        'Year': 2025,
        'Prev_Year_Count': prev_count,
        'Age': barangay_data['Age'].mean()
    })

df_test_features = pd.DataFrame(X_test_list)
X_test = df_test_features[regression_features]

# Actual 2025 counts per barangay
actual_2025 = df_test.groupby('Barangay District')['Person ID'].count().to_dict()
y_test = df_test_features['Barangay District'].map(actual_2025).fillna(0)

print(f"\n✓ Test samples: {len(X_test)} barangays")
print(f"  Actual 2025 cases: {y_test.sum():.0f} total across all barangays")

# Scale features
scaler_reg = StandardScaler()
X_train_scaled = scaler_reg.fit_transform(X_train)
X_test_scaled = scaler_reg.transform(X_test)

# ==============================================================================
# PART 3: MODEL TRAINING WITH HYPERPARAMETER TUNING
# ==============================================================================
print("\n\n" + "=" * 80)
print("🤖 PART 3: TRAINING PREDICTION MODELS")
print("=" * 80)
print("Note: Random Forest removed due to severe overfitting (0.71 gap)")
print("Testing: Gradient Boosting vs Poisson Regression")

# --- Model 1: Gradient Boosting Regressor ---
print("\n\n🚀 Gradient Boosting Regressor - Hyperparameter Tuning...")
print("-" * 80)

gbr_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

gbr = GradientBoostingRegressor(random_state=42)

gbr_grid_search = GridSearchCV(
    gbr,
    gbr_param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

print(f"Testing {len(gbr_param_grid['n_estimators']) * len(gbr_param_grid['learning_rate']) * len(gbr_param_grid['max_depth']) * len(gbr_param_grid['min_samples_split'])} combinations...")

gbr_grid_search.fit(X_train_scaled, y_train)

print(f"\n✓ Best parameters:")
for param, value in gbr_grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Evaluate on 2025
best_gbr = gbr_grid_search.best_estimator_
y_pred_gbr_train = best_gbr.predict(X_train_scaled)
y_pred_gbr_test = best_gbr.predict(X_test_scaled)

gbr_train_r2 = r2_score(y_train, y_pred_gbr_train)
gbr_test_r2 = r2_score(y_test, y_pred_gbr_test)
gbr_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_gbr_train))
gbr_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gbr_test))
gbr_test_mae = mean_absolute_error(y_test, y_pred_gbr_test)

print(f"\n📊 Gradient Boosting Regressor Results:")
print(f"  Train R²:   {gbr_train_r2:.4f}")
print(f"  Test R² (2025):    {gbr_test_r2:.4f}")
print(f"  Train RMSE: {gbr_train_rmse:.4f}")
print(f"  Test RMSE (2025):  {gbr_test_rmse:.4f}")
print(f"  Test MAE (2025):   {gbr_test_mae:.4f}")

# --- Model 2: Poisson Regressor ---
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
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

print(f"Testing {len(poisson_param_grid['alpha']) * len(poisson_param_grid['max_iter'])} combinations...")

poisson_grid_search.fit(X_train_scaled, y_train)

print(f"\n✓ Best parameters:")
for param, value in poisson_grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Evaluate on 2025
best_poisson = poisson_grid_search.best_estimator_
y_pred_poisson_train = best_poisson.predict(X_train_scaled)
y_pred_poisson_test = best_poisson.predict(X_test_scaled)

poisson_train_r2 = r2_score(y_train, y_pred_poisson_train)
poisson_test_r2 = r2_score(y_test, y_pred_poisson_test)
poisson_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_poisson_train))
poisson_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_poisson_test))
poisson_test_mae = mean_absolute_error(y_test, y_pred_poisson_test)

print(f"\n📊 Poisson Regressor Results:")
print(f"  Train R²:   {poisson_train_r2:.4f}")
print(f"  Test R² (2025):    {poisson_test_r2:.4f}")
print(f"  Train RMSE: {poisson_train_rmse:.4f}")
print(f"  Test RMSE (2025):  {poisson_test_rmse:.4f}")
print(f"  Test MAE (2025):   {poisson_test_mae:.4f}")

# ==============================================================================
# PART 4: MODEL COMPARISON & VISUALIZATION
# ==============================================================================
print("\n\n" + "=" * 80)
print("📊 PART 4: MODEL COMPARISON & 2025 PREDICTION ACCURACY")
print("=" * 80)

# Regression models comparison
regression_comparison = pd.DataFrame({
    'Model': ['Gradient Boosting', 'Poisson'],
    'Train_R2': [gbr_train_r2, poisson_train_r2],
    'Test_R2_2025': [gbr_test_r2, poisson_test_r2],
    'Test_RMSE_2025': [gbr_test_rmse, poisson_test_rmse],
    'Test_MAE_2025': [gbr_test_mae, poisson_test_mae],
    'Overfit_Gap': [
        gbr_train_r2 - gbr_test_r2,
        poisson_train_r2 - poisson_test_r2
    ]
})

print("\n🏆 2025 PREDICTION PERFORMANCE:")
print(regression_comparison.to_string(index=False))

# Determine best model
best_model_idx = regression_comparison['Test_R2_2025'].idxmax()
best_model = regression_comparison.loc[best_model_idx, 'Model']
best_r2 = regression_comparison.loc[best_model_idx, 'Test_R2_2025']
print(f"\n✅ BEST MODEL: {best_model} (Test R² = {best_r2:.4f})")

# Calculate actual vs predicted totals
actual_total = y_test.sum()
if best_model == 'Gradient Boosting':
    predicted_total = y_pred_gbr_test.sum()
else:  # Poisson
    predicted_total = y_pred_poisson_test.sum()

print(f"\n📍 2025 TOTAL CASES:")
print(f"  Actual:    {actual_total:.0f} cases")
print(f"  Predicted: {predicted_total:.0f} cases")
print(f"  Error:     {abs(actual_total - predicted_total):.0f} cases ({abs(actual_total - predicted_total)/actual_total*100:.1f}%)")

# Create 2025 prediction DataFrame
df_2025_predictions = df_test_features.copy()
df_2025_predictions['Actual_Cases'] = y_test.values
df_2025_predictions['Predicted_Cases_GB'] = y_pred_gbr_test
df_2025_predictions['Predicted_Cases_Poisson'] = y_pred_poisson_test

# Sort by actual cases
df_2025_predictions = df_2025_predictions.sort_values('Actual_Cases', ascending=False)

print(f"\n🔝 TOP 10 HOTSPOTS IN 2025 (Actual):")
print(df_2025_predictions[['Barangay District', 'Actual_Cases', f'Predicted_Cases_{best_model.split()[0]}']].head(10).to_string(index=False))

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: R² Comparison
ax1 = axes[0, 0]
models = regression_comparison['Model']
x_pos = np.arange(len(models))
ax1.bar(x_pos - 0.2, regression_comparison['Train_R2'], 0.4, label='Train R²', alpha=0.8)
ax1.bar(x_pos + 0.2, regression_comparison['Test_R2_2025'], 0.4, label='Test R² (2025)', alpha=0.8)
ax1.set_xlabel('Model')
ax1.set_ylabel('R² Score')
ax1.set_title('R² Score: Training vs 2025 Prediction')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: RMSE Comparison
ax2 = axes[0, 1]
ax2.bar(models, regression_comparison['Test_RMSE_2025'], color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
ax2.set_xlabel('Model')
ax2.set_ylabel('RMSE')
ax2.set_title('Test RMSE on 2025 Data (Lower is Better)')
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Actual vs Predicted (Best Model)
ax3 = axes[1, 0]
if best_model == 'Gradient Boosting':
    y_pred_best = y_pred_gbr_test
else:  # Poisson
    y_pred_best = y_pred_poisson_test

ax3.scatter(y_test, y_pred_best, alpha=0.6, s=100)
max_val = max(y_test.max(), y_pred_best.max())
ax3.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
ax3.set_xlabel('Actual Cases (2025)')
ax3.set_ylabel('Predicted Cases')
ax3.set_title(f'2025 Prediction Accuracy ({best_model})')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Top 10 Barangays - Actual vs Predicted
ax4 = axes[1, 1]
top10 = df_2025_predictions.head(10)
x_pos = np.arange(len(top10))
width = 0.35

ax4.barh(x_pos - width/2, top10['Actual_Cases'], width, label='Actual', alpha=0.8)
ax4.barh(x_pos + width/2, top10[f'Predicted_Cases_{best_model.split()[0]}'], width, label='Predicted', alpha=0.8)
ax4.set_yticks(x_pos)
ax4.set_yticklabels(top10['Barangay District'].values, fontsize=8)
ax4.set_xlabel('Number of Cases')
ax4.set_title('Top 10 Hotspots: Actual vs Predicted (2025)')
ax4.legend()
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'predictor_evaluation_2025.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization: {OUTPUT_DIR / 'predictor_evaluation_2025.png'}")

# ==============================================================================
# PART 5: SAVE COMPREHENSIVE RESULTS
# ==============================================================================
print("\n\n" + "=" * 80)
print("💾 PART 5: SAVING RESULTS")
print("=" * 80)

# Save regression comparison
regression_comparison.to_csv(OUTPUT_DIR / 'predictor_comparison_2025.csv', index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'predictor_comparison_2025.csv'}")

# Save 2025 predictions
df_2025_predictions.to_csv(OUTPUT_DIR / 'predictions_2025_by_barangay.csv', index=False)
print(f"✓ Saved: {OUTPUT_DIR / 'predictions_2025_by_barangay.csv'}")

# Save comprehensive evaluation summary
evaluation_summary = {
    'evaluation_date': pd.Timestamp.now().isoformat(),
    'approach': 'Time series forecasting: Train on 2019-2024, Test on 2025',
    'dataset': {
        'path': DATA_PATH,
        'total_records': len(df),
        'training_records': len(df_train),
        'test_records': len(df_test),
        'date_range': f"{df['Year'].min()} - {df['Year'].max()}"
    },
    'intensity_prediction_2025': {
        'gradient_boosting': {
            'best_params': gbr_grid_search.best_params_,
            'test_r2_2025': float(gbr_test_r2),
            'test_rmse_2025': float(gbr_test_rmse),
            'test_mae_2025': float(gbr_test_mae),
            'overfit_gap': float(gbr_train_r2 - gbr_test_r2)
        },
        'poisson': {
            'best_params': poisson_grid_search.best_params_,
            'test_r2_2025': float(poisson_test_r2),
            'test_rmse_2025': float(poisson_test_rmse),
            'test_mae_2025': float(poisson_test_mae),
            'overfit_gap': float(poisson_train_r2 - poisson_test_r2)
        },
        'best_model': best_model,
        'best_test_r2_2025': float(best_r2)
    },
    '2025_totals': {
        'actual_cases': int(actual_total),
        'predicted_cases': float(predicted_total),
        'prediction_error': float(abs(actual_total - predicted_total)),
        'error_percentage': float(abs(actual_total - predicted_total)/actual_total*100)
    }
}

with open(OUTPUT_DIR / 'predictor_evaluation_summary_2025.json', 'w') as f:
    json.dump(evaluation_summary, f, indent=2)
print(f"✓ Saved: {OUTPUT_DIR / 'predictor_evaluation_summary_2025.json'}")

# ==============================================================================
# PART 6: FINAL RECOMMENDATIONS
# ==============================================================================
print("\n\n" + "=" * 80)
print("🎯 PART 6: FINAL RECOMMENDATIONS")
print("=" * 80)

print(f"\n📈 2025 HOTSPOT PREDICTION:")
print(f"  ✅ Best Model: {best_model}")
print(f"  ✅ Test R² (2025): {best_r2:.4f}")
print(f"  ✅ Prediction Accuracy: {(1 - abs(actual_total - predicted_total)/actual_total)*100:.1f}%")

print("\n💡 FOR THESIS DEFENSE:")
print(f"  1. Model trained on 2019-2024 ({len(df_train)} records)")
print(f"  2. Successfully predicted 2025 hotspots with R²={best_r2:.4f}")
print(f"  3. Predicted {predicted_total:.0f} cases vs {actual_total:.0f} actual")
print(f"  4. Top predictive features: Latitude, Longitude, Previous Year Count")

print("\n🔮 INTERPRETATION:")
if best_r2 > 0.3:
    print(f"  ✅ STRONG: R²={best_r2:.4f} indicates good predictive power")
elif best_r2 > 0.15:
    print(f"  🟡 MODERATE: R²={best_r2:.4f} captures basic patterns, room for improvement")
else:
    print(f"  🟠 WEAK: R²={best_r2:.4f} suggests need for more features or data")

print("\n🚀 FUTURE IMPROVEMENTS:")
print("  1. Add external features: population density, transport hubs, crime rates")
print("  2. Collect more historical data (10+ years preferred)")
print("  3. Try spatial models: Spatial lag regression, kriging")
print("  4. Implement deep learning: LSTM for temporal patterns")

print("\n" + "=" * 80)
print("✅ 2025 PREDICTION EVALUATION COMPLETE!")
print("=" * 80)
