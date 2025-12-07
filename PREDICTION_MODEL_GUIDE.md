# Prediction Model Configuration Guide

This guide explains how to manually configure and switch between prediction models in the web application.

## Available Prediction Models

The system supports two prediction models for hotspot intensity forecasting:

### 1. **Poisson Regressor** (Recommended)
- **Best for**: Count data prediction (number of cases per location)
- **Advantages**: 
  - Minimal overfitting (gap ~0.03-0.10)
  - Designed specifically for count data
  - Better for total case predictions (92% accuracy)
  - More interpretable coefficients
- **Disadvantages**: 
  - Lower spatial R² scores
  - Simpler model with fewer hyperparameters

### 2. **Gradient Boosting Regressor**
- **Best for**: Complex pattern detection
- **Advantages**:
  - Higher training R² scores (0.86+)
  - Better feature importance visualization
  - Can capture non-linear patterns
- **Disadvantages**:
  - Severe overfitting (gap ~1.4)
  - Slower training time
  - Less accurate total predictions

## How to Change Prediction Models

### Method 1: Edit Configuration File (Recommended)

1. Open `config/settings.yaml`
2. Locate the `modeling` section:
   ```yaml
   modeling:
     clustering_method: dbscan
     prediction_model: poisson  # Change this line
     spatial_bandwidth_km: 25
     random_seed: 42
   ```
3. Change `prediction_model` to either:
   - `poisson` (for Poisson Regressor)
   - `gradient_boosting` (for Gradient Boosting)
4. Save the file
5. Run your application:
   ```bash
   python demo.py
   ```

### Method 2: Programmatic Override

In your Python code, you can override the configured model:

```python
from core.analysis.predictor import SpatialPredictor

# Initialize predictor (reads from settings.yaml)
predictor = SpatialPredictor()

# Override with specific model
metrics = predictor.train_configured_model(
    df_clean, 
    model_name='gradient_boosting'  # or 'poisson'
)
```

## Model Comparison Results

Based on evaluation with 2025 test data:

| Metric | Gradient Boosting | Poisson |
|--------|------------------|---------|
| **Test R²** | -0.57 | **-0.01** ✓ |
| **Test RMSE** | 1.91 | **1.53** ✓ |
| **Test MAE** | 1.52 | **1.26** ✓ |
| **Overfit Gap** | 1.43 | **0.03** ✓ |
| **Total Accuracy** | 85% | **92%** ✓ |
| **Training Time** | ~20s | ~5s ✓ |

✓ = Better performance

## Recommendation for Thesis

**Use Poisson Regressor** because:
1. ✅ Minimal overfitting (more reliable for unseen data)
2. ✅ 92% accuracy on total 2025 case predictions
3. ✅ Designed for count data (matches the problem domain)
4. ✅ Faster training and prediction
5. ✅ More interpretable for stakeholders

**Note**: Negative R² values are expected when predicting rare spatial events. The model performs well on total volume prediction while spatial distribution remains challenging.

## Configuration Examples

### Example 1: Production Setup (Poisson)
```yaml
modeling:
  clustering_method: dbscan
  prediction_model: poisson
  spatial_bandwidth_km: 25
  random_seed: 42
```

### Example 2: Thesis Comparison (Gradient Boosting)
```yaml
modeling:
  clustering_method: kmeans
  prediction_model: gradient_boosting
  spatial_bandwidth_km: 25
  random_seed: 42
```

## Testing Your Configuration

After changing the model in `settings.yaml`:

```bash
# Full demo with all steps
python demo.py

# Quick evaluation only
python evaluate_predictors.py

# Web application
streamlit run streamlit_app.py
```

You should see output like:
```
[STEP 6] PREDICTIVE MODELING
Using model from config/settings.yaml: poisson
🤖 Training Poisson model...
  Best parameters: {'alpha': 2.0, 'max_iter': 100}
✓ Poisson trained successfully!
  Test R²: -0.0089
  Test RMSE: 1.53
```

## Troubleshooting

### Issue: Model not changing
**Solution**: Restart Python kernel or terminal session to reload the configuration.

### Issue: Import errors
**Solution**: Ensure PoissonRegressor is imported correctly:
```python
from sklearn.linear_model import PoissonRegressor
```

### Issue: ValueError about unknown model
**Solution**: Check spelling in `settings.yaml`. Valid options:
- `gradient_boosting` (with underscore)
- `poisson` (lowercase)

## Future Enhancements

Consider adding these models in future versions:
- **XGBoost Regressor**: Advanced boosting with better regularization
- **LightGBM**: Faster training for large datasets
- **Neural Networks**: LSTM for temporal patterns
- **Spatial Models**: Kriging, Spatial Lag Regression

---

**Last Updated**: December 2025  
**Project**: Missing Person Heatmap Analysis  
**Version**: 1.0
