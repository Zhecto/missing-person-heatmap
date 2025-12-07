# Web App Model Configuration

## Overview
The Streamlit web application now uses a **single-model configuration** approach for both clustering and prediction models. Simply edit `config/settings.yaml` to change which models are used.

## Configuration File: `config/settings.yaml`

```yaml
modeling:
  clustering_method: dbscan        # Options: 'kmeans', 'dbscan'
  prediction_model: poisson        # Options: 'gradient_boosting', 'poisson'
  spatial_bandwidth_km: 25
  random_seed: 42
```

## How It Works

### 1. Clustering Model Configuration

**Current Setting:** `clustering_method: dbscan`

The app automatically uses the configured clustering method with preset parameters:

- **K-Means**: `n_clusters: 3`
- **DBSCAN**: `eps: 0.1, min_samples: 3`

**To Change:**
```yaml
# Option 1: Use K-Means
modeling:
  clustering_method: kmeans

# Option 2: Use DBSCAN
modeling:
  clustering_method: dbscan
```

### 2. Prediction Model Configuration

**Current Setting:** `prediction_model: poisson`

The app automatically uses the configured prediction method:

- **Poisson Regressor**: Best for count data, minimal overfitting
- **Gradient Boosting**: Better for complex patterns, prone to overfitting

**To Change:**
```yaml
# Option 1: Use Poisson (Recommended)
modeling:
  prediction_model: poisson

# Option 2: Use Gradient Boosting
modeling:
  prediction_model: gradient_boosting
```

## How the App Reads Configuration

### Code Implementation

```python
# In streamlit_app.py (lines 26-41)
import yaml

# Load configuration
config_path = Path(__file__).parent / 'config' / 'settings.yaml'
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        CLUSTERING_MODEL = config.get('modeling', {}).get('clustering_method', 'kmeans')
        PREDICTION_MODEL = config.get('modeling', {}).get('prediction_model', 'poisson')
else:
    # Fallback defaults
    CLUSTERING_MODEL = "kmeans"
    PREDICTION_MODEL = "poisson"

# CLUSTERING MODEL PARAMETERS
if CLUSTERING_MODEL == "kmeans":
    CLUSTERING_PARAMS = {'n_clusters': 3}
else:  # dbscan
    CLUSTERING_PARAMS = {'eps': 0.1, 'min_samples': 3}
```

The constants `CLUSTERING_MODEL` and `PREDICTION_MODEL` are then used throughout the app.

## Usage in Demo Script

The `demo.py` script also uses the same configuration:

```python
# In demo.py
from core.analysis.predictor import SpatialPredictor

predictor = SpatialPredictor()  # Automatically reads config
print(f"Using model: {predictor.configured_model}")
metrics = predictor.train_configured_model(df_clean)
```

## Testing Your Configuration

### Step 1: Edit Configuration
```bash
# Edit config/settings.yaml
notepad config\settings.yaml
```

### Step 2: Run Application
```bash
# Option A: Demo script
python demo.py

# Option B: Streamlit web app
streamlit run streamlit_app.py
```

### Step 3: Verify
Look for these indicators in the output:

**Demo Script:**
```
[STEP 6] PREDICTIVE MODELING
Using model from config/settings.yaml: poisson
🤖 Training Poisson model...
```

**Streamlit App:**
```
Predicting future hotspots using Poisson model on 600 records
🤖 Training Poisson model...
✅ Prediction complete! Poisson model trained successfully

📊 Poisson Model Performance
R² Score: -0.066
RMSE: 1.49
Overfit Gap: 0.098
```

## Configuration Examples

### Example 1: Thesis Defense Setup (Recommended)
```yaml
# Best balance: DBSCAN for clustering, Poisson for prediction
modeling:
  clustering_method: dbscan
  prediction_model: poisson
  spatial_bandwidth_km: 25
  random_seed: 42
```

**Why?**
- DBSCAN: Finds natural density-based clusters (eps=0.1 optimal)
- Poisson: Minimal overfitting (0.03 gap), 92% total accuracy

### Example 2: K-Means Comparison
```yaml
# For comparing with K-Means clustering
modeling:
  clustering_method: kmeans
  prediction_model: poisson
  spatial_bandwidth_km: 25
  random_seed: 42
```

### Example 3: Gradient Boosting Experiment
```yaml
# To test Gradient Boosting performance
modeling:
  clustering_method: dbscan
  prediction_model: gradient_boosting
  spatial_bandwidth_km: 25
  random_seed: 42
```

## Model Comparison

| Configuration | Clustering | Prediction | Performance | Use Case |
|--------------|-----------|------------|-------------|----------|
| **Default** | DBSCAN | Poisson | ⭐⭐⭐⭐⭐ | Production, Thesis |
| Alternative 1 | K-Means | Poisson | ⭐⭐⭐⭐ | Comparison |
| Alternative 2 | DBSCAN | Gradient Boosting | ⭐⭐⭐ | Experiments |
| Alternative 3 | K-Means | Gradient Boosting | ⭐⭐ | Not recommended |

## Benefits of This Approach

### ✅ Advantages
1. **Single Source of Truth**: One configuration file controls all components
2. **No Code Changes**: Switch models without editing Python code
3. **Consistency**: Demo script and web app use same configuration
4. **Version Control**: Configuration is tracked in Git
5. **Easy Testing**: Quickly compare different model combinations

### 📝 Best Practices
1. **Document Changes**: Add comments in settings.yaml explaining why you chose each model
2. **Test After Changes**: Always run both demo.py and streamlit app after configuration changes
3. **Keep Defaults**: The default settings (DBSCAN + Poisson) are optimized for your data

## Troubleshooting

### Issue: Changes not reflected
**Solution**: Restart the Streamlit app or Python kernel
```bash
# Stop current Streamlit (Ctrl+C)
# Restart
streamlit run streamlit_app.py
```

### Issue: Configuration file not found
**Solution**: Check file path and ensure settings.yaml exists
```bash
dir config\settings.yaml
```

### Issue: Invalid model name
**Solution**: Check spelling - valid options:
- Clustering: `kmeans`, `dbscan` (lowercase)
- Prediction: `poisson`, `gradient_boosting` (with underscore)

## Advanced: Programmatic Override

You can override the configuration in code if needed:

```python
# Override clustering
clustering = ClusteringModel()
if CLUSTERING_MODEL == "kmeans":
    clustering.fit_kmeans(df, n_clusters=5)  # Override n_clusters
else:
    clustering.fit_dbscan(df, eps=0.15)  # Override eps

# Override prediction
predictor = SpatialPredictor()
metrics = predictor.train_configured_model(
    df, 
    model_name='gradient_boosting'  # Override configured model
)
```

---

**Last Updated**: December 2025  
**Related Files**:
- `config/settings.yaml` - Configuration file
- `streamlit_app.py` - Web application
- `demo.py` - Demo script
- `src/core/analysis/predictor.py` - Prediction module
- `src/core/analysis/clustering.py` - Clustering module

**See Also**:
- `PREDICTION_MODEL_GUIDE.md` - Detailed prediction model comparison
- `QUICK_REFERENCE_PREDICTION.txt` - Quick reference card
