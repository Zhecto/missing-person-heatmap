# Quick Start Guide - Missing Person Heatmap Analysis

This guide helps you get started with the system before you have actual data.

## ✅ Prerequisites

- Python 3.10 or higher
- pip or Poetry package manager
- Text editor or IDE (VS Code recommended)

## 🚀 Installation

### Option 1: Using pip (Recommended for Quick Start)

```powershell
# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Poetry

```powershell
# Install Poetry (if not already installed)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

## 🎯 Running the Demo (No Dataset Required)

The demo script generates sample data and runs the complete analysis pipeline:

```powershell
python demo.py
```

### What the Demo Does:

1. ✅ Generates 500 sample missing person records
2. ✅ Validates and loads the data
3. ✅ Cleans and preprocesses
4. ✅ Runs K-means clustering (finds 5 clusters)
5. ✅ Analyzes trends (yearly, seasonal)
6. ✅ Trains prediction model
7. ✅ Generates visualizations (heatmap + charts)

### Demo Output:

All output files are saved to `data/outputs/`:
- `demo_heatmap.html` - Interactive heatmap
- `charts/` - Statistical charts (6 charts)
- `sample_data.csv` - Generated sample data

**Open the heatmap:**
```powershell
# Open in default browser
start data/outputs/demo_heatmap.html
```

## 🌐 Starting the API Server

The FastAPI backend provides REST endpoints for data processing:

```powershell
uvicorn src.backend.main:app --reload
```

### Access Points:

- **API Root**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### Try the API:

1. Open http://localhost:8000/docs in your browser
2. Click on any endpoint to expand it
3. Click "Try it out"
4. Fill in parameters and click "Execute"

## 📋 Using with Your Own Data

### 1. Prepare Your CSV

Your CSV must have these columns:
- Person ID
- Gender
- Age
- Date Reported Missing
- Time Reported Missing
- Location last seen
- Latitude
- Longitude
- Barangay District
- Post URL

### 2. Upload via API

```python
import requests

# Upload CSV
with open("your_data.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/data/upload",
        files={"file": f}
    )

print(response.json())

# Clean data
response = requests.post("http://localhost:8000/api/data/clean")
print(response.json())

# Run clustering
response = requests.post(
    "http://localhost:8000/api/analysis/clustering",
    json={"algorithm": "kmeans", "n_clusters": 5}
)
print(response.json())
```

### 3. Or Use Python Directly

```python
from pathlib import Path
from core.ingestion.data_loader import DataLoader
from core.preprocessing.data_cleaner import DataCleaner
from core.analysis.clustering import ClusteringModel

# Load your data
loader = DataLoader("path/to/your_data.csv")
df = loader.load_csv()

# Clean
cleaner = DataCleaner()
df_clean = cleaner.preprocess_pipeline(df)

# Cluster
model = ClusteringModel()
model.fit_kmeans(df_clean, n_clusters=5)
df_clustered = model.add_cluster_labels(df_clean)

print(model.get_cluster_statistics(df_clustered))
```

## 🎨 Generating Visualizations

### Heatmap

```python
from core.visualization.visualizer import HeatmapGenerator

heatmap = HeatmapGenerator()
heatmap.create_base_map()
heatmap.add_heatmap_layer(df_clean)
heatmap.add_cluster_markers(df_clean)
heatmap.save_map("output/my_heatmap.html")
```

### Charts

```python
from core.visualization.visualizer import ChartGenerator

charts = ChartGenerator()
charts.create_gender_distribution(df_clean)
charts.create_age_breakdown(df_clean)
charts.create_yearly_trend(df_clean)
charts.save_all_charts("output/charts/")
```

## 📊 Complete Example Workflow

```python
from pathlib import Path
import sys
sys.path.append("src")

from core.ingestion.data_loader import load_and_validate
from core.preprocessing.data_cleaner import DataCleaner
from core.analysis.clustering import ClusteringModel
from core.analysis.predictor import SpatialPredictor
from core.visualization.visualizer import HeatmapGenerator, ChartGenerator

# 1. Load and validate
print("Loading data...")
df, loader = load_and_validate(Path("data/your_data.csv"))

# 2. Clean
print("Cleaning data...")
cleaner = DataCleaner()
df_clean = cleaner.preprocess_pipeline(df)
print(cleaner.get_cleaning_report())

# 3. Clustering
print("Running clustering...")
clustering = ClusteringModel()

# Find optimal k first
optimal = clustering.find_optimal_k(df_clean, k_range=(3, 8))
print("Optimal K results:", optimal)

# Run with chosen k
clustering.fit_kmeans(df_clean, n_clusters=5)
df_clustered = clustering.add_cluster_labels(df_clean)

# Get insights
stats = clustering.get_cluster_statistics(df_clustered)
print("\nCluster Statistics:")
print(stats)

target_groups = clustering.identify_target_groups(df_clustered)
print("\nTarget Groups:")
print(target_groups)

# 4. Prediction
print("\nTraining prediction model...")
predictor = SpatialPredictor()
metrics = predictor.train_hotspot_intensity_predictor(df_clean)
print(f"Model R²: {metrics['test_r2']:.3f}")

predictions = predictor.predict_next_year_hotspots(df_clean, 2026, top_n=10)
print("\nTop 10 Predicted Hotspots for 2026:")
print(predictions)

# 5. Visualizations
print("\nGenerating visualizations...")

# Heatmap
heatmap = HeatmapGenerator()
heatmap.create_base_map()
heatmap.add_heatmap_layer(df_clustered)
heatmap.add_cluster_markers(df_clustered)
heatmap.save_map(Path("output/heatmap.html"))

# Charts
charts = ChartGenerator()
charts.create_gender_distribution(df_clustered)
charts.create_age_breakdown(df_clustered)
charts.create_yearly_trend(df_clustered)
charts.create_monthly_pattern(df_clustered)
charts.create_cluster_size_chart(stats)
charts.save_all_charts(Path("output/charts"))

print("\n✅ Analysis complete!")
print("📁 Output saved to: output/")
```

## 🔍 Common Use Cases

### Find Optimal Number of Clusters

```python
from core.analysis.clustering import ClusteringModel

model = ClusteringModel()
results = model.find_optimal_k(df_clean, k_range=(2, 10))

# Print results
for k, metrics in results.items():
    print(f"k={k}: Silhouette={metrics['silhouette_score']:.3f}")
```

### Analyze Temporal Trends

```python
from core.analysis.predictor import TrendAnalyzer

analyzer = TrendAnalyzer()

# Yearly growth
growth = analyzer.calculate_yearly_growth_rate(df_clean)
print(growth)

# Seasonal patterns
seasonal = analyzer.identify_seasonal_patterns(df_clean)
print(f"Peak month: {seasonal['peak_month']}")
print(f"Seasonal distribution: {seasonal['seasonal_distribution']}")
```

### Get Feature Importance

```python
from core.analysis.predictor import SpatialPredictor

predictor = SpatialPredictor()
predictor.train_hotspot_intensity_predictor(df_clean)

importance = predictor.get_feature_importance()
print("Top 5 important features:")
print(importance.head())
```

## 🐛 Troubleshooting

### "Module not found" errors

Make sure you're in the project root directory and have installed dependencies:

```powershell
pip install -r requirements.txt
```

### API won't start

Check if port 8000 is already in use:

```powershell
# Use a different port
uvicorn src.backend.main:app --reload --port 8080
```

### Heatmap doesn't display

Make sure you have the output file and open it in a web browser:

```powershell
start data/outputs/demo_heatmap.html
```

### Import errors with core modules

Add the src directory to your Python path:

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))
```

## 📚 Next Steps

1. **Run the demo** to understand the workflow
2. **Prepare your CSV data** with required columns
3. **Upload via API** or process directly with Python
4. **Explore visualizations** in the output directory
5. **Customize parameters** (number of clusters, prediction years, etc.)
6. **Export results** for presentations or reports

## 🎓 Learning Resources

- See `README.md` for detailed documentation
- Check `demo.py` for complete example code
- Explore API docs at http://localhost:8000/docs
- Read module docstrings for function details

## 💡 Tips

- Start with the demo to verify everything works
- Use small sample data (100-500 records) for testing
- Save intermediate results (cleaned data, cluster labels)
- Generate visualizations incrementally
- Use API for web integration, Python scripts for analysis

---

**Need help?** Check the main README.md or explore the API documentation.
