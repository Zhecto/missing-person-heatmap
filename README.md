# Missing Person Heatmap Analysis

**Pattern Analysis and Prediction of Missing Persons Using Data Mining Techniques**

A comprehensive data mining system for analyzing missing person cases in Manila, Philippines. This project uses clustering, predictive modeling, and interactive visualizations to identify hotspots and patterns.

## Project Overview

This system processes missing persons data to:
- Generate interactive heatmaps showing high-risk areas
- Analyze demographic patterns (gender, age, location)
- Identify clusters of similar cases using machine learning
- Predict future hotspot locations
- Track temporal trends and seasonal patterns

## Input/Process/Output

### INPUT
- Cleaned CSV dataset of missing persons in Manila with fields:
  - Person ID, Gender, Age
  - Date/Time Reported Missing
  - Location, Latitude, Longitude
  - Barangay District, Post URL

### PROCESS
1. **Load Data** - CSV validation and schema checking
2. **Preprocessing** - Handle missing values, encode categories, standardize
3. **Clustering** - K-means/DBSCAN to identify hotspots
4. **Visualization** - Generate heatmaps and statistical charts
5. **Prediction** - Train models to predict next-year hotspots

### OUTPUT
- Interactive heatmap showing high-risk areas
- Charts: gender distribution, age breakdown, yearly trends
- Cluster groups (e.g., "Young Females", "Elderly Males")
- Predictive hotspot map for next year

## Quick Start (Without Dataset)

You can explore the system functionality without actual data using the demo script:

### 1. Install Dependencies

```powershell
# Using pip
pip install -r requirements.txt

# Or using Poetry
poetry install
```

### 2. Run Demo (Optional but Recommended)

```powershell
python demo.py
```

This will:
- Generate 500 sample missing person records
- Run the complete analysis pipeline
- Create visualizations in `data/outputs/`

### 3. Start the Web Application

```powershell
# Option A: Using the startup script
.\start-server.ps1

# Option B: Direct command
uvicorn src.backend.main:app --reload
```

Then open your browser to **http://localhost:8000**

You'll see the complete web interface where you can:
- ✅ Upload CSV data or use demo data
- ✅ Run preprocessing and cleaning
- ✅ Perform clustering analysis (K-means/DBSCAN)
- ✅ Train prediction models
- ✅ Generate interactive heatmaps and charts
- ✅ View results in real-time

**API Documentation**: http://localhost:8000/docs

### 2. Run Demo

```powershell
python demo.py
```

This will:
- Generate 500 sample missing person records
- Run the complete analysis pipeline
- Create visualizations in `data/outputs/`
- Show you exactly how the system works

### 3. Start API Server

```powershell
uvicorn src.backend.main:app --reload
```

Access the API at:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Project Structure

```
missing-person-heatmap/
├── src/
│   ├── core/
│   │   ├── ingestion/
│   │   │   └── data_loader.py          # CSV loading and validation
│   │   ├── preprocessing/
│   │   │   └── data_cleaner.py         # Data cleaning pipeline
│   │   ├── analysis/
│   │   │   ├── clustering.py           # K-means/DBSCAN clustering
│   │   │   └── predictor.py            # Spatial prediction models
│   │   └── visualization/
│   │       └── visualizer.py           # Heatmaps and charts
│   ├── backend/
│   │   └── main.py                     # FastAPI application
│   └── frontend/
│       ├── index.html                  # Web interface
│       ├── app.js                      # Frontend logic
│       └── style.css                   # Styling
├── data/
│   ├── outputs/                        # Generated visualizations
│   └── sample_data.csv                 # Demo data
├── notebooks/                          # Jupyter notebooks for exploration
├── config/
│   └── settings.yaml                   # Configuration
├── demo.py                             # Demo script
├── start-server.ps1                    # Server startup script
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Poetry configuration
├── TESTING.md                          # Complete testing guide
└── README.md                           # This file
```

## 🔧 Core Modules

### 1. Data Loading (`core/ingestion/data_loader.py`)

```python
from core.ingestion.data_loader import DataLoader

loader = DataLoader("path/to/data.csv")
df = loader.load_csv()
loader.validate_schema()
loader.validate_coordinates()
summary = loader.get_data_summary()
```

### 2. Data Preprocessing (`core/preprocessing/data_cleaner.py`)

```python
from core.preprocessing.data_cleaner import DataCleaner

cleaner = DataCleaner()
df_clean = cleaner.preprocess_pipeline(df)
print(cleaner.get_cleaning_report())
```

Features:
- Smart missing value handling
- Date parsing and temporal feature extraction
- Categorical encoding
- Duplicate removal
- Age group classification

### 3. Clustering Analysis (`core/analysis/clustering.py`)

```python
from core.analysis.clustering import ClusteringModel

# K-means clustering
model = ClusteringModel()
model.fit_kmeans(df_clean, n_clusters=5)
df_clustered = model.add_cluster_labels(df_clean)

# Get cluster statistics
stats = model.get_cluster_statistics(df_clustered)
target_groups = model.identify_target_groups(df_clustered)

# Find optimal K
optimal = model.find_optimal_k(df_clean, k_range=(2, 10))
```

### 4. Predictive Modeling (`core/analysis/predictor.py`)

```python
from core.analysis.predictor import SpatialPredictor

predictor = SpatialPredictor()
metrics = predictor.train_hotspot_intensity_predictor(df_clean)
predictions = predictor.predict_next_year_hotspots(df_clean, next_year=2026)
feature_importance = predictor.get_feature_importance()
```

### 5. Visualization (`core/visualization/visualizer.py`)

```python
from core.visualization.visualizer import HeatmapGenerator, ChartGenerator

# Heatmap
heatmap = HeatmapGenerator()
heatmap.create_base_map()
heatmap.add_heatmap_layer(df_clean)
heatmap.add_cluster_markers(df_clean)
heatmap.save_map("output/heatmap.html")

# Charts
charts = ChartGenerator()
charts.create_gender_distribution(df_clean)
charts.create_age_breakdown(df_clean)
charts.create_yearly_trend(df_clean)
charts.save_all_charts("output/charts/")
```

## API Endpoints

### Data Management
- `POST /api/data/upload` - Upload CSV file
- `POST /api/data/clean` - Clean and preprocess data
- `GET /api/data/summary` - Get data statistics
- `GET /api/data/status` - Check pipeline status

### Analysis
- `POST /api/analysis/clustering` - Run clustering analysis
- `GET /api/analysis/optimal-clusters` - Find optimal K
- `POST /api/analysis/predict` - Train and predict hotspots
- `GET /api/analysis/trends` - Analyze temporal trends

### Visualization
- `GET /api/visualization/heatmap` - Generate heatmap
- `GET /api/visualization/charts` - Generate all charts
- `GET /api/visualization/download/{type}` - Download files

## Example Usage with Real Data

Once you have your actual CSV dataset:

```python
from pathlib import Path
from core.ingestion.data_loader import load_and_validate
from core.preprocessing.data_cleaner import DataCleaner
from core.analysis.clustering import ClusteringModel
from core.analysis.predictor import SpatialPredictor
from core.visualization.visualizer import HeatmapGenerator

# 1. Load and validate
df, loader = load_and_validate(Path("data/missing_persons.csv"))

# 2. Clean data
cleaner = DataCleaner()
df_clean = cleaner.preprocess_pipeline(df)

# 3. Clustering
clustering = ClusteringModel()
clustering.fit_kmeans(df_clean, n_clusters=5)
df_clustered = clustering.add_cluster_labels(df_clean)
print(clustering.get_cluster_statistics(df_clustered))

# 4. Prediction
predictor = SpatialPredictor()
predictor.train_hotspot_intensity_predictor(df_clean)
predictions = predictor.predict_next_year_hotspots(df_clean, 2026)
print(predictions)

# 5. Visualization
heatmap = HeatmapGenerator()
heatmap.create_base_map()
heatmap.add_heatmap_layer(df_clustered)
heatmap.add_cluster_markers(df_clustered)
heatmap.save_map(Path("output/heatmap.html"))
```

## 🛠️ Technology Stack

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Geospatial**: folium, geopandas, shapely
- **Visualization**: plotly, matplotlib, seaborn
- **Web Framework**: FastAPI, uvicorn
- **Data Validation**: pydantic

## 📈 Methodology

### 1. Data Collection
Gather real-world missing persons data from Facebook posts and pages dedicated to missing persons reports.

### 2. Data Preprocessing
- Handle missing/inconsistent entries
- Convert dates to usable formats
- Encode categorical variables
- Remove duplicates
- Standardize data for consistency

### 3. Exploratory Data Analysis (EDA)
Initial exploration to understand trends and distributions.

### 4. Spatial Analysis (Heat Mapping)
Map each reported case onto Manila using latitude/longitude coordinates.
- Obtain GeoJSON boundary map of Manila
- Plot missing person reports
- Generate heatmap showing hotspots (high/moderate/low density areas)

Tools: Leaflet.js, Folium, or Kepler.gl

### 5. Pattern Analysis using Clustering
Use unsupervised machine learning (K-means or DBSCAN) to detect underlying pattern groups.

### 6. Predictive Modeling
Predict which barangays will likely record more missing person cases next year using spatial prediction models.

## 📝 Expected Results

1. **Hotspot Identification**: Certain barangays show consistently high numbers of missing person cases
2. **Demographic Patterns**: More missing reports among teenagers and young adults (15-30 age range)
3. **Gender Imbalance**: Higher percentage of missing females compared to males
4. **Temporal Trends**: Seasonal variations in report frequency
5. **Target Groups**: Clusters like "Young Females in Urban Areas" or "Elderly Males in Suburban Districts"

## 🔄 Workflow

```
Data Collection → Data Preprocessing → EDA → Heatmap Spatial Analysis → 
Clustering → Prediction → Web App Integration → Final Insights
```

## 🚦 Getting Started with Your Data

1. **Prepare your CSV** with the required columns (see INPUT section)
2. **Place CSV in `data/` directory**
3. **Run the analysis**:

```powershell
# Option 1: Using Python directly
python demo.py

# Option 2: Using the API
uvicorn src.backend.main:app --reload
# Then use the /api/data/upload endpoint
```

4. **View results** in `data/outputs/`

## 🐛 Troubleshooting

**Import Errors**: Make sure all dependencies are installed
```powershell
pip install -r requirements.txt
```

**No data loaded**: Upload data via API or place CSV in `data/` directory

**Visualization not showing**: Check that output files are generated in `data/outputs/`

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Folium Documentation](https://python-visualization.github.io/folium/)
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Plotly Documentation](https://plotly.com/python/)

## 🤝 Contributing

This is an academic project for Data Mining course. Contributions and suggestions are welcome!

## 📄 License

Academic project - all rights reserved.

---

**Project**: 4th Year, 1st Semester - Data Mining  
**Topic**: Pattern Analysis and Prediction of Missing Persons Using Data Mining Techniques
