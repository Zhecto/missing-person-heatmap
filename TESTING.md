# Testing Guide - Complete System

This guide walks you through testing the entire Missing Person Heatmap Analysis system with the web frontend.

## 🎯 Overview

You now have a **complete full-stack application**:
- **Backend**: FastAPI REST API (Python)
- **Frontend**: Web interface (HTML/CSS/JavaScript)
- **Analysis Engine**: Data mining and ML models

## 📋 Prerequisites

Make sure you have:
- Python 3.10+
- All dependencies installed
- Terminal/PowerShell access

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 2: Generate Demo Data (Optional but Recommended)

```powershell
python demo.py
```

This creates sample data in `data/sample_data.csv` and pre-generates visualizations.

### Step 3: Start the Server

```powershell
uvicorn src.backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Server will start at**: http://localhost:8000

## 🌐 Accessing the Application

### Web Interface
Open your browser and go to:
```
http://localhost:8000
```

You'll see the full web interface with:
- Upload section
- Data preprocessing controls
- Analysis tools (Clustering & Prediction)
- Visualization generators
- Real-time status tracking

### API Documentation
For API testing and exploration:
```
http://localhost:8000/docs        (Swagger UI)
http://localhost:8000/redoc       (ReDoc)
http://localhost:8000/api         (API info)
```

## 🧪 Testing Workflow

### Option A: Using Demo Data (Recommended First)

1. **Start the server** (see Step 3 above)

2. **Run demo.py separately** to generate sample data:
   ```powershell
   python demo.py
   ```

3. **Open the web interface**: http://localhost:8000

4. **In the web UI**:
   - Click "Use Demo Data" button
   - Or manually upload `data/sample_data.csv` using the file upload

5. **Click through the workflow**:
   - ✅ Upload Data → See validation results
   - ✅ Clean Data → View preprocessing report
   - ✅ Run Clustering → Choose K-means with 5 clusters
   - ✅ Train & Predict → Set target year to 2026
   - ✅ Generate Heatmap → View interactive map
   - ✅ Generate Charts → Create all visualizations

### Option B: Using Your Own Data

1. **Prepare your CSV** with required columns:
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

2. **Start the server**

3. **Open web interface**: http://localhost:8000

4. **Upload your CSV**:
   - Click "Select CSV File"
   - Choose your file
   - Click "Upload & Load Data"

5. **Follow the same workflow** as Option A

## 📊 Feature Testing Checklist

### Data Management
- [ ] Upload CSV file
- [ ] View validation results
- [ ] Clean and preprocess data
- [ ] View data summary statistics
- [ ] Check pipeline status indicators

### Clustering Analysis
- [ ] Select K-means algorithm
- [ ] Set number of clusters (try 3-7)
- [ ] Run clustering
- [ ] View cluster statistics table
- [ ] View target groups identified
- [ ] Try "Find Optimal K" feature
- [ ] Switch to DBSCAN and test

### Predictive Modeling
- [ ] Set target year for prediction
- [ ] Train prediction model
- [ ] View model performance metrics (R², RMSE)
- [ ] View top predicted hotspots
- [ ] Analyze trends (yearly/seasonal)

### Visualizations
- [ ] Generate interactive heatmap
- [ ] Generate statistical charts
- [ ] View heatmap preview (if available)
- [ ] Check output files in `data/outputs/`

## 🎨 Web Interface Features

### Real-Time Status Bar
At the top, you'll see 4 status indicators:
- 📤 Data Loaded
- 🧹 Data Cleaned  
- 🔍 Clustering Done
- 📈 Prediction Ready

Each lights up green when completed.

### Interactive Cards
- **Upload & Preprocess**: File handling and data cleaning
- **Clustering Analysis**: Algorithm selection and execution
- **Predictive Modeling**: Future hotspot prediction
- **Visualizations**: Map and chart generation

### Results Tables
Click the tabs to see:
- Cluster Statistics
- Target Groups
- Predictions

## 🔍 Testing Scenarios

### Scenario 1: First-Time User Experience
```
1. Open http://localhost:8000
2. Click "Use Demo Data" 
3. Click "Clean Data"
4. Click "Run Clustering" (K-means, 5 clusters)
5. Click "Generate Heatmap"
6. View results in tables and status bar
```

### Scenario 2: Full Analysis Pipeline
```
1. Upload CSV file
2. Clean data and review report
3. Find optimal K (2-10 range)
4. Run clustering with optimal K
5. Train prediction model for 2026
6. Analyze trends
7. Generate all visualizations
8. Download output files
```

### Scenario 3: Comparing Algorithms
```
1. Upload/clean data
2. Run K-means with k=5
3. Note results
4. Switch to DBSCAN
5. Adjust eps and min_samples
6. Compare cluster outcomes
```

## 🐛 Troubleshooting

### Server Won't Start

**Error: "Address already in use"**
```powershell
# Use a different port
uvicorn src.backend.main:app --reload --port 8080
```

Then access at: http://localhost:8080

**Error: "Module not found"**
```powershell
# Reinstall dependencies
pip install -r requirements.txt
```

### Frontend Issues

**Page shows "API is running. Frontend not found."**
- Check that files exist in `src/frontend/`
- Verify `index.html`, `app.js`, `style.css` are present

**Buttons are disabled**
- Upload data first
- Each step enables the next
- Check status bar for current state

**API calls fail**
- Verify backend is running
- Check browser console (F12) for errors
- Ensure no CORS issues

### Data Upload Issues

**"Failed to upload data"**
- Check CSV format and required columns
- File size limit (adjust if needed)
- Verify file encoding (UTF-8 recommended)

**Validation errors**
- Review validation report in upload result
- Check coordinate ranges (Manila bounds)
- Ensure date formats are correct

### Visualization Issues

**Heatmap won't display in preview**
- File path issues with local files
- Open directly: `data/outputs/heatmap.html`
- Use browser's "Open File" option

**Charts not generating**
- Check backend logs for errors
- Verify data is cleaned first
- Check `data/outputs/charts/` directory

## 📁 Output Files

After running the complete workflow, check these locations:

```
data/
├── outputs/
│   ├── heatmap.html              # Interactive map
│   └── charts/
│       ├── chart_1.html          # Gender distribution
│       ├── chart_2.html          # Age breakdown
│       ├── chart_3.html          # Yearly trend
│       ├── chart_4.html          # Monthly pattern
│       ├── chart_5.html          # Location heatmap
│       └── chart_6.html          # Cluster sizes
└── sample_data.csv               # Demo data (if generated)
```

## 🎓 Advanced Testing

### API Testing with curl

**Check health:**
```powershell
curl http://localhost:8000/health
```

**Get status:**
```powershell
curl http://localhost:8000/api/data/status
```

**Upload file:**
```powershell
curl -X POST -F "file=@data/sample_data.csv" http://localhost:8000/api/data/upload
```

### Testing with Python

```python
import requests

# Check API
response = requests.get('http://localhost:8000/api')
print(response.json())

# Upload data
with open('data/sample_data.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/api/data/upload', files=files)
    print(response.json())

# Run clustering
data = {'algorithm': 'kmeans', 'n_clusters': 5}
response = requests.post('http://localhost:8000/api/analysis/clustering', json=data)
print(response.json())
```

## 💡 Tips for Best Testing Experience

1. **Start with demo data** - Ensures everything works before using real data
2. **Follow the pipeline order** - Upload → Clean → Analyze → Visualize
3. **Check status bar** - Shows what's enabled/disabled
4. **Review tables** - Detailed results appear in tabs
5. **Open output files** - View heatmaps and charts in browser
6. **Use API docs** - Test endpoints at `/docs`
7. **Monitor terminal** - Backend logs show detailed info

## 🎉 Success Indicators

You've successfully tested the system when you see:

✅ All 4 status indicators are green  
✅ Data summary shows your records  
✅ Cluster statistics table is populated  
✅ Target groups are identified  
✅ Predictions table shows hotspots  
✅ Heatmap.html opens and displays map  
✅ Charts folder contains 6 HTML files  

## 📞 Getting Help

If you encounter issues:

1. **Check terminal output** - Backend logs show errors
2. **Check browser console** - F12 for frontend errors
3. **Review this guide** - Solutions for common issues
4. **Check file locations** - Ensure all files exist
5. **Verify dependencies** - Run `pip list` to check installations

## 🚀 Next Steps After Testing

Once testing is complete:

1. **Replace demo data** with your actual dataset
2. **Adjust parameters** based on your data characteristics
3. **Export results** for presentations/reports
4. **Customize frontend** (colors, logos, etc.)
5. **Deploy** to a server for wider access

---

**Ready to test?** Start the server and open http://localhost:8000! 🎊
