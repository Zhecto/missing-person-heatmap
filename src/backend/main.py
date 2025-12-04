"""
FastAPI backend application for missing person heatmap analysis.
Provides REST API endpoints for data processing, analysis, and visualization.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Import core modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.ingestion.data_loader import DataLoader, load_and_validate
from core.preprocessing.data_cleaner import DataCleaner
from core.analysis.clustering import ClusteringModel
from core.analysis.predictor import SpatialPredictor, TrendAnalyzer
from core.visualization.visualizer import HeatmapGenerator, ChartGenerator


# Initialize FastAPI app
app = FastAPI(
    title="Missing Person Heatmap API",
    description="API for analyzing and visualizing missing person data in Manila",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# Global state (in production, use proper state management)
class AppState:
    """Application state container."""
    df_raw: Optional[pd.DataFrame] = None
    df_cleaned: Optional[pd.DataFrame] = None
    clustering_model: Optional[ClusteringModel] = None
    predictor_model: Optional[SpatialPredictor] = None
    data_loader: Optional[DataLoader] = None
    cleaner: Optional[DataCleaner] = None

state = AppState()


# Pydantic models for request/response
class AnalysisStatus(BaseModel):
    """Status of the analysis pipeline."""
    data_loaded: bool
    data_cleaned: bool
    clustering_done: bool
    prediction_ready: bool
    total_records: int
    clean_records: int

class ClusteringRequest(BaseModel):
    """Request model for clustering."""
    algorithm: str = "kmeans"  # "kmeans" or "dbscan"
    n_clusters: Optional[int] = 5
    eps: Optional[float] = 0.01
    min_samples: Optional[int] = 5

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    target_year: int
    top_n: int = 10


# Root endpoint - serve frontend
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend application."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        with open(frontend_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>Missing Person Heatmap API</h1>
                <p>API is running. Frontend not found.</p>
                <p>Access API docs at: <a href="/docs">/docs</a></p>
            </body>
        </html>
        """)

@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "message": "Missing Person Heatmap API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "data": "/api/data/*",
            "analysis": "/api/analysis/*",
            "visualization": "/api/visualization/*",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Data endpoints
@app.post("/api/data/upload")
async def upload_data(file: UploadFile = File(...)):
    """
    Upload and load CSV data file.
    
    Args:
        file: CSV file with missing persons data
        
    Returns:
        Status and summary of loaded data
    """
    try:
        # Save uploaded file temporarily
        temp_path = Path("data/temp_upload.csv")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Load and validate
        state.data_loader = DataLoader(temp_path)
        state.df_raw = state.data_loader.load_csv()
        
        # Validate
        schema_valid = state.data_loader.validate_schema()
        coords_valid = state.data_loader.validate_coordinates()
        
        # Get summary
        summary = state.data_loader.get_data_summary()
        
        return {
            "status": "success",
            "message": "Data loaded successfully",
            "validation": {
                "schema_valid": schema_valid,
                "coordinates_valid": coords_valid,
                "errors": state.data_loader.validation_errors
            },
            "summary": summary
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to upload data: {str(e)}")


@app.post("/api/data/clean")
async def clean_data():
    """
    Clean and preprocess the loaded data.
    
    Returns:
        Cleaning report and statistics
    """
    if state.df_raw is None:
        raise HTTPException(status_code=400, detail="No data loaded. Upload data first.")
    
    try:
        state.cleaner = DataCleaner()
        state.df_cleaned = state.cleaner.preprocess_pipeline(state.df_raw)
        
        return {
            "status": "success",
            "message": "Data cleaned successfully",
            "report": state.cleaner.get_cleaning_report(),
            "records_before": len(state.df_raw),
            "records_after": len(state.df_cleaned)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clean data: {str(e)}")


@app.get("/api/data/summary")
async def get_data_summary():
    """
    Get summary statistics of loaded data.
    
    Returns:
        Summary statistics
    """
    if state.df_cleaned is None:
        raise HTTPException(status_code=400, detail="No cleaned data available")
    
    df = state.df_cleaned
    
    summary = {
        "total_records": len(df),
        "date_range": {
            "earliest": str(df['Date Reported Missing'].min()) if 'Date Reported Missing' in df.columns else None,
            "latest": str(df['Date Reported Missing'].max()) if 'Date Reported Missing' in df.columns else None
        },
        "gender_distribution": df['Gender'].value_counts().to_dict() if 'Gender' in df.columns else {},
        "age_statistics": {
            "mean": float(df['Age'].mean()) if 'Age' in df.columns else None,
            "median": float(df['Age'].median()) if 'Age' in df.columns else None,
            "min": float(df['Age'].min()) if 'Age' in df.columns else None,
            "max": float(df['Age'].max()) if 'Age' in df.columns else None
        },
        "locations": {
            "unique_barangays": int(df['Barangay District'].nunique()) if 'Barangay District' in df.columns else 0,
            "top_10": df['Barangay District'].value_counts().head(10).to_dict() if 'Barangay District' in df.columns else {}
        }
    }
    
    return summary


@app.get("/api/data/status")
async def get_status() -> AnalysisStatus:
    """
    Get current status of the analysis pipeline.
    
    Returns:
        Status information
    """
    return AnalysisStatus(
        data_loaded=state.df_raw is not None,
        data_cleaned=state.df_cleaned is not None,
        clustering_done=state.clustering_model is not None and state.clustering_model.labels_ is not None,
        prediction_ready=state.predictor_model is not None and state.predictor_model.is_fitted,
        total_records=len(state.df_raw) if state.df_raw is not None else 0,
        clean_records=len(state.df_cleaned) if state.df_cleaned is not None else 0
    )


# Analysis endpoints
@app.post("/api/analysis/clustering")
async def run_clustering(request: ClusteringRequest):
    """
    Perform clustering analysis on the data.
    
    Args:
        request: Clustering configuration
        
    Returns:
        Clustering results and statistics
    """
    if state.df_cleaned is None:
        raise HTTPException(status_code=400, detail="No cleaned data available")
    
    try:
        state.clustering_model = ClusteringModel()
        
        if request.algorithm == "kmeans":
            state.clustering_model.fit_kmeans(
                state.df_cleaned,
                n_clusters=request.n_clusters
            )
        elif request.algorithm == "dbscan":
            state.clustering_model.fit_dbscan(
                state.df_cleaned,
                eps=request.eps,
                min_samples=request.min_samples
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid algorithm. Use 'kmeans' or 'dbscan'")
        
        # Add labels to dataframe
        state.df_cleaned = state.clustering_model.add_cluster_labels(state.df_cleaned)
        
        # Get statistics
        cluster_stats = state.clustering_model.get_cluster_statistics(state.df_cleaned)
        evaluation = state.clustering_model.evaluate_clustering(state.df_cleaned)
        target_groups = state.clustering_model.identify_target_groups(state.df_cleaned)
        
        return {
            "status": "success",
            "algorithm": request.algorithm,
            "n_clusters": state.clustering_model.n_clusters,
            "cluster_statistics": cluster_stats.to_dict(orient='records'),
            "evaluation_metrics": evaluation,
            "target_groups": target_groups.to_dict(orient='records')
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")


@app.get("/api/analysis/optimal-clusters")
async def find_optimal_clusters(min_k: int = 2, max_k: int = 10):
    """
    Find optimal number of clusters using elbow method.
    
    Args:
        min_k: Minimum number of clusters to test
        max_k: Maximum number of clusters to test
        
    Returns:
        Evaluation metrics for different k values
    """
    if state.df_cleaned is None:
        raise HTTPException(status_code=400, detail="No cleaned data available")
    
    try:
        model = ClusteringModel()
        results = model.find_optimal_k(state.df_cleaned, k_range=(min_k, max_k))
        
        return {
            "status": "success",
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.post("/api/analysis/predict")
async def run_prediction(request: PredictionRequest):
    """
    Train prediction model and predict future hotspots.
    
    Args:
        request: Prediction configuration
        
    Returns:
        Prediction results
    """
    if state.df_cleaned is None:
        raise HTTPException(status_code=400, detail="No cleaned data available")
    
    try:
        state.predictor_model = SpatialPredictor()
        
        # Train model
        metrics = state.predictor_model.train_hotspot_intensity_predictor(state.df_cleaned)
        
        # Predict
        predictions = state.predictor_model.predict_next_year_hotspots(
            state.df_cleaned,
            next_year=request.target_year,
            top_n=request.top_n
        )
        
        # Get feature importance
        feature_importance = state.predictor_model.get_feature_importance()
        
        return {
            "status": "success",
            "target_year": request.target_year,
            "training_metrics": metrics,
            "predictions": predictions.to_dict(orient='records'),
            "feature_importance": feature_importance.to_dict(orient='records')
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/analysis/trends")
async def analyze_trends():
    """
    Analyze temporal trends in the data.
    
    Returns:
        Trend analysis results
    """
    if state.df_cleaned is None:
        raise HTTPException(status_code=400, detail="No cleaned data available")
    
    try:
        analyzer = TrendAnalyzer()
        
        # Calculate growth rates
        growth_rates = analyzer.calculate_yearly_growth_rate(state.df_cleaned)
        
        # Identify seasonal patterns
        seasonal = analyzer.identify_seasonal_patterns(state.df_cleaned)
        
        return {
            "status": "success",
            "yearly_growth": growth_rates.to_dict(orient='records'),
            "seasonal_patterns": seasonal
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")


# Visualization endpoints
@app.get("/api/visualization/heatmap")
async def generate_heatmap(include_clusters: bool = False):
    """
    Generate interactive heatmap.
    
    Args:
        include_clusters: Whether to include cluster markers
        
    Returns:
        Path to generated heatmap HTML
    """
    if state.df_cleaned is None:
        raise HTTPException(status_code=400, detail="No cleaned data available")
    
    try:
        generator = HeatmapGenerator()
        generator.create_base_map()
        generator.add_heatmap_layer(state.df_cleaned)
        
        if include_clusters and 'Cluster' in state.df_cleaned.columns:
            generator.add_cluster_markers(state.df_cleaned)
        
        # Save map
        output_path = Path("data/outputs/heatmap.html")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generator.save_map(output_path)
        
        return {
            "status": "success",
            "message": "Heatmap generated",
            "file_path": str(output_path)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {str(e)}")


@app.get("/api/visualization/charts")
async def generate_charts():
    """
    Generate all statistical charts.
    
    Returns:
        Paths to generated charts
    """
    if state.df_cleaned is None:
        raise HTTPException(status_code=400, detail="No cleaned data available")
    
    try:
        generator = ChartGenerator()
        
        # Generate charts
        generator.create_gender_distribution(state.df_cleaned)
        generator.create_age_breakdown(state.df_cleaned)
        generator.create_yearly_trend(state.df_cleaned)
        generator.create_monthly_pattern(state.df_cleaned)
        generator.create_location_heatmap_chart(state.df_cleaned)
        
        # Save charts
        output_dir = Path("data/outputs/charts")
        generator.save_all_charts(output_dir)
        
        return {
            "status": "success",
            "message": f"Generated {len(generator.figures)} charts",
            "output_directory": str(output_dir)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")


@app.get("/api/visualization/download/{file_type}")
async def download_visualization(file_type: str):
    """
    Download generated visualization files.
    
    Args:
        file_type: Type of file to download ('heatmap' or 'charts')
        
    Returns:
        File response
    """
    if file_type == "heatmap":
        file_path = Path("data/outputs/heatmap.html")
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Heatmap not generated yet")
        return FileResponse(file_path, filename="heatmap.html")
    
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")


# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
