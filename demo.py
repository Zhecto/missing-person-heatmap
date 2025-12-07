"""
Demo script showing how to use the missing person heatmap system.
This demonstrates the complete workflow without requiring actual data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from core.ingestion.data_loader import DataLoader
from core.preprocessing.data_cleaner import DataCleaner
from core.analysis.clustering import ClusteringModel
from core.analysis.predictor import SpatialPredictor, TrendAnalyzer
from core.visualization.visualizer import HeatmapGenerator, ChartGenerator


def generate_sample_data(n_records: int = 500, include_2025: bool = True) -> pd.DataFrame:
    """
    Generate sample missing persons data for Metro Manila (NCR).
    
    Args:
        n_records: Number of records to generate for historical data (2019-2024)
        include_2025: If True, generate additional ~100 records for 2025
        
    Returns:
        Sample DataFrame
    """
    total_records = n_records + (100 if include_2025 else 0)
    print(f"\nGenerating {total_records} sample records for Metro Manila (NCR)...")
    if include_2025:
        print(f"  - Historical data (2019-2024): {n_records} records")
        print(f"  - Future data (2025): 100 records")
    
    np.random.seed(42)
    
    # Metro Manila (NCR) coordinates (expanded bounds covering 16 cities + 1 municipality)
    lat_min, lat_max = 14.35, 14.85
    lon_min, lon_max = 120.90, 121.15
    
    # Generate hotspot centers across different NCR cities
    hotspots = [
        (14.5995, 120.9842),  # Manila City center
        (14.6091, 121.0223),  # Makati
        (14.6760, 121.0437),  # Quezon City (north)
        (14.6507, 121.0494),  # Quezon City (central)
        (14.5547, 121.0244),  # Mandaluyong
        (14.5764, 120.9772),  # Manila - Ermita
        (14.5350, 121.0500),  # Pasig
        (14.6504, 120.9830),  # Caloocan
        (14.4500, 121.0400),  # Taguig/BGC
    ]
    
    data = []
    start_date = datetime(2019, 1, 1)  # Starting from 2019
    
    # Barangays from different cities across Metro Manila
    barangays = [
        # Manila City
        'Tondo', 'Pandacan', 'Santa Cruz', 'Sampaloc', 'Quiapo', 'Binondo', 'Ermita', 'Malate',
        'San Miguel', 'Sta. Mesa', 'Port Area', 'Intramuros', 'Paco', 'San Nicolas',
        # Quezon City
        'Commonwealth', 'Batasan Hills', 'Fairview', 'Novaliches', 'Cubao', 'Diliman',
        # Makati
        'Poblacion', 'Bel-Air', 'San Lorenzo', 'Pio del Pilar',
        # Pasig
        'Kapitolyo', 'Rosario', 'Manggahan', 'Santolan',
        # Caloocan
        'Bagong Silang', 'Camarin', 'Bagumbong',
        # Taguig
        'Fort Bonifacio', 'Western Bicutan', 'Upper Bicutan',
        # Mandaluyong
        'Highway Hills', 'Wack-Wack', 'Plainview',
        # Other NCR areas
        'Marikina Heights', 'Pasay', 'Parañaque', 'Las Piñas', 'Muntinlupa'
    ]
    
    for i in range(n_records):
        # Select a random hotspot and add noise
        hotspot = hotspots[np.random.choice(len(hotspots))]
        lat = np.random.normal(hotspot[0], 0.02)
        lon = np.random.normal(hotspot[1], 0.02)
        
        # Ensure within Manila bounds
        lat = np.clip(lat, lat_min, lat_max)
        lon = np.clip(lon, lon_min, lon_max)
        
        # Random date within range (2019-2024, ~6 years of data)
        days_offset = np.random.randint(0, 365 * 6)
        report_date = start_date + timedelta(days=days_offset)
        
        # Generate demographics
        age = np.random.choice([
            np.random.randint(0, 12),    # Children
            np.random.randint(13, 17),   # Teens
            np.random.randint(18, 30),   # Young adults
            np.random.randint(31, 50),   # Adults
            np.random.randint(51, 80),   # Seniors
        ], p=[0.15, 0.25, 0.35, 0.15, 0.10])
        
        gender = np.random.choice(['Male', 'Female'], p=[0.45, 0.55])
        
        record = {
            'Person ID': f'MP{i+1:05d}',
            'Gender': gender,
            'Age': age,
            'Date Reported Missing': report_date.strftime('%Y-%m-%d %H:%M:%S'),
            'Time Reported Missing': f"{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}",
            'Location last seen': f"Street {np.random.randint(1, 100)}",
            'Latitude': lat,
            'Longitude': lon,
            'Barangay District': np.random.choice(barangays),
            'Post URL': f'https://example.com/post/{i+1}'
        }
        
        data.append(record)
    
    # Generate 2025 data if requested
    if include_2025:
        print("  Historical data (2019-2024) generated")
        print("  Generating 2025 predictions...")
        
        start_2025 = datetime(2025, 1, 1)
        
        for i in range(100):
            # Select a random hotspot and add noise
            hotspot = hotspots[np.random.choice(len(hotspots))]
            lat = np.random.normal(hotspot[0], 0.02)
            lon = np.random.normal(hotspot[1], 0.02)
            
            # Ensure within Manila bounds
            lat = np.clip(lat, lat_min, lat_max)
            lon = np.clip(lon, lon_min, lon_max)
            
            # Random date in 2025
            days_offset = np.random.randint(0, 365)
            report_date = start_2025 + timedelta(days=days_offset)
            
            # Generate demographics (same distribution as historical)
            age = np.random.choice([
                np.random.randint(0, 12),    # Children
                np.random.randint(13, 17),   # Teens
                np.random.randint(18, 30),   # Young adults
                np.random.randint(31, 50),   # Adults
                np.random.randint(51, 80),   # Seniors
            ], p=[0.15, 0.25, 0.35, 0.15, 0.10])
            
            gender = np.random.choice(['Male', 'Female'], p=[0.45, 0.55])
            
            record = {
                'Person ID': f'MP{n_records + i + 1:05d}',
                'Gender': gender,
                'Age': age,
                'Date Reported Missing': report_date.strftime('%Y-%m-%d %H:%M:%S'),
                'Time Reported Missing': f"{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}",
                'Location last seen': f"Street {np.random.randint(1, 100)}",
                'Latitude': lat,
                'Longitude': lon,
                'Barangay District': np.random.choice(barangays),
                'Post URL': f'https://example.com/post/{n_records + i + 1}'
            }
            
            data.append(record)
        
        print(" 2025 data generated")
    
    df = pd.DataFrame(data)
    print("Complete dataset generated")
    
    return df


def demo_pipeline():
    """Run complete demo of the analysis pipeline."""
    
    print("="*70)
    print("MISSING PERSON HEATMAP ANALYSIS - DEMO")
    print("Metro Manila (National Capital Region)")
    print("="*70)
    
    # Step 1: Generate sample data
    print("\n[STEP 1] DATA GENERATION")
    print("-" * 70)
    df = generate_sample_data(n_records=500)
    print(f"Generated dataset shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Step 2: Data Loading and Validation
    print("\n[STEP 2] DATA LOADING & VALIDATION")
    print("-" * 70)
    
    # Save sample data
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    sample_path = data_dir / "sample_data.csv"
    df.to_csv(sample_path, index=False)
    print(f"✓ Saved sample data to {sample_path}")
    
    # Load with DataLoader
    loader = DataLoader(sample_path)
    df_loaded = loader.load_csv()
    
    # Validate
    loader.validate_schema()
    loader.validate_coordinates()
    print(loader.get_validation_report())
    
    # Get summary
    summary = loader.get_data_summary()
    print(f"\nData Summary:")
    print(f"  Total Records: {summary['total_records']}")
    print(f"  Unique Locations: {summary['unique_locations']}")
    print(f"  Gender Distribution: {summary['gender_distribution']}")
    
    # Step 3: Data Preprocessing
    print("\n[STEP 3] DATA PREPROCESSING")
    print("-" * 70)
    
    cleaner = DataCleaner()
    df_clean = cleaner.preprocess_pipeline(df_loaded)
    print(cleaner.get_cleaning_report())
    
    # Step 4: Clustering Analysis
    print("\n[STEP 4] CLUSTERING ANALYSIS")
    print("-" * 70)
    
    # Find optimal k
    clustering = ClusteringModel()
    print("Finding optimal number of clusters...")
    optimal_results = clustering.find_optimal_k(df_clean, k_range=(3, 8))
    
    print("\nOptimal K Analysis:")
    for k, metrics in optimal_results.items():
        print(f"  k={k}: Silhouette={metrics['silhouette_score']:.3f}, Inertia={metrics['inertia']:.0f}")
    
    # Run K-means with optimal k
    print("\n🔍 Running K-means clustering with k=5...")
    clustering.fit_kmeans(df_clean, n_clusters=5)
    df_clean = clustering.add_cluster_labels(df_clean)
    
    # Get cluster statistics
    cluster_stats = clustering.get_cluster_statistics(df_clean)
    print("\nCluster Statistics:")
    print(cluster_stats[['Cluster', 'Size', 'Percentage', 'Top_Location']])
    
    # Identify target groups
    target_groups = clustering.identify_target_groups(df_clean)
    print("\nTarget Groups Identified:")
    print(target_groups[['Cluster', 'Description', 'Size']])
    
    # Evaluate clustering
    evaluation = clustering.evaluate_clustering(df_clean)
    print(f"\nClustering Evaluation:")
    print(f"  Silhouette Score: {evaluation.get('silhouette_score', 'N/A'):.3f}")
    print(f"  Number of Clusters: {evaluation['n_clusters']}")
    
    # Step 5: Trend Analysis
    print("\n[STEP 5] TREND ANALYSIS")
    print("-" * 70)
    
    analyzer = TrendAnalyzer()
    
    # Yearly growth
    growth = analyzer.calculate_yearly_growth_rate(df_clean)
    print("\nYearly Growth Rates:")
    print(growth)
    
    # Seasonal patterns
    seasonal = analyzer.identify_seasonal_patterns(df_clean)
    print(f"\nSeasonal Patterns:")
    print(f"  Peak Month: {seasonal['peak_month']}")
    print(f"  Lowest Month: {seasonal['lowest_month']}")
    print(f"  Seasonal Distribution: {seasonal['seasonal_distribution']}")
    
    # Step 6: Predictive Modeling
    print("\n[STEP 6] PREDICTIVE MODELING")
    print("-" * 70)
    
    predictor = SpatialPredictor()
    print("Training hotspot intensity predictor...")
    print(f"Using model from config/settings.yaml: {predictor.configured_model}")
    
    metrics = predictor.train_configured_model(df_clean)
    print(f"\nModel Performance:")
    print(f"  R² Score: {metrics['test_r2']:.3f}")
    print(f"  RMSE: {metrics['test_rmse']:.2f}")
    
    # Predict next year hotspots (2025 for validation)
    next_year = 2025
    print(f"\nPredicting top hotspots for {next_year}...")
    predictions = predictor.predict_next_year_hotspots(df_clean, next_year, top_n=10)
    print("\nTop 10 Predicted Hotspots:")
    print(predictions[['Barangay District', 'Predicted_Cases', 'Prev_Year_Count']])
    
    # Feature importance
    feature_importance = predictor.get_feature_importance()
    print("\nFeature Importance:")
    print(feature_importance.head(5))
    
    # Step 7: Visualization
    print("\n[STEP 7] VISUALIZATION GENERATION")
    print("-" * 70)
    
    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate heatmap
    print("Generating interactive heatmap...")
    heatmap_gen = HeatmapGenerator()
    heatmap_gen.create_base_map()
    heatmap_gen.add_heatmap_layer(df_clean)
    heatmap_gen.add_cluster_markers(df_clean)
    
    heatmap_path = output_dir / "demo_heatmap.html"
    heatmap_gen.save_map(heatmap_path)
    
    # Generate charts
    print("Generating statistical charts...")
    chart_gen = ChartGenerator()
    
    chart_gen.create_gender_distribution(df_clean)
    chart_gen.create_age_breakdown(df_clean)
    chart_gen.create_yearly_trend(df_clean)
    chart_gen.create_monthly_pattern(df_clean)
    chart_gen.create_location_heatmap_chart(df_clean)
    
    if 'Cluster' in df_clean.columns:
        chart_gen.create_cluster_size_chart(cluster_stats)
    
    charts_dir = output_dir / "charts"
    chart_gen.save_all_charts(charts_dir)
    
    # Summary
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"\n✓ Processed {len(df_clean)} records")
    print(f"✓ Identified {clustering.n_clusters} clusters")
    print(f"✓ Generated predictions for {next_year}")
    print(f"✓ Created visualizations in: {output_dir}")
    print(f"\n📁 Output Files:")
    print(f"  - Heatmap: {heatmap_path}")
    print(f"  - Charts: {charts_dir}")
    print(f"  - Sample Data: {sample_path}")
    
    print("\n🚀 Next Steps:")
    print("  1. Replace sample data with your actual CSV dataset")
    print("  2. Run the FastAPI backend: uvicorn src.backend.main:app --reload")
    print("  3. Access API at: http://localhost:8000")
    print("  4. API docs at: http://localhost:8000/docs")
    print("  5. Open the generated heatmap in your browser")
    
    return df_clean


if __name__ == "__main__":
    try:
        df_result = demo_pipeline()
        print("\n✅ Demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
