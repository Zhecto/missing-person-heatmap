"""
Streamlit Web Application for Missing Person Hotspot Analysis
Alternative interface using Streamlit for rapid prototyping and academic presentation.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add project paths
sys.path.append(str(Path(__file__).parent / "src"))

from core.ingestion.data_loader import DataLoader
from core.preprocessing.data_cleaner import DataCleaner
from core.analysis.clustering import ClusteringModel
from core.analysis.predictor import SpatialPredictor, TrendAnalyzer
from core.visualization.visualizer import HeatmapGenerator, ChartGenerator
import yaml

# Load configuration
config_path = Path(__file__).parent / 'config' / 'settings.yaml'
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        CLUSTERING_MODEL = config.get('modeling', {}).get('clustering_method', 'kmeans')
        PREDICTION_MODEL = config.get('modeling', {}).get('prediction_model', 'poisson')
else:
    CLUSTERING_MODEL = "kmeans"
    PREDICTION_MODEL = "poisson"

# CLUSTERING MODEL PARAMETERS
if CLUSTERING_MODEL == "kmeans":
    CLUSTERING_PARAMS = {'n_clusters': 3}
else:  # dbscan
    CLUSTERING_PARAMS = {'eps': 0.1, 'min_samples': 3}




# Page configuration
st.set_page_config(
    page_title="Missing Person Hotspot Analysis - Metro Manila (NCR)",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'geocoding_done' not in st.session_state:
    st.session_state.geocoding_done = False
if 'clustering_done' not in st.session_state:
    st.session_state.clustering_done = False
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_geocoded' not in st.session_state:
    st.session_state.df_geocoded = None
if 'cluster_results' not in st.session_state:
    st.session_state.cluster_results = None
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None


# Header
st.markdown('<div class="main-header">🗺️ Missing Person Hotspot Analysis System</div>', 
            unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Metro Manila (National Capital Region)</p>',
            unsafe_allow_html=True)
st.markdown("---")


# Sidebar
with st.sidebar:
    st.header("📋 Pipeline Status")
    
    # Status indicators
    status_data_loaded = "✅" if st.session_state.data_loaded else "⏳"
    status_geocoding = "✅" if st.session_state.geocoding_done else "⏳"
    status_clustering = "✅" if st.session_state.clustering_done else "⏳"
    status_prediction = "✅" if st.session_state.prediction_done else "⏳"
    
    st.markdown(f"""
    - {status_data_loaded} **Data Loaded**
    - {status_geocoding} **Geocoding Complete**
    - {status_clustering} **Clustering Complete**
    - {status_prediction} **Prediction Complete**
    """)
    
    st.markdown("---")
    
    # Navigation
    st.header("🧭 Navigation")
    page = st.radio(
        "Select Module:",
        ["📤 Data Upload", "🗺️ Geocoding", "📊 Clustering Results", 
         "🔮 Prediction Results", "📈 Visualizations"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.caption("Built with Streamlit | Data Mining Project")


# =============================================================================
# PAGE 1: Data Upload
# =============================================================================
if page == "📤 Data Upload":
    st.header("📤 Data Upload & Loading")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file with missing person data",
            type=['csv'],
            help="File should contain: Date, Location, Coordinates, Demographics"
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("Loading data..."):
                    # Save uploaded file temporarily
                    temp_path = Path("data/uploaded_data.csv")
                    temp_path.parent.mkdir(exist_ok=True)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load data
                    loader = DataLoader(str(temp_path))
                    st.session_state.df_raw = loader.load_csv()
                    st.session_state.data_loaded = True
                    
                    st.success(f"✅ Data loaded successfully! {len(st.session_state.df_raw)} records")
                    
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
    
    with col2:
        st.subheader("Or Use Demo Data")
        if st.button("🎲 Load Demo Data", use_container_width=True):
            demo_path = Path("data/sample_data.csv")
            if demo_path.exists():
                try:
                    loader = DataLoader(str(demo_path))
                    st.session_state.df_raw = loader.load_csv()
                    st.session_state.data_loaded = True
                    st.success("✅ Demo data loaded!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Run `python demo.py` first to generate demo data")
    
    # Display loaded data
    if st.session_state.data_loaded:
        st.markdown("---")
        st.subheader("📊 Data Preview")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        df = st.session_state.df_raw
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            if 'Barangay District' in df.columns:
                st.metric("Unique Locations", df['Barangay District'].nunique())
        with col4:
            if 'Date Reported Missing' in df.columns:
                date_col = pd.to_datetime(df['Date Reported Missing'], errors='coerce')
                years = date_col.dt.year.nunique()
                st.metric("Years Covered", years)
        
        # Data table
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data info
        with st.expander("📋 Column Information"):
            st.dataframe(pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Missing': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
            }), use_container_width=True)


# =============================================================================
# PAGE 2: Geocoding
# =============================================================================
elif page == "🗺️ Geocoding":
    st.header("🗺️ Geocoding Missing Coordinates")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first in the Data Upload page")
    else:
        df = st.session_state.df_raw.copy()
        
        # Check for missing coordinates
        missing_lat = df['Latitude'].isnull().sum()
        missing_lon = df['Longitude'].isnull().sum()
        missing_both = df[df['Latitude'].isnull() | df['Longitude'].isnull()].shape[0]
        
        st.info(f"Records with missing coordinates: **{missing_both}** out of {len(df)}")
        
        if missing_both == 0:
            st.success("✅ All records have complete coordinates. No geocoding needed.")
            st.session_state.df_geocoded = df
            st.session_state.geocoding_done = True
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Missing Latitude", missing_lat)
            with col2:
                st.metric("Missing Longitude", missing_lon)
            with col3:
                st.metric("Total Missing", missing_both)
            
            # Preview records with missing coordinates
            st.markdown("### Records Requiring Geocoding")
            missing_df = df[df['Latitude'].isnull() | df['Longitude'].isnull()]
            st.dataframe(missing_df[['Barangay District', 'Latitude', 'Longitude']].head(10), use_container_width=True)
            
            st.markdown("---")
            
            if st.button("🌍 Apply Geocoding", type="primary", use_container_width=True):
                with st.spinner("Geocoding missing coordinates using OpenStreetMap..."):
                    try:
                        from geopy.geocoders import Nominatim
                        from geopy.exc import GeocoderTimedOut, GeocoderServiceError
                        import time
                        
                        geolocator = Nominatim(user_agent="missing_person_heatmap")
                        
                        success_count = 0
                        fail_count = 0
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, row in df.iterrows():
                            if pd.isnull(row['Latitude']) or pd.isnull(row['Longitude']):
                                try:
                                    # Add Metro Manila context to improve accuracy
                                    location_query = f"{row['Barangay District']}, Metro Manila, Philippines"
                                    location = geolocator.geocode(location_query, timeout=10)
                                    
                                    if location:
                                        df.at[idx, 'Latitude'] = location.latitude
                                        df.at[idx, 'Longitude'] = location.longitude
                                        success_count += 1
                                    else:
                                        fail_count += 1
                                    
                                    time.sleep(1)  # Respect API rate limits
                                    
                                except (GeocoderTimedOut, GeocoderServiceError):
                                    fail_count += 1
                                
                                # Update progress
                                progress = (success_count + fail_count) / missing_both
                                progress_bar.progress(progress)
                                status_text.text(f"Geocoded: {success_count} | Failed: {fail_count}")
                        
                        st.session_state.df_geocoded = df
                        st.session_state.geocoding_done = True
                        
                        st.success(f"✅ Geocoding complete! Successfully geocoded {success_count} out of {missing_both} records")
                        
                        if fail_count > 0:
                            st.warning(f"⚠️ {fail_count} records could not be geocoded. They will be excluded from analysis.")
                        
                    except ImportError:
                        st.error("❌ Geopy library not found. Please install it: `pip install geopy`")
                    except Exception as e:
                        st.error(f"❌ Geocoding failed: {str(e)}")
        
        # Show geocoded data if available
        if st.session_state.geocoding_done:
            st.markdown("---")
            st.subheader("📊 Geocoded Data")
            
            final_df = st.session_state.df_geocoded
            complete_coords = final_df.dropna(subset=['Latitude', 'Longitude'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", len(final_df))
            with col2:
                st.metric("Records with Valid Coordinates", len(complete_coords))
            
            # Preview
            st.dataframe(complete_coords[['Barangay District', 'Latitude', 'Longitude']].head(10), use_container_width=True)
            
            # Download option
            csv = complete_coords.to_csv(index=False).encode('utf-8')
            st.download_button(
                "💾 Download Geocoded Data",
                csv,
                "geocoded_data.csv",
                "text/csv",
                use_container_width=True
            )


# =============================================================================
# PAGE 3: Clustering Results
# =============================================================================
elif page == "📊 Clustering Results":
    st.header("📊 Spatial Clustering Analysis")
    
    if not st.session_state.geocoding_done:
        st.warning("⚠️ Please complete data upload and geocoding first")
    else:
        df = st.session_state.df_geocoded.dropna(subset=['Latitude', 'Longitude']).copy()
        
        st.info(f"Analyzing {len(df)} records using {CLUSTERING_MODEL.upper()}: {CLUSTERING_PARAMS}")
        
        # Auto-run clustering
        if not st.session_state.clustering_done:
            with st.spinner(f"Running {CLUSTERING_MODEL.upper()} clustering..."):
                try:
                    cluster_model = ClusteringModel()
                    
                    if CLUSTERING_MODEL == "kmeans":
                        cluster_model.fit_kmeans(df, n_clusters=CLUSTERING_PARAMS['n_clusters'])
                    elif CLUSTERING_MODEL == "dbscan":
                        cluster_model.fit_dbscan(df, eps=CLUSTERING_PARAMS['eps'], 
                                                min_samples=CLUSTERING_PARAMS['min_samples'])
                    
                    # Get results
                    metrics = cluster_model.evaluate_clustering(df)
                    results = {
                        'model': cluster_model,
                        'labels': cluster_model.labels_,
                        'centers': cluster_model.cluster_centers_,
                        'n_clusters': cluster_model.n_clusters,
                        'silhouette_score': metrics.get('silhouette_score'),
                        'davies_bouldin_score': metrics.get('davies_bouldin_score'),
                        'inertia': metrics.get('inertia') if CLUSTERING_MODEL == "kmeans" else None
                    }
                    
                    st.session_state.cluster_results = results
                    df['cluster'] = results['labels']
                    st.session_state.df_geocoded = df
                    st.session_state.clustering_done = True
                    
                    st.success(f"✅ Clustering complete! Identified {results['n_clusters']} clusters")
                    
                except Exception as e:
                    st.error(f"❌ Clustering failed: {str(e)}")
                    st.exception(e)
                    st.stop()
        
        # Display results
        if st.session_state.clustering_done:
            results = st.session_state.cluster_results
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Clusters", results['n_clusters'])
            with col2:
                st.metric("Silhouette Score", f"{results['silhouette_score']:.3f}")
            with col3:
                st.metric("Total Records", len(df))
            
            st.markdown("---")
            
            # Cluster statistics
            st.subheader("📊 Cluster Statistics")
            
            cluster_stats = df.groupby('cluster').agg({
                'Latitude': 'count',
                'Barangay District': lambda x: x.nunique()
            }).rename(columns={'Latitude': 'Records', 'Barangay District': 'Barangays'})
            
            cluster_stats = cluster_stats.reset_index()
            cluster_stats['Cluster'] = 'Cluster ' + cluster_stats['cluster'].astype(str)
            cluster_stats = cluster_stats[['Cluster', 'Records', 'Barangays']]
            
            st.dataframe(cluster_stats, use_container_width=True)
            
            # Cluster size distribution chart
            st.markdown("---")
            st.subheader("📈 Cluster Size Distribution")
            
            fig = px.bar(
                cluster_stats,
                x='Cluster',
                y='Records',
                title="Number of Records per Cluster",
                color='Records',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Geographic cluster map
            st.markdown("---")
            st.subheader("🗺️ Geographic Cluster Distribution")
            
            fig = px.scatter_mapbox(
                df,
                lat='Latitude',
                lon='Longitude',
                color='cluster',
                hover_data=['Barangay District'],
                title="Missing Person Incidents by Cluster",
                mapbox_style="carto-positron",
                zoom=10,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top locations per cluster
            st.markdown("---")
            st.subheader("📍 Top Locations by Cluster")
            
            for cluster_id in sorted(df['cluster'].unique()):
                with st.expander(f"Cluster {cluster_id}"):
                    cluster_data = df[df['cluster'] == cluster_id]
                    top_locations = cluster_data['Barangay District'].value_counts().head(5)
                    st.write(f"**Total Records:** {len(cluster_data)}")
                    st.write("**Top 5 Barangays:**")
                    for location, count in top_locations.items():
                        st.write(f"- {location}: {count} incidents")


# =============================================================================
# PAGE 4: Prediction Results
# =============================================================================
elif page == "🔮 Prediction Results":
    st.header("🔮 Hotspot Prediction for 2025 (with Validation)")
    
    if not st.session_state.geocoding_done:
        st.warning("⚠️ Please complete data upload and geocoding first")
    else:
        df = st.session_state.df_geocoded.dropna(subset=['Latitude', 'Longitude']).copy()
        
        model_display = PREDICTION_MODEL.replace('_', ' ').title()
        st.info(f"""
        **Training Strategy:** Train on 2019-2024 data → Predict 2025 → Validate against actual 2025 data
        
        Using {model_display} model on {len(df)} records
        """)
        
        # Auto-run prediction
        if not st.session_state.prediction_done:
            with st.spinner(f"Training {model_display} model and generating predictions..."):
                try:
                    predictor = SpatialPredictor()
                    
                    # Train configured model
                    metrics = predictor.train_configured_model(df, model_name=PREDICTION_MODEL)
                    
                    # Generate predictions for 2025
                    predictions = predictor.predict_next_year_hotspots(
                        df,
                        next_year=2025,
                        top_n=10
                    )
                    
                    # Get feature importance
                    feature_importance = predictor.get_feature_importance()
                    
                    st.session_state.prediction_results = {
                        'metrics': metrics,
                        'predictions': predictions,
                        'feature_importance': feature_importance,
                        'model_name': PREDICTION_MODEL
                    }
                    st.session_state.predictor = predictor  # Store predictor for heatmap generation
                    st.session_state.prediction_done = True
                    
                    st.success(f"✅ Prediction complete! {model_display} model trained successfully")
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.exception(e)
                    st.stop()
        
        # Display results
        if st.session_state.prediction_done:
            results = st.session_state.prediction_results
            model_display = results.get('model_name', PREDICTION_MODEL).replace('_', ' ').title()
            
            # Model performance metrics
            st.subheader(f"📊 {model_display} Model Performance")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{results['metrics']['test_r2']:.3f}")
            with col2:
                st.metric("RMSE", f"{results['metrics']['test_rmse']:.3f}")
            with col3:
                st.metric("Overfit Gap", f"{results['metrics'].get('overfit_gap', 0):.3f}")
            
            with st.expander("ℹ️ About these metrics"):
                st.markdown("""
                - **R² Score**: Proportion of variance explained (higher is better, max 1.0)
                - **RMSE**: Root Mean Square Error (lower is better)
                - **Overfit Gap**: Difference between train and test R² (lower is better)
                """)
            
            st.markdown("---")
            
            # Top predicted hotspots
            st.subheader("🎯 Top 10 Predicted Hotspots for 2025")
            
            pred_df = results['predictions'].copy()
            pred_df.index = range(1, len(pred_df) + 1)  # Start ranking from 1
            
            st.dataframe(pred_df, use_container_width=True)
            
            # Full predictions table for all barangays
            st.markdown("---")
            st.subheader("📋 Complete Predictions for All Barangays (2025)")
            
            # Get all predictions from the predictor
            predictor = st.session_state.predictor
            all_predictions = predictor.get_all_predictions()
            
            # Sort by predicted cases (descending)
            all_predictions_sorted = all_predictions.sort_values('Predicted_Cases', ascending=False).copy()
            
            # Add ranking
            all_predictions_sorted.insert(0, 'Rank', range(1, len(all_predictions_sorted) + 1))
            
            # Select relevant columns for display
            display_cols = ['Rank', 'Barangay District', 'Predicted_Cases', 'Prev_Year_Count', 'Latitude', 'Longitude']
            all_predictions_display = all_predictions_sorted[display_cols].copy()
            
            # Round predicted cases to 2 decimals
            all_predictions_display['Predicted_Cases'] = all_predictions_display['Predicted_Cases'].round(2)
            all_predictions_display['Latitude'] = all_predictions_display['Latitude'].round(6)
            all_predictions_display['Longitude'] = all_predictions_display['Longitude'].round(6)
            
            st.dataframe(all_predictions_display, use_container_width=True, hide_index=True)
            
            st.info(f"📊 Total Barangays Analyzed: {len(all_predictions_display)}")
            
            # Model feature importance (interpretability)
            st.markdown("---")
            st.subheader("🔍 Feature Importance")
            
            if 'feature_importance' in results:
                feat_df = results['feature_importance']
                st.dataframe(feat_df, use_container_width=True, hide_index=True)
                
                with st.expander("ℹ️ How to interpret feature importance"):
                    st.markdown(f"""
                    **{model_display} Feature Importance:**
                
                - **Higher values**: These features have stronger influence on predictions
                - **Feature**: The input variable used by the model
                - **Importance**: Relative contribution to prediction accuracy
                
                Features show which factors most strongly affect the predicted case counts.
                """)
            
            # Generate prediction heatmap
            st.markdown("---")
            st.subheader("🗺️ Prediction vs Actual Comparison")
            
            if st.button("🗺️ Generate Comparison Heatmaps", use_container_width=True, type="primary"):
                with st.spinner("Generating prediction and actual data heatmaps..."):
                    try:
                        # Use the already trained predictor from session state
                        predictor = st.session_state.predictor
                        
                        # Ensure Year column exists in df
                        df_copy = df.copy()
                        if 'Year' not in df_copy.columns and 'Date Reported Missing' in df_copy.columns:
                            df_copy['Date Reported Missing'] = pd.to_datetime(df_copy['Date Reported Missing'], errors='coerce')
                            df_copy['Year'] = df_copy['Date Reported Missing'].dt.year
                        
                        # Generate prediction heatmap
                        prediction_path = predictor.generate_prediction_heatmap(2025)
                        
                        # Generate actual data heatmap
                        actual_path = predictor.generate_actual_heatmap(df_copy, 2025)
                        
                        st.success(f"✅ Both heatmaps generated successfully!")
                        
                        # Display both heatmaps side by side
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Predicted Hotspots for 2025**")
                            with open(prediction_path, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            st.components.v1.html(html_content, height=600, scrolling=True)
                        
                        with col2:
                            st.markdown("**Actual 2025 Data**")
                            with open(actual_path, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            st.components.v1.html(html_content, height=600, scrolling=True)
                        
                    except Exception as e:
                        st.error(f"❌ Failed to generate heatmaps: {str(e)}")
                        st.exception(e)
            
            # Download predictions button below
            st.markdown("---")
            csv = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "💾 Download Predictions",
                csv,
                "hotspot_predictions_2025.csv",
                "text/csv",
                use_container_width=True
            )


# =============================================================================
# PAGE 5: Visualizations
# =============================================================================
elif page == "📈 Visualizations":
    st.header("📈 Interactive Visualizations")
    
    if not st.session_state.geocoding_done:
        st.warning("⚠️ Please complete data upload and geocoding first")
    else:
        df = st.session_state.df_geocoded.dropna(subset=['Latitude', 'Longitude']).copy()
        
        tab1, tab2 = st.tabs(["🗺️ Heatmap", "📊 Charts"])
        
        with tab1:
            st.subheader("Geographic Heatmap")
            
            include_clusters = st.checkbox(
                "Include Cluster Boundaries",
                value=st.session_state.clustering_done
            )
            
            if st.button("🗺️ Generate Heatmap", use_container_width=True):
                with st.spinner("Generating heatmap..."):
                    try:
                        heatmap_gen = HeatmapGenerator()
                        output_path = heatmap_gen.create_heatmap(
                            df,
                            include_clusters=include_clusters and st.session_state.clustering_done
                        )
                        
                        st.success(f"✅ Heatmap saved to: {output_path}")
                        
                        # Display the heatmap
                        with open(output_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        
                        st.components.v1.html(html_content, height=600, scrolling=True)
                        
                    except Exception as e:
                        st.error(f"❌ Failed to generate heatmap: {str(e)}")
        
        with tab2:
            st.subheader("Statistical Charts")
            
            if st.button("📊 Generate All Charts", use_container_width=True):
                with st.spinner("Generating charts..."):
                    try:
                        chart_gen = ChartGenerator()
                        output_dir = chart_gen.generate_all_charts(df)
                        
                        st.success(f"✅ Charts saved to: {output_dir}")
                        
                        # Display sample charts
                        st.markdown("### Sample Charts")
                        chart_files = list(Path(output_dir).glob("*.html"))
                        
                        for chart_file in chart_files[:3]:  # Show first 3 charts
                            st.markdown(f"**{chart_file.stem.replace('_', ' ').title()}**")
                            with open(chart_file, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            st.components.v1.html(html_content, height=500, scrolling=True)
                        
                    except Exception as e:
                        st.error(f"❌ Failed to generate charts: {str(e)}")


# Footer
st.markdown("---")
st.caption("Missing Person Hotspot Analysis System | Data Mining Project 2025")
