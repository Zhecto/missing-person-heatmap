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
if 'data_cleaned' not in st.session_state:
    st.session_state.data_cleaned = False
if 'clustering_done' not in st.session_state:
    st.session_state.clustering_done = False
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'cluster_results' not in st.session_state:
    st.session_state.cluster_results = None


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
    status_data_cleaned = "✅" if st.session_state.data_cleaned else "⏳"
    status_clustering = "✅" if st.session_state.clustering_done else "⏳"
    
    st.markdown(f"""
    - {status_data_loaded} **Data Loaded**
    - {status_data_cleaned} **Data Cleaned**
    - {status_clustering} **Clustering Complete**
    """)
    
    st.markdown("---")
    
    # Navigation
    st.header("🧭 Navigation")
    page = st.radio(
        "Select Module:",
        ["📤 Data Upload", "🧹 Preprocessing", "📊 Clustering", 
         "🔮 Prediction", "🗺️ Visualization"],
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
                    st.session_state.df_raw = loader.load()
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
                    st.session_state.df_raw = loader.load()
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
# PAGE 2: Preprocessing
# =============================================================================
elif page == "🧹 Preprocessing":
    st.header("🧹 Data Preprocessing")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first in the Data Upload page")
    else:
        st.info("Clean and transform the raw data for analysis")
        
        if st.button("🚀 Run Data Cleaning", type="primary", use_container_width=True):
            with st.spinner("Cleaning data..."):
                try:
                    cleaner = DataCleaner()
                    st.session_state.df_cleaned = cleaner.clean(st.session_state.df_raw)
                    st.session_state.data_cleaned = True
                    
                    st.success("✅ Data cleaned successfully!")
                    
                    # Show before/after comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Records Before", len(st.session_state.df_raw))
                    with col2:
                        st.metric("Records After", len(st.session_state.df_cleaned))
                    
                except Exception as e:
                    st.error(f"❌ Cleaning failed: {str(e)}")
        
        if st.session_state.data_cleaned:
            st.markdown("---")
            st.subheader("📊 Cleaned Data Summary")
            
            df = st.session_state.df_cleaned
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Temporal Coverage**")
                if 'Year' in df.columns:
                    st.write(f"Years: {df['Year'].min()} - {df['Year'].max()}")
                    st.write(f"Records per year: {len(df) / df['Year'].nunique():.1f}")
            
            with col2:
                st.markdown("**Geographic Coverage**")
                if 'Barangay District' in df.columns:
                    st.write(f"Locations: {df['Barangay District'].nunique()}")
                    top_location = df['Barangay District'].value_counts().index[0]
                    st.write(f"Top location: {top_location}")
            
            with col3:
                st.markdown("**Demographics**")
                if 'Age' in df.columns:
                    st.write(f"Avg Age: {df['Age'].mean():.1f} years")
                if 'Gender' in df.columns:
                    gender_dist = df['Gender'].value_counts()
                    st.write(f"Gender: {gender_dist.index[0]} ({gender_dist.values[0]})")
            
            # Preview cleaned data
            st.dataframe(df.head(10), use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "💾 Download Cleaned Data",
                csv,
                "cleaned_data.csv",
                "text/csv",
                use_container_width=True
            )


# =============================================================================
# PAGE 3: Clustering Analysis
# =============================================================================
elif page == "📊 Clustering":
    st.header("📊 Spatial Clustering Analysis")
    
    if not st.session_state.data_cleaned:
        st.warning("⚠️ Please clean the data first in the Preprocessing page")
    else:
        st.info("Identify geographic concentrations of missing person incidents")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            algorithm = st.selectbox(
                "Clustering Algorithm",
                ["K-Means", "DBSCAN"],
                help="K-Means: partitions data into k clusters | DBSCAN: density-based clustering"
            )
        
        with col2:
            if algorithm == "K-Means":
                n_clusters = st.slider("Number of Clusters (k)", 3, 10, 5)
            else:
                eps = st.slider("Epsilon (distance)", 0.005, 0.05, 0.01, 0.005)
                min_samples = st.slider("Min Samples", 3, 10, 5)
        
        if st.button("🔍 Run Clustering", type="primary", use_container_width=True):
            with st.spinner("Running clustering analysis..."):
                try:
                    cluster_model = ClusteringModel()
                    
                    if algorithm == "K-Means":
                        results = cluster_model.run_kmeans(
                            st.session_state.df_cleaned,
                            n_clusters=n_clusters
                        )
                    else:
                        results = cluster_model.run_dbscan(
                            st.session_state.df_cleaned,
                            eps=eps,
                            min_samples=min_samples
                        )
                    
                    st.session_state.cluster_results = results
                    st.session_state.df_cleaned['cluster'] = results['labels']
                    st.session_state.clustering_done = True
                    
                    st.success(f"✅ Clustering complete! Found {results['n_clusters']} clusters")
                    
                except Exception as e:
                    st.error(f"❌ Clustering failed: {str(e)}")
        
        # Display results
        if st.session_state.clustering_done:
            st.markdown("---")
            st.subheader("📈 Clustering Results")
            
            results = st.session_state.cluster_results
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Clusters", results['n_clusters'])
            with col2:
                st.metric("Silhouette Score", f"{results['silhouette_score']:.3f}")
            with col3:
                noise_points = (results['labels'] == -1).sum() if -1 in results['labels'] else 0
                st.metric("Noise Points", noise_points)
            
            # Cluster distribution chart
            cluster_counts = pd.Series(results['labels']).value_counts().sort_index()
            fig = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                labels={'x': 'Cluster ID', 'y': 'Number of Points'},
                title="Cluster Size Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot
            df = st.session_state.df_cleaned
            fig = px.scatter_mapbox(
                df,
                lat='Latitude',
                lon='Longitude',
                color='cluster',
                hover_data=['Barangay District'],
                title="Geographic Cluster Distribution",
                mapbox_style="open-street-map",
                zoom=10
            )
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE 4: Prediction
# =============================================================================
elif page == "🔮 Prediction":
    st.header("🔮 Hotspot Prediction")
    
    if not st.session_state.data_cleaned:
        st.warning("⚠️ Please clean the data first")
    else:
        st.info("Predict future missing person hotspots using machine learning")
        
        # Model selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_type = st.selectbox(
                "Prediction Model",
                ["Gradient Boosting", "Poisson Regression", "Compare Both"],
                help="Gradient Boosting: High accuracy | Poisson: Interpretable coefficients"
            )
        
        with col2:
            target_year = st.number_input(
                "Target Year",
                min_value=2026,
                max_value=2030,
                value=2026
            )
        
        with col3:
            top_n = st.slider("Top N Hotspots", 5, 20, 10)
        
        if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
            with st.spinner("Training model and generating predictions..."):
                try:
                    predictor = SpatialPredictor()
                    
                    if model_type == "Compare Both":
                        # Train and compare both models
                        st.subheader("📊 Model Comparison")
                        comparison = predictor.compare_models(st.session_state.df_cleaned)
                        st.dataframe(comparison, use_container_width=True)
                        
                        # Get predictions from both
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🌲 Gradient Boosting Predictions")
                            pred_gb = predictor.predict_next_year_hotspots(
                                st.session_state.df_cleaned,
                                target_year,
                                top_n
                            )
                            st.dataframe(pred_gb, use_container_width=True)
                        
                        with col2:
                            st.subheader("📈 Poisson Regression Predictions")
                            pred_poisson = predictor.predict_next_year_hotspots_poisson(
                                st.session_state.df_cleaned,
                                target_year,
                                top_n
                            )
                            st.dataframe(pred_poisson, use_container_width=True)
                        
                        # Poisson coefficients
                        st.markdown("---")
                        st.subheader("📉 Poisson Regression Coefficients")
                        coefficients = predictor.get_poisson_coefficients()
                        st.dataframe(coefficients, use_container_width=True)
                        
                    elif model_type == "Poisson Regression":
                        # Train Poisson model
                        metrics = predictor.train_poisson_regressor(st.session_state.df_cleaned)
                        predictions = predictor.predict_next_year_hotspots_poisson(
                            st.session_state.df_cleaned,
                            target_year,
                            top_n
                        )
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Score", f"{metrics['test_r2']:.3f}")
                        with col2:
                            st.metric("RMSE", f"{metrics['test_rmse']:.3f}")
                        with col3:
                            st.metric("AIC", f"{metrics['aic']:.2f}")
                        
                        # Predictions
                        st.subheader(f"🎯 Top {top_n} Predicted Hotspots for {target_year}")
                        st.dataframe(predictions, use_container_width=True)
                        
                        # Coefficients
                        st.markdown("---")
                        st.subheader("📊 Model Coefficients (Interpretable)")
                        coefficients = predictor.get_poisson_coefficients()
                        st.dataframe(coefficients, use_container_width=True)
                        
                        with st.expander("ℹ️ How to interpret coefficients"):
                            st.markdown("""
                            - **Rate Ratio > 1**: Factor increases incident rate
                            - **Rate Ratio < 1**: Factor decreases incident rate
                            - **P-Value < 0.05**: Statistically significant effect
                            - Coefficients show multiplicative effect on incident rate
                            """)
                    
                    else:  # Gradient Boosting
                        # Train GB model
                        metrics = predictor.train_hotspot_intensity_predictor(st.session_state.df_cleaned)
                        predictions = predictor.predict_next_year_hotspots(
                            st.session_state.df_cleaned,
                            target_year,
                            top_n
                        )
                        
                        # Display metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("R² Score", f"{metrics['test_r2']:.3f}")
                        with col2:
                            st.metric("RMSE", f"{metrics['test_rmse']:.3f}")
                        
                        # Predictions
                        st.subheader(f"🎯 Top {top_n} Predicted Hotspots for {target_year}")
                        st.dataframe(predictions, use_container_width=True)
                        
                        # Feature importance
                        st.markdown("---")
                        st.subheader("📊 Feature Importance")
                        importance = predictor.get_feature_importance()
                        fig = px.bar(
                            importance,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title="Which features matter most?"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.success("✅ Prediction complete!")
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.exception(e)


# =============================================================================
# PAGE 5: Visualization
# =============================================================================
elif page == "🗺️ Visualization":
    st.header("🗺️ Interactive Visualizations")
    
    if not st.session_state.data_cleaned:
        st.warning("⚠️ Please clean the data first")
    else:
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
                            st.session_state.df_cleaned,
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
                        output_dir = chart_gen.generate_all_charts(st.session_state.df_cleaned)
                        
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
