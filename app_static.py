import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Manila City Missing Persons",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Navigation
page = st.sidebar.radio("Navigation", ["📊 Current Cases", "🔮 Predictions"])

@st.cache_data
def load_data():
    data_path = Path(__file__).parent / "notebook" / "Missing People - cleaned.csv"
    return pd.read_csv(data_path)

@st.cache_data
def load_predictions():
    predictions_path = Path(__file__).parent / "notebook" / "outputs" / "2026_predictions.csv"
    return pd.read_csv(predictions_path)

@st.cache_data
def load_2025_predictions():
    predictions_path = Path(__file__).parent / "data" / "outputs" / "predictions_2025_by_district.csv"
    return pd.read_csv(predictions_path)

# Page routing
if page == "📊 Current Cases":
    st.title("🔍 Manila City Missing Person Cases")
    
    # Load data
    try:
        df = load_data()
        
        # Filter valid coordinates
        df_map = df.dropna(subset=['Latitude', 'Longitude']).copy()
        
        # Year filter
        years = sorted(df_map['Year'].dropna().unique())
        year_options = ['All'] + [int(y) for y in years]
        selected_year = st.selectbox("Filter by Year", year_options, index=0)
        
        if selected_year != 'All':
            df_map = df_map[df_map['Year'] == selected_year]
        
        # Toggle for district labels
        show_labels = st.toggle("Show District Labels", value=False)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cases", len(df))
        with col2:
            st.metric("With Coordinates", len(df_map))
        with col3:
            years_count = df['Year'].dropna().nunique()
            st.metric("Years", years_count)
        with col4:
            districts = df['District_Cleaned'].dropna().nunique()
            st.metric("Districts", districts)
        
        # Create map centered on Manila City
        center_lat = df_map['Latitude'].mean()
        center_lon = df_map['Longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='cartodbpositron'
        )
        
        # Prepare heatmap data
        heat_data = [[row['Latitude'], row['Longitude']] for idx, row in df_map.iterrows()]
        
        # Add heatmap layer with larger spread
        HeatMap(
            heat_data,
            min_opacity=0.3,
            max_opacity=0.9,
            radius=35,
            blur=35,
            gradient={0.4: 'blue', 0.5: 'lime', 0.65: 'yellow', 0.8: 'orange', 1.0: 'red'}
        ).add_to(m)
        
        # Add district labels if toggle is on
        if show_labels:
            district_centers = df_map.groupby('District_Cleaned').agg({
                'Latitude': 'mean',
                'Longitude': 'mean',
                'Person_ID': 'count'
            }).reset_index()
            district_centers.columns = ['District', 'Latitude', 'Longitude', 'Case_Count']
            
            for idx, row in district_centers.iterrows():
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    icon=folium.DivIcon(html=f"""
                        <div style="
                            font-size: 11px;
                            font-weight: bold;
                            color: white;
                            text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;
                            white-space: nowrap;
                        ">
                            {row['District']}<br>
                            <span style="font-size: 10px;">({row['Case_Count']} cases)</span>
                        </div>
                    """)
                ).add_to(m)
        
        # Display map
        st.components.v1.html(m._repr_html_(), height=600)
        
        st.success(f"✅ Showing heatmap for {len(df_map)} cases")
        
        # Show data preview
        with st.expander("View Data"):
            st.dataframe(df_map[['Person_ID', 'AGE', 'GENDER', 'District_Cleaned', 'Date Reported Missing', 'Latitude', 'Longitude']])
        
        # Show analysis outputs
        st.subheader("📊 Analysis Outputs")
        outputs_path = Path(__file__).parent / "notebook" / "outputs"
        
        # Display PNG images
        png_files = [
            ("Age Group Distribution", "age_group_distribution.png"),
            ("Hourly Missing Pattern", "hourly_pattern.png"),
            ("Location Completeness", "location_completeness_pie.png"),
            ("Metro Manila City Counts", "metro_manila_city_counts.png"),
            ("Missing Values Overview", "missing_values_bar.png"),
            ("Monthly Timeline", "monthly_timeline.png"),
            ("Seasonality Pattern", "seasonality_polar_plot.png"),
            ("Top Districts", "top_districts_bar.png"),
            ("DBSCAN Clusters Scatter", "dbscan_clusters_scatter.png")
        ]
        
        for title, filename in png_files:
            filepath = outputs_path / filename
            if filepath.exists():
                with st.expander(title):
                    st.image(str(filepath), use_container_width=True)
        
        # Display HTML maps
        html_files = [
            ("DBSCAN Clusters Map", "dbscan_clusters_map.html")
        ]
        
        for title, filename in html_files:
            filepath = outputs_path / filename
            if filepath.exists():
                with st.expander(title):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=600, scrolling=True)
    
    except Exception as e:
        st.error(f"Error loading data: {e}")

elif page == "🔮 Predictions":
    st.title("🔮 Missing Person Predictions - Manila City")
    
    # Prediction year selector
    prediction_year = st.radio("Select Prediction Year", ["2025", "2026"], horizontal=True)
    
    try:
        if prediction_year == "2025":
            df_pred_raw = load_2025_predictions()
            # Standardize column names to match 2026 format
            df_pred = df_pred_raw.rename(columns={
                'District_Cleaned': 'Barangay District',
                'Actual_Cases': 'Actual_Cases_2025',
                'Predicted_Cases_GB': 'Predicted_Cases'
            })
            # Keep only necessary columns and add Prev_Year_Count (2024)
            if 'Prev_Year_Count' not in df_pred.columns:
                # Calculate prev year count from the training data for each district
                df_full = load_data()
                prev_counts = df_full[df_full['Year'] == 2024].groupby('District_Cleaned')['Person_ID'].count().to_dict()
                df_pred['Prev_Year_Count'] = df_pred['Barangay District'].map(prev_counts).fillna(0)
        else:  # 2026
            df_pred = load_predictions()
        
        # For 2025, add option to view predicted vs actual
        if prediction_year == "2025":
            map_view = st.radio("Map View", ["Predicted", "Actual"], horizontal=True, key="map_view_2025")
        
        # Toggle for district labels
        show_pred_labels = st.toggle("Show District Labels", value=False, key="pred_labels")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_predicted = df_pred['Predicted_Cases'].sum()
            st.metric(f"Total Predicted Cases ({prediction_year})", f"{total_predicted:.0f}")
        with col2:
            if prediction_year == "2025":
                total_actual = df_pred.get('Actual_Cases_2025', pd.Series([0])).sum()
                st.metric("Actual 2025 Cases", f"{total_actual:.0f}")
            else:
                total_prev_year = df_pred['Prev_Year_Count'].sum()
                st.metric("2025 Cases", f"{total_prev_year:.0f}")
        with col3:
            if prediction_year == "2025":
                accuracy = (1 - abs(total_predicted - total_actual) / total_actual * 100) if total_actual > 0 else 0
                st.metric("Prediction Accuracy", f"{accuracy:.1f}%")
            else:
                total_prev_year = df_pred['Prev_Year_Count'].sum()
                change = total_predicted - total_prev_year
                change_pct = (change / total_prev_year * 100) if total_prev_year > 0 else 0
                st.metric("Expected Change", f"{change:+.0f}", f"{change_pct:+.1f}%")
        with col4:
            top_district = df_pred.loc[df_pred['Predicted_Cases'].idxmax(), 'Barangay District']
            st.metric("Highest Risk District", top_district)
        
        # Create map centered on Manila City
        center_lat = df_pred['Latitude'].mean()
        center_lon = df_pred['Longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='cartodbpositron'
        )
        
        # Prepare heatmap data based on view selection
        if prediction_year == "2025" and map_view == "Actual":
            # Load actual 2025 case locations from main dataset
            df_full = load_data()
            df_actual_2025 = df_full[df_full['Year'] == 2025].dropna(subset=['Latitude', 'Longitude'])
            
            # Create heatmap from individual case locations
            heat_data = [[row['Latitude'], row['Longitude']] for idx, row in df_actual_2025.iterrows()]
            
            HeatMap(
                heat_data,
                min_opacity=0.3,
                max_opacity=0.9,
                radius=35,
                blur=35,
                gradient={0.4: 'blue', 0.5: 'lime', 0.65: 'yellow', 0.8: 'orange', 1.0: 'red'}
            ).add_to(m)
        else:
            # Weighted heatmap by predicted cases (for 2026 or 2025 predicted view)
            heat_data = [
                [row['Latitude'], row['Longitude'], row['Predicted_Cases']] 
                for idx, row in df_pred.iterrows()
            ]
            
            HeatMap(
                heat_data,
                min_opacity=0.3,
                max_opacity=0.9,
                radius=35,
                blur=35,
                gradient={0.4: 'blue', 0.5: 'lime', 0.65: 'yellow', 0.8: 'orange', 1.0: 'red'}
            ).add_to(m)
        
        # Add district markers with prediction info
        for idx, row in df_pred.iterrows():
            if prediction_year == "2025":
                # For 2025, compare predicted vs actual
                actual = row.get('Actual_Cases_2025', 0)
                change = row['Predicted_Cases'] - actual
                if abs(change) <= 1:
                    color = 'green'  # Good prediction
                elif abs(change) <= 2:
                    color = 'orange'  # Moderate error
                else:
                    color = 'red'  # Large error
                
                popup_text = f"""
                    <b>District:</b> {row['Barangay District']}<br>
                    <b>Predicted 2025:</b> {row['Predicted_Cases']:.1f} cases<br>
                    <b>Actual 2025:</b> {actual:.0f} cases<br>
                    <b>Error:</b> {change:+.1f} cases
                """
            else:  # 2026
                # For 2026, compare predicted vs previous year
                change = row['Predicted_Cases'] - row['Prev_Year_Count']
                if change > 1:
                    color = 'red'
                elif change > 0:
                    color = 'orange'
                elif change < -1:
                    color = 'green'
                else:
                    color = 'blue'
                
                popup_text = f"""
                    <b>District:</b> {row['Barangay District']}<br>
                    <b>Predicted 2026:</b> {row['Predicted_Cases']:.1f} cases<br>
                    <b>Previous Year:</b> {row['Prev_Year_Count']:.0f} cases<br>
                    <b>Change:</b> {change:+.1f} ({(change/row['Prev_Year_Count']*100) if row['Prev_Year_Count'] > 0 else 0:+.1f}%)
                """
            
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=8,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6,
                weight=2,
                popup=popup_text
            ).add_to(m)
        
        # Add district labels if toggle is on
        if show_pred_labels:
            for idx, row in df_pred.iterrows():
                # Determine label text based on view
                if prediction_year == "2025" and map_view == "Actual":
                    actual = row.get('Actual_Cases_2025', 0)
                    label_text = f"{row['Barangay District']}<br><span style='font-size: 10px;'>({actual:.0f} actual)</span>"
                else:
                    label_text = f"{row['Barangay District']}<br><span style='font-size: 10px;'>({row['Predicted_Cases']:.1f} pred.)</span>"
                
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    icon=folium.DivIcon(html=f"""
                        <div style="
                            font-size: 11px;
                            font-weight: bold;
                            color: white;
                            text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;
                            white-space: nowrap;
                        ">
                            {label_text}
                        </div>
                    """)
                ).add_to(m)
        
        # Display map
        st.components.v1.html(m._repr_html_(), height=600)
        
        if prediction_year == "2025":
            if map_view == "Actual":
                st.info("""
                📍 **Map Legend (2025 Actual Cases):**
                - Heatmap shows actual case locations from 2025 data
                - Markers show actual case counts per district
                - 🟢 Green: Accurate prediction (error ≤1 case)
                - 🟠 Orange: Moderate error (1-2 cases)
                - 🔴 Red: Large error (>2 cases)
                """)
            else:
                st.info("""
                📍 **Map Legend (2025 Prediction Accuracy):**
                - Heatmap shows predicted case distribution
                - Markers show predicted vs actual comparison
                - 🟢 Green: Accurate prediction (error ≤1 case)
                - 🟠 Orange: Moderate error (1-2 cases)
                - 🔴 Red: Large error (>2 cases)
                """)
        else:
            st.info("""
            📍 **Map Legend (2026 Prediction):**
            - 🔴 Red: Significant increase predicted (>1 case)
            - 🟠 Orange: Slight increase predicted
            - 🔵 Blue: Stable
            - 🟢 Green: Decrease predicted
            """)
        
        # Show predictions table
        with st.expander("View Detailed Predictions"):
            # Sort by predicted cases
            df_display = df_pred.copy()
            df_display['Predicted_Cases'] = df_display['Predicted_Cases'].round(2)
            
            if prediction_year == "2025":
                actual = df_display.get('Actual_Cases_2025', 0)
                df_display['Actual_2025'] = actual
                df_display['Error'] = (df_display['Predicted_Cases'] - df_display['Actual_2025']).round(2)
                df_display['Abs_Error'] = df_display['Error'].abs()
                
                st.dataframe(
                    df_display[['Barangay District', 'Predicted_Cases', 'Actual_2025', 'Error', 'Abs_Error']]
                    .sort_values('Predicted_Cases', ascending=False)
                    .reset_index(drop=True),
                    use_container_width=True
                )
            else:  # 2026
                df_display['Change'] = df_display['Predicted_Cases'] - df_display['Prev_Year_Count']
                df_display['Change %'] = (df_display['Change'] / df_display['Prev_Year_Count'] * 100).round(1)
                
                st.dataframe(
                    df_display[['Barangay District', 'Predicted_Cases', 'Prev_Year_Count', 'Change', 'Change %']]
                    .sort_values('Predicted_Cases', ascending=False)
                    .reset_index(drop=True),
                    use_container_width=True
                )
    
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
