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

st.title("🔍 Manila City Missing Person Cases")

@st.cache_data
def load_data():
    data_path = Path(__file__).parent / "notebook" / "Missing People - cleaned.csv"
    return pd.read_csv(data_path)

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
        years = df['Year'].dropna().nunique()
        st.metric("Years", years)
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

except Exception as e:
    st.error(f"Error loading data: {e}")
