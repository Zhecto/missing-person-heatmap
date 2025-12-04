"""
Visualization module for generating heatmaps and charts.
Uses Folium for interactive maps and Plotly for charts.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


class HeatmapGenerator:
    """Generates interactive heatmaps for missing persons hotspots."""
    
    def __init__(self, center_coords: Tuple[float, float] = (14.5995, 120.9842)):
        """
        Initialize heatmap generator.
        
        Args:
            center_coords: (latitude, longitude) for map center. Defaults to Manila.
        """
        self.center_coords = center_coords
        self.map: Optional[folium.Map] = None
    
    def create_base_map(
        self,
        zoom_start: int = 12,
        tiles: str = 'OpenStreetMap'
    ) -> folium.Map:
        """
        Create base map centered on Manila.
        
        Args:
            zoom_start: Initial zoom level
            tiles: Map tile style ('OpenStreetMap', 'Stamen Terrain', 'CartoDB positron')
            
        Returns:
            Folium Map object
        """
        self.map = folium.Map(
            location=self.center_coords,
            zoom_start=zoom_start,
            tiles=tiles
        )
        
        return self.map
    
    def add_heatmap_layer(
        self,
        df: pd.DataFrame,
        lat_col: str = 'Latitude',
        lon_col: str = 'Longitude',
        radius: int = 15,
        blur: int = 20,
        max_zoom: int = 13
    ) -> folium.Map:
        """
        Add heatmap layer showing density of missing person reports.
        
        Args:
            df: DataFrame with coordinates
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            radius: Heatmap radius
            blur: Heatmap blur amount
            max_zoom: Maximum zoom level
            
        Returns:
            Map with heatmap layer
        """
        if self.map is None:
            self.create_base_map()
        
        # Prepare heatmap data
        heat_data = [[row[lat_col], row[lon_col]] for idx, row in df.iterrows()]
        
        # Add heatmap
        HeatMap(
            heat_data,
            radius=radius,
            blur=blur,
            max_zoom=max_zoom,
            gradient={
                0.0: 'blue',
                0.5: 'yellow',
                0.7: 'orange',
                1.0: 'red'
            }
        ).add_to(self.map)
        
        print(f"✓ Added heatmap layer with {len(heat_data)} points")
        
        return self.map
    
    def add_cluster_markers(
        self,
        df: pd.DataFrame,
        cluster_col: str = 'Cluster',
        lat_col: str = 'Latitude',
        lon_col: str = 'Longitude'
    ) -> folium.Map:
        """
        Add cluster markers to the map.
        
        Args:
            df: DataFrame with cluster labels and coordinates
            cluster_col: Name of cluster column
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            
        Returns:
            Map with cluster markers
        """
        if self.map is None:
            self.create_base_map()
        
        # Color mapping for clusters
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue']
        
        # Add markers for each cluster
        for cluster_id in sorted(df[cluster_col].unique()):
            if cluster_id == -1:  # Skip noise points
                continue
            
            cluster_data = df[df[cluster_col] == cluster_id]
            color = colors[cluster_id % len(colors)]
            
            # Create a feature group for this cluster
            cluster_group = folium.FeatureGroup(name=f'Cluster {cluster_id}')
            
            # Add cluster center marker
            center_lat = cluster_data[lat_col].mean()
            center_lon = cluster_data[lon_col].mean()
            
            folium.Marker(
                location=[center_lat, center_lon],
                popup=f'Cluster {cluster_id}<br>Size: {len(cluster_data)}',
                icon=folium.Icon(color=color, icon='info-sign'),
                tooltip=f'Cluster {cluster_id}'
            ).add_to(cluster_group)
            
            cluster_group.add_to(self.map)
        
        # Add layer control
        folium.LayerControl().add_to(self.map)
        
        print(f"✓ Added cluster markers for {len(df[cluster_col].unique())} clusters")
        
        return self.map
    
    def add_marker_cluster(
        self,
        df: pd.DataFrame,
        lat_col: str = 'Latitude',
        lon_col: str = 'Longitude'
    ) -> folium.Map:
        """
        Add interactive marker cluster layer.
        
        Args:
            df: DataFrame with coordinates
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            
        Returns:
            Map with marker cluster
        """
        if self.map is None:
            self.create_base_map()
        
        marker_cluster = MarkerCluster()
        
        for idx, row in df.iterrows():
            # Create popup content
            popup_text = f"<b>Case #{idx}</b><br>"
            if 'Date Reported Missing' in df.columns:
                popup_text += f"Date: {row['Date Reported Missing']}<br>"
            if 'Age' in df.columns:
                popup_text += f"Age: {row['Age']}<br>"
            if 'Gender' in df.columns:
                popup_text += f"Gender: {row['Gender']}<br>"
            
            folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=popup_text
            ).add_to(marker_cluster)
        
        marker_cluster.add_to(self.map)
        
        print(f"✓ Added marker cluster with {len(df)} markers")
        
        return self.map
    
    def save_map(self, output_path: Path) -> None:
        """
        Save map to HTML file.
        
        Args:
            output_path: Path to save HTML file
        """
        if self.map is None:
            raise ValueError("No map created yet")
        
        self.map.save(str(output_path))
        print(f"✓ Map saved to {output_path}")


class ChartGenerator:
    """Generates statistical charts for data analysis."""
    
    def __init__(self):
        """Initialize chart generator."""
        self.figures: List[go.Figure] = []
    
    def create_gender_distribution(self, df: pd.DataFrame, gender_col: str = 'Gender') -> go.Figure:
        """
        Create pie chart for gender distribution.
        
        Args:
            df: DataFrame with gender data
            gender_col: Name of gender column
            
        Returns:
            Plotly figure
        """
        gender_counts = df[gender_col].value_counts()
        
        fig = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title='Gender Distribution of Missing Persons',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        self.figures.append(fig)
        print("✓ Created gender distribution chart")
        
        return fig
    
    def create_age_breakdown(
        self,
        df: pd.DataFrame,
        age_col: str = 'Age_Group',
        use_bins: bool = True
    ) -> go.Figure:
        """
        Create bar chart for age breakdown.
        
        Args:
            df: DataFrame with age data
            age_col: Name of age column (or Age_Group)
            use_bins: If True, uses Age_Group; if False, bins Age column
            
        Returns:
            Plotly figure
        """
        if use_bins and 'Age_Group' in df.columns:
            age_counts = df['Age_Group'].value_counts().sort_index()
            
            fig = px.bar(
                x=age_counts.index,
                y=age_counts.values,
                title='Age Group Distribution',
                labels={'x': 'Age Group', 'y': 'Count'},
                color=age_counts.values,
                color_continuous_scale='Viridis'
            )
        else:
            fig = px.histogram(
                df,
                x='Age',
                nbins=20,
                title='Age Distribution of Missing Persons',
                labels={'Age': 'Age', 'count': 'Number of Cases'}
            )
        
        fig.update_layout(showlegend=False)
        
        self.figures.append(fig)
        print("✓ Created age breakdown chart")
        
        return fig
    
    def create_yearly_trend(
        self,
        df: pd.DataFrame,
        date_col: str = 'Date Reported Missing',
        year_col: str = 'Year'
    ) -> go.Figure:
        """
        Create line chart showing yearly trend of missing person reports.
        
        Args:
            df: DataFrame with date/year data
            date_col: Name of date column
            year_col: Name of year column
            
        Returns:
            Plotly figure
        """
        if year_col in df.columns:
            yearly_counts = df[year_col].value_counts().sort_index()
        else:
            df_copy = df.copy()
            df_copy['Year'] = pd.to_datetime(df_copy[date_col]).dt.year
            yearly_counts = df_copy['Year'].value_counts().sort_index()
        
        fig = px.line(
            x=yearly_counts.index,
            y=yearly_counts.values,
            title='Yearly Trend of Missing Person Reports',
            labels={'x': 'Year', 'y': 'Number of Reports'},
            markers=True
        )
        
        fig.update_traces(line_color='#1f77b4', line_width=3)
        fig.update_layout(hovermode='x unified')
        
        self.figures.append(fig)
        print("✓ Created yearly trend chart")
        
        return fig
    
    def create_monthly_pattern(
        self,
        df: pd.DataFrame,
        month_col: str = 'Month'
    ) -> go.Figure:
        """
        Create bar chart showing monthly patterns.
        
        Args:
            df: DataFrame with month data
            month_col: Name of month column
            
        Returns:
            Plotly figure
        """
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        monthly_counts = df[month_col].value_counts().sort_index()
        
        fig = px.bar(
            x=[month_names[i-1] for i in monthly_counts.index],
            y=monthly_counts.values,
            title='Monthly Pattern of Missing Person Reports',
            labels={'x': 'Month', 'y': 'Number of Reports'},
            color=monthly_counts.values,
            color_continuous_scale='RdYlGn_r'
        )
        
        self.figures.append(fig)
        print("✓ Created monthly pattern chart")
        
        return fig
    
    def create_cluster_size_chart(
        self,
        cluster_stats: pd.DataFrame
    ) -> go.Figure:
        """
        Create bar chart showing cluster sizes.
        
        Args:
            cluster_stats: DataFrame with cluster statistics
            
        Returns:
            Plotly figure
        """
        fig = px.bar(
            cluster_stats,
            x='Cluster',
            y='Size',
            title='Cluster Sizes',
            labels={'Cluster': 'Cluster ID', 'Size': 'Number of Cases'},
            text='Size',
            color='Size',
            color_continuous_scale='Blues'
        )
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        
        self.figures.append(fig)
        print("✓ Created cluster size chart")
        
        return fig
    
    def create_location_heatmap_chart(
        self,
        df: pd.DataFrame,
        location_col: str = 'Barangay District',
        top_n: int = 15
    ) -> go.Figure:
        """
        Create horizontal bar chart for top locations.
        
        Args:
            df: DataFrame with location data
            location_col: Name of location column
            top_n: Number of top locations to display
            
        Returns:
            Plotly figure
        """
        location_counts = df[location_col].value_counts().head(top_n)
        
        fig = px.bar(
            x=location_counts.values,
            y=location_counts.index,
            orientation='h',
            title=f'Top {top_n} Barangay Districts with Most Missing Person Reports',
            labels={'x': 'Number of Reports', 'y': 'Barangay District'},
            color=location_counts.values,
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        self.figures.append(fig)
        print("✓ Created location heatmap chart")
        
        return fig
    
    def save_all_charts(self, output_dir: Path) -> None:
        """
        Save all generated charts to HTML files.
        
        Args:
            output_dir: Directory to save charts
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, fig in enumerate(self.figures):
            output_path = output_dir / f"chart_{i+1}.html"
            fig.write_html(str(output_path))
        
        print(f"✓ Saved {len(self.figures)} charts to {output_dir}")
