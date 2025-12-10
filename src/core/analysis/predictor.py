"""
SIMPLIFIED Prediction module for missing person hotspot forecasting.
Uses only the configured model from settings.yaml - much cleaner!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SpatialPredictor:
    """Simple predictor using model from config/settings.yaml"""
    
    def __init__(self):
        # Load model choice from config
        config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'settings.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.configured_model = config.get('modeling', {}).get('prediction_model', 'poisson')
        else:
            self.configured_model = 'poisson'
        
        # Initialize
        self.model = None
        self.scaler = StandardScaler()
        self.model_type = ''
        self.feature_columns = []
        self.is_fitted = False
    
    def aggregate_by_location_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate cases by location and year."""
        # Ensure Year column exists
        if 'Year' not in df.columns:
            if 'Date Reported Missing' in df.columns:
                df = df.copy()
                df['Date Reported Missing'] = pd.to_datetime(df['Date Reported Missing'], errors='coerce')
                df['Year'] = df['Date Reported Missing'].dt.year
            else:
                raise ValueError("DataFrame must have 'Year' or 'Date Reported Missing' column")
        
        agg_df = df.groupby(['Barangay District', 'Year']).agg({
            'Person ID': 'count',
            'Latitude': 'mean',
            'Longitude': 'mean',
            'Age': 'mean'
        }).reset_index()
        agg_df.rename(columns={'Person ID': 'Case_Count'}, inplace=True)
        return agg_df
    
    def train_configured_model(self, df: pd.DataFrame, model_name: Optional[str] = None) -> Dict:
        """
        Train the model specified in settings.yaml
        
        Simple 3-step process:
        1. Aggregate data by location/year
        2. Train configured model (Poisson or Gradient Boosting)
        3. Return performance metrics
        """
        model_to_use = model_name if model_name else self.configured_model
        print(f"🤖 Training {model_to_use.replace('_', ' ').title()}...")
        
        # Step 1: Aggregate by location and year
        agg_df = self.aggregate_by_location_time(df)
        agg_df = agg_df.sort_values(['Barangay District', 'Year'])
        agg_df['Prev_Year_Count'] = agg_df.groupby('Barangay District')['Case_Count'].shift(1)
        agg_df = agg_df.dropna(subset=['Prev_Year_Count'])
        
        # Prepare features
        self.feature_columns = ['Latitude', 'Longitude', 'Year', 'Prev_Year_Count', 'Age']
        X = agg_df[self.feature_columns]
        y = agg_df['Case_Count']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Step 2: Train model with hyperparameter tuning
        if model_to_use == 'poisson':
            param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0], 'max_iter': [100, 200]}
            base_model = PoissonRegressor()
        elif model_to_use == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [5, 7]
            }
            base_model = GradientBoostingRegressor(random_state=42)
        else:
            raise ValueError(f"Unknown model: {model_to_use}")
        
        grid_search = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        self.model = grid_search.best_estimator_
        self.model_type = model_to_use
        self.is_fitted = True
        
        # Step 3: Evaluate
        y_pred_test = self.model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"✓ Trained! Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
        
        return {
            'model': model_to_use,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'train_r2': r2_score(y_train, self.model.predict(X_train_scaled)),
            'overfit_gap': r2_score(y_train, self.model.predict(X_train_scaled)) - test_r2
        }
    
    def predict_next_year_hotspots(self, df: pd.DataFrame, next_year: int, top_n: int = 10) -> pd.DataFrame:
        """Predict top N hotspots for next year."""
        if not self.is_fitted:
            raise ValueError("Train model first!")
        
        # Get locations
        locations = df.groupby('Barangay District').agg({
            'Latitude': 'mean', 'Longitude': 'mean', 'Age': 'mean'
        }).reset_index()
        
        # Get previous year counts
        agg_df = self.aggregate_by_location_time(df)
        prev_year = agg_df[agg_df['Year'] == next_year - 1][['Barangay District', 'Case_Count']]
        prev_year = prev_year.rename(columns={'Case_Count': 'Prev_Year_Count'})
        
        locations = locations.merge(prev_year, on='Barangay District', how='left')
        locations['Prev_Year_Count'].fillna(0, inplace=True)
        locations['Year'] = next_year
        
        # Predict
        X_pred = self.scaler.transform(locations[self.feature_columns])
        locations['Predicted_Cases'] = self.model.predict(X_pred)
        
        # Store all predictions for heatmap generation
        self.all_predictions = locations.copy()
        
        top = locations.nlargest(top_n, 'Predicted_Cases')[
            ['Barangay District', 'Latitude', 'Longitude', 'Predicted_Cases', 'Prev_Year_Count']
        ]
        
        print(f"✓ Predicted top {top_n} hotspots for {next_year}")
        return top
    
    def get_all_predictions(self) -> pd.DataFrame:
        """Get predictions for all locations (for heatmap generation)."""
        if not hasattr(self, 'all_predictions'):
            raise ValueError("Run predict_next_year_hotspots first!")
        return self.all_predictions
    
    def generate_prediction_heatmap(self, next_year: int, output_path: str = None) -> str:
        """
        Generate interactive heatmap showing predicted hotspot intensity.
        
        Args:
            next_year: Year being predicted
            output_path: Where to save the HTML file
            
        Returns:
            Path to generated HTML file
        """
        import folium
        from folium.plugins import HeatMap
        
        if not hasattr(self, 'all_predictions'):
            raise ValueError("Run predict_next_year_hotspots first!")
        
        predictions = self.all_predictions
        
        # Create base map centered on mean location
        center_lat = predictions['Latitude'].mean()
        center_lon = predictions['Longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles='cartodbpositron'
        )
        
        # Prepare heatmap data (weighted by predicted cases)
        heat_data = []
        for _, row in predictions.iterrows():
            # Each prediction gets weight based on predicted cases
            weight = max(row['Predicted_Cases'], 0.1)  # Minimum weight 0.1
            heat_data.append([row['Latitude'], row['Longitude'], weight])
        
        # Add heatmap layer
        HeatMap(
            heat_data,
            min_opacity=0.3,
            max_opacity=0.8,
            radius=15,
            blur=20,
            gradient={
                0.0: 'blue',
                0.3: 'lime',
                0.5: 'yellow',
                0.7: 'orange',
                1.0: 'red'
            }
        ).add_to(m)
        
        # Add markers for top hotspots
        top_locations = predictions.nlargest(10, 'Predicted_Cases')
        for idx, row in top_locations.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=8,
                popup=f"""
                    <b>{row['Barangay District']}</b><br>
                    Predicted Cases: {row['Predicted_Cases']:.2f}<br>
                    Previous Year: {row['Prev_Year_Count']:.0f}
                """,
                color='darkred',
                fill=True,
                fillColor='red',
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
        
        # Add title
        title_html = f'''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 400px; height: 60px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:16px; padding: 10px">
            <b>Predicted Hotspots for {next_year}</b><br>
            <span style="font-size:12px">Model: {self.model_type.replace('_', ' ').title()}</span>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Save map
        if output_path is None:
            output_path = f"data/outputs/prediction_heatmap_{next_year}.html"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(output_path)
        
        print(f"✓ Prediction heatmap saved to {output_path}")
        return output_path
    
    def generate_actual_heatmap(self, df: pd.DataFrame, year: int, output_path: str = None) -> str:
        """
        Generate heatmap showing actual incident density for a specific year.
        
        Args:
            df: DataFrame with actual data (must have Year, Latitude, Longitude, Barangay District)
            year: Year to visualize
            output_path: Where to save the HTML file
            
        Returns:
            Path to generated HTML file
        """
        import folium
        from folium.plugins import HeatMap
        
        # Filter to specific year
        df_year = df[df['Year'] == year].copy()
        
        if len(df_year) == 0:
            raise ValueError(f"No data available for year {year}")
        
        # Aggregate by location to get actual case counts
        location_counts = df_year.groupby('Barangay District').agg({
            'Latitude': 'mean',
            'Longitude': 'mean'
        }).reset_index()
        location_counts['Actual_Cases'] = df_year.groupby('Barangay District').size().values
        
        # Create base map centered on mean location
        center_lat = location_counts['Latitude'].mean()
        center_lon = location_counts['Longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles='cartodbpositron'
        )
        
        # Prepare heatmap data (weighted by actual cases)
        heat_data = []
        for _, row in location_counts.iterrows():
            weight = max(row['Actual_Cases'], 0.1)
            heat_data.append([row['Latitude'], row['Longitude'], weight])
        
        # Add heatmap layer with same style as predictions
        HeatMap(
            heat_data,
            min_opacity=0.3,
            max_opacity=0.8,
            radius=15,
            blur=20,
            gradient={
                0.0: 'blue',
                0.3: 'lime',
                0.5: 'yellow',
                0.7: 'orange',
                1.0: 'red'
            }
        ).add_to(m)
        
        # Add markers for top 10 actual hotspots
        top_locations = location_counts.nlargest(10, 'Actual_Cases')
        for idx, row in top_locations.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=8,
                popup=f"""
                    <b>{row['Barangay District']}</b><br>
                    Actual Cases: {row['Actual_Cases']:.0f}
                """,
                color='darkred',
                fill=True,
                fillColor='red',
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
        
        # Add title
        title_html = f'''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 400px; height: 60px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:16px; padding: 10px">
            <b>Actual Data for {year}</b><br>
            <span style="font-size:12px">Total Records: {len(df_year)}</span>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Save map
        if output_path is None:
            output_path = f"data/outputs/actual_{year}_heatmap.html"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(output_path)
        
        print(f"✓ Actual data heatmap saved to {output_path}")
        return output_path
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Train model first!")
        
        if self.model_type == 'gradient_boosting':
            importances = self.model.feature_importances_
        elif self.model_type == 'poisson':
            importances = np.abs(self.model.coef_)
        else:
            raise ValueError("Model doesn't support feature importance")
        
        return pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)


class TrendAnalyzer:
    """Simple trend analysis."""
    
    def calculate_yearly_growth_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate year-over-year growth."""
        yearly = df.groupby('Year').size().reset_index(name='Cases')
        yearly['Growth_Rate'] = yearly['Cases'].pct_change() * 100
        yearly['Growth_Rate'] = yearly['Growth_Rate'].apply(lambda x: f"{x:.1f}%")
        return yearly
    
    def identify_seasonal_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify seasonal patterns."""
        monthly = df.groupby('Month').size()
        seasonal_map = {12: 'Summer', 1: 'Summer', 2: 'Summer',
                       3: 'Fall', 4: 'Fall', 5: 'Fall',
                       6: 'Winter', 7: 'Winter', 8: 'Winter',
                       9: 'Spring', 10: 'Spring', 11: 'Spring'}
        seasonal = df.copy()
        seasonal['Season'] = seasonal['Month'].map(seasonal_map)
        seasonal_dist = seasonal['Season'].value_counts().to_dict()
        
        return {
            'peak_month': monthly.idxmax(),
            'lowest_month': monthly.idxmin(),
            'seasonal_distribution': seasonal_dist
        }
