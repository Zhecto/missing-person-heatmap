"""
Prediction module for forecasting future missing person hotspots.
Uses historical patterns to predict next-year spatial distribution.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Poisson
import warnings
warnings.filterwarnings('ignore')


class SpatialPredictor:
    """Predicts future hotspot locations based on historical data."""
    
    def __init__(self):
        """Initialize spatial predictor."""
        self.model: Optional[object] = None
        self.poisson_model: Optional[object] = None
        self.scaler = StandardScaler()
        self.model_type: str = ''
        self.feature_columns: List[str] = []
        self.is_fitted: bool = False
        self.poisson_fitted: bool = False
    
    def prepare_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features for prediction.
        
        Args:
            df: DataFrame with date information
            
        Returns:
            DataFrame with temporal features
        """
        df_features = df.copy()
        
        if 'Date Reported Missing' in df_features.columns:
            df_features['Date Reported Missing'] = pd.to_datetime(
                df_features['Date Reported Missing'],
                errors='coerce'
            )
            
            # Extract temporal features
            df_features['Year'] = df_features['Date Reported Missing'].dt.year
            df_features['Month'] = df_features['Date Reported Missing'].dt.month
            df_features['Quarter'] = df_features['Date Reported Missing'].dt.quarter
            df_features['DayOfWeek'] = df_features['Date Reported Missing'].dt.dayofweek
            df_features['DayOfYear'] = df_features['Date Reported Missing'].dt.dayofyear
            
            # Cyclical encoding for month and day of week
            df_features['Month_Sin'] = np.sin(2 * np.pi * df_features['Month'] / 12)
            df_features['Month_Cos'] = np.cos(2 * np.pi * df_features['Month'] / 12)
            df_features['DayOfWeek_Sin'] = np.sin(2 * np.pi * df_features['DayOfWeek'] / 7)
            df_features['DayOfWeek_Cos'] = np.cos(2 * np.pi * df_features['DayOfWeek'] / 7)
        
        return df_features
    
    def aggregate_by_location_time(
        self,
        df: pd.DataFrame,
        location_col: str = 'Barangay District',
        time_col: str = 'Year'
    ) -> pd.DataFrame:
        """
        Aggregate cases by location and time period.
        
        Args:
            df: Input DataFrame
            location_col: Column name for location
            time_col: Column name for time period (Year, Month, etc.)
            
        Returns:
            Aggregated DataFrame
        """
        agg_df = df.groupby([location_col, time_col]).agg({
            'Person ID': 'count',  # Number of cases
            'Latitude': 'mean',
            'Longitude': 'mean',
            'Age': 'mean'
        }).reset_index()
        
        agg_df.rename(columns={'Person ID': 'Case_Count'}, inplace=True)
        
        return agg_df
    
    def train_location_predictor(
        self,
        df: pd.DataFrame,
        location_col: str = 'Barangay District',
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train model to predict high-risk locations.
        
        Args:
            df: Training DataFrame with temporal and spatial features
            location_col: Target column (location to predict)
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        df_prep = self.prepare_temporal_features(df)
        
        # Define features
        feature_cols = [
            'Latitude', 'Longitude', 'Month', 'Quarter', 'DayOfWeek',
            'Month_Sin', 'Month_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos'
        ]
        
        # Add encoded demographics if available
        if 'Age' in df_prep.columns:
            feature_cols.append('Age')
        if 'Gender_Encoded' in df_prep.columns:
            feature_cols.append('Gender_Encoded')
        
        # Filter available columns
        self.feature_columns = [col for col in feature_cols if col in df_prep.columns]
        
        # Prepare X and y
        X = df_prep[self.feature_columns].dropna()
        y = df_prep.loc[X.index, location_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest Classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.model_type = 'location_classifier'
        self.is_fitted = True
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'n_features': len(self.feature_columns),
            'n_samples': len(X)
        }
        
        print(f"✓ Location predictor trained: {test_score:.3f} accuracy")
        
        return metrics
    
    def train_hotspot_intensity_predictor(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train model to predict intensity of cases in a location.
        
        Args:
            df: Aggregated DataFrame with case counts per location/time
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        # Aggregate by location and year
        agg_df = self.aggregate_by_location_time(df)
        
        # Create lag features (previous year's count)
        agg_df = agg_df.sort_values(['Barangay District', 'Year'])
        agg_df['Prev_Year_Count'] = agg_df.groupby('Barangay District')['Case_Count'].shift(1)
        agg_df = agg_df.dropna(subset=['Prev_Year_Count'])
        
        # Define features
        feature_cols = ['Latitude', 'Longitude', 'Year', 'Prev_Year_Count', 'Age']
        self.feature_columns = [col for col in feature_cols if col in agg_df.columns]
        
        # Prepare X and y
        X = agg_df[self.feature_columns]
        y = agg_df['Case_Count']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Gradient Boosting Regressor
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.model_type = 'intensity_regressor'
        self.is_fitted = True
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        metrics = {
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'n_features': len(self.feature_columns),
            'n_samples': len(X)
        }
        
        print(f"✓ Intensity predictor trained: R² = {metrics['test_r2']:.3f}")
        
        return metrics
    
    def train_poisson_regressor(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Train Poisson Regression model to predict count data.
        
        Poisson Regression is specifically designed for count data (non-negative integers)
        and models the rate of incident occurrence. It's more interpretable than
        Gradient Boosting for understanding factors affecting incident rates.
        
        Args:
            df: Aggregated DataFrame with case counts per location/time
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        # Aggregate by location and year
        agg_df = self.aggregate_by_location_time(df)
        
        # Create lag features (previous year's count)
        agg_df = agg_df.sort_values(['Barangay District', 'Year'])
        agg_df['Prev_Year_Count'] = agg_df.groupby('Barangay District')['Case_Count'].shift(1)
        agg_df = agg_df.dropna(subset=['Prev_Year_Count'])
        
        # Define features
        feature_cols = ['Latitude', 'Longitude', 'Year', 'Prev_Year_Count', 'Age']
        self.feature_columns = [col for col in feature_cols if col in agg_df.columns]
        
        # Prepare X and y
        X = agg_df[self.feature_columns]
        y = agg_df['Case_Count']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features for numerical stability
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Add constant for intercept
        X_train_scaled = sm.add_constant(X_train_scaled)
        X_test_scaled = sm.add_constant(X_test_scaled)
        
        # Train Poisson Regression
        try:
            self.poisson_model = sm.GLM(
                y_train, 
                X_train_scaled,
                family=sm.families.Poisson()
            ).fit()
            
            self.poisson_fitted = True
            
            # Predictions
            y_pred_train = self.poisson_model.predict(X_train_scaled)
            y_pred_test = self.poisson_model.predict(X_test_scaled)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            metrics = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'aic': self.poisson_model.aic,
                'bic': self.poisson_model.bic,
                'deviance': self.poisson_model.deviance,
                'n_features': len(self.feature_columns),
                'n_samples': len(X),
                'model_type': 'poisson_regression'
            }
            
            print(f"✓ Poisson Regression trained: R² = {test_r2:.3f}, AIC = {metrics['aic']:.2f}")
            
            return metrics
            
        except Exception as e:
            print(f"✗ Poisson Regression training failed: {str(e)}")
            raise
    
    def predict_next_year_hotspots_poisson(
        self,
        df: pd.DataFrame,
        next_year: int,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Predict top hotspot locations for next year using Poisson Regression.
        
        Args:
            df: Historical data
            next_year: Year to predict for
            top_n: Number of top hotspots to return
            
        Returns:
            DataFrame with predicted hotspots and rate ratios
        """
        if not self.poisson_fitted:
            raise ValueError("Poisson model not trained yet. Call train_poisson_regressor first.")
        
        # Get unique locations with their coordinates
        locations = df.groupby('Barangay District').agg({
            'Latitude': 'mean',
            'Longitude': 'mean',
            'Age': 'mean'
        }).reset_index()
        
        # Get previous year's count
        agg_df = self.aggregate_by_location_time(df)
        prev_year_data = agg_df[agg_df['Year'] == next_year - 1][
            ['Barangay District', 'Case_Count']
        ].rename(columns={'Case_Count': 'Prev_Year_Count'})
        
        # Merge
        locations = locations.merge(prev_year_data, on='Barangay District', how='left')
        locations['Prev_Year_Count'].fillna(0, inplace=True)
        locations['Year'] = next_year
        
        # Prepare features
        X_pred = locations[self.feature_columns]
        X_pred_scaled = self.scaler.transform(X_pred)
        X_pred_scaled = sm.add_constant(X_pred_scaled)
        
        # Predict
        predictions = self.poisson_model.predict(X_pred_scaled)
        locations['Predicted_Cases'] = predictions
        
        # Calculate rate ratio (effect size)
        locations['Rate_Ratio'] = locations['Predicted_Cases'] / (locations['Prev_Year_Count'] + 1)
        
        # Get top N
        top_hotspots = locations.nlargest(top_n, 'Predicted_Cases')[
            ['Barangay District', 'Latitude', 'Longitude', 'Predicted_Cases', 
             'Prev_Year_Count', 'Rate_Ratio']
        ]
        
        print(f"✓ Predicted top {top_n} hotspots for {next_year} (Poisson Regression)")
        
        return top_hotspots
    
    def get_poisson_coefficients(self) -> pd.DataFrame:
        """
        Get interpretable coefficients from Poisson Regression.
        
        Returns:
            DataFrame with coefficients, their exponentials (rate ratios), 
            p-values, and confidence intervals
        """
        if not self.poisson_fitted:
            raise ValueError("Poisson model not trained yet")
        
        # Get coefficients
        coef_names = ['Intercept'] + self.feature_columns
        coefficients = self.poisson_model.params
        std_errors = self.poisson_model.bse
        p_values = self.poisson_model.pvalues
        conf_int = self.poisson_model.conf_int()
        
        # Create DataFrame
        coef_df = pd.DataFrame({
            'Feature': coef_names,
            'Coefficient': coefficients.values,
            'Std_Error': std_errors.values,
            'Rate_Ratio': np.exp(coefficients.values),  # Exponentiated coefficient
            'P_Value': p_values.values,
            'CI_Lower': conf_int.iloc[:, 0].values,
            'CI_Upper': conf_int.iloc[:, 1].values,
            'Significant': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' 
                          for p in p_values.values]
        })
        
        return coef_df
    
    def compare_models(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> pd.DataFrame:
        """
        Train and compare both Gradient Boosting and Poisson Regression models.
        
        Args:
            df: Training data
            test_size: Proportion of data for testing
            
        Returns:
            DataFrame comparing model performance
        """
        print("\n🔬 Training and comparing prediction models...")
        print("=" * 60)
        
        # Train Gradient Boosting
        print("\n1️⃣  Training Gradient Boosting Regressor...")
        gb_metrics = self.train_hotspot_intensity_predictor(df, test_size)
        
        # Train Poisson Regression
        print("\n2️⃣  Training Poisson Regression...")
        poisson_metrics = self.train_poisson_regressor(df, test_size)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Model': ['Gradient Boosting', 'Poisson Regression'],
            'Test_R2': [gb_metrics['test_r2'], poisson_metrics['test_r2']],
            'Test_RMSE': [gb_metrics['test_rmse'], poisson_metrics['test_rmse']],
            'Train_R2': [gb_metrics['train_r2'], poisson_metrics['train_r2']],
            'AIC': [None, poisson_metrics['aic']],
            'Interpretability': ['Low (Black-box)', 'High (Coefficients)'],
            'Use_Case': ['Maximum Accuracy', 'Statistical Inference']
        })
        
        print("\n" + "=" * 60)
        print("📊 Model Comparison:")
        print(comparison.to_string(index=False))
        print("=" * 60)
        
        return comparison

    def predict_next_year_hotspots(
        self,
        df: pd.DataFrame,
        next_year: int,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Predict top hotspot locations for next year.
        
        Args:
            df: Historical data
            next_year: Year to predict for
            top_n: Number of top hotspots to return
            
        Returns:
            DataFrame with predicted hotspots
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet. Call train_hotspot_intensity_predictor first.")
        
        if self.model_type != 'intensity_regressor':
            raise ValueError("Wrong model type. Use intensity_regressor for this prediction.")
        
        # Get unique locations with their coordinates
        locations = df.groupby('Barangay District').agg({
            'Latitude': 'mean',
            'Longitude': 'mean',
            'Age': 'mean'
        }).reset_index()
        
        # Get previous year's count
        agg_df = self.aggregate_by_location_time(df)
        prev_year_data = agg_df[agg_df['Year'] == next_year - 1][
            ['Barangay District', 'Case_Count']
        ].rename(columns={'Case_Count': 'Prev_Year_Count'})
        
        # Merge
        locations = locations.merge(prev_year_data, on='Barangay District', how='left')
        locations['Prev_Year_Count'].fillna(0, inplace=True)
        locations['Year'] = next_year
        
        # Prepare features
        X_pred = locations[self.feature_columns]
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Predict
        predictions = self.model.predict(X_pred_scaled)
        locations['Predicted_Cases'] = predictions
        
        # Get top N
        top_hotspots = locations.nlargest(top_n, 'Predicted_Cases')[
            ['Barangay District', 'Latitude', 'Longitude', 'Predicted_Cases', 'Prev_Year_Count']
        ]
        
        print(f"✓ Predicted top {top_n} hotspots for {next_year}")
        
        return top_hotspots
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        if self.model_type == 'location_classifier':
            importances = self.model.feature_importances_
        elif self.model_type == 'intensity_regressor':
            importances = self.model.feature_importances_
        else:
            raise ValueError("Model type doesn't support feature importance")
        
        feature_importance = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return feature_importance
    
    def simulate_future_scenarios(
        self,
        df: pd.DataFrame,
        years: List[int],
        location_col: str = 'Barangay District'
    ) -> Dict[int, pd.DataFrame]:
        """
        Simulate multiple future year scenarios.
        
        Args:
            df: Historical data
            years: List of years to predict
            location_col: Location column name
            
        Returns:
            Dictionary mapping year to prediction DataFrame
        """
        scenarios = {}
        
        for year in years:
            try:
                predictions = self.predict_next_year_hotspots(df, year, top_n=10)
                scenarios[year] = predictions
                print(f"  Scenario for {year}: {len(predictions)} locations")
            except Exception as e:
                print(f"  Failed to predict for {year}: {str(e)}")
        
        return scenarios


class TrendAnalyzer:
    """Analyzes temporal trends in missing persons data."""
    
    def __init__(self):
        """Initialize trend analyzer."""
        pass
    
    def calculate_yearly_growth_rate(self, df: pd.DataFrame, year_col: str = 'Year') -> pd.DataFrame:
        """
        Calculate year-over-year growth rate.
        
        Args:
            df: DataFrame with year information
            year_col: Name of year column
            
        Returns:
            DataFrame with growth rates
        """
        yearly_counts = df[year_col].value_counts().sort_index()
        
        growth_rates = []
        for i in range(1, len(yearly_counts)):
            prev_year = yearly_counts.iloc[i-1]
            curr_year = yearly_counts.iloc[i]
            growth_rate = ((curr_year - prev_year) / prev_year) * 100
            
            growth_rates.append({
                'Year': yearly_counts.index[i],
                'Cases': curr_year,
                'Growth_Rate': f"{growth_rate:.1f}%"
            })
        
        return pd.DataFrame(growth_rates)
    
    def identify_seasonal_patterns(self, df: pd.DataFrame, month_col: str = 'Month') -> Dict[str, any]:
        """
        Identify seasonal patterns in missing person reports.
        
        Args:
            df: DataFrame with month information
            month_col: Name of month column
            
        Returns:
            Dictionary with seasonal insights
        """
        monthly_avg = df.groupby(month_col).size()
        
        peak_month = monthly_avg.idxmax()
        low_month = monthly_avg.idxmin()
        
        season_map = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }
        
        seasonal_counts = df[month_col].map(season_map).value_counts()
        
        return {
            'peak_month': peak_month,
            'lowest_month': low_month,
            'monthly_average': monthly_avg.to_dict(),
            'seasonal_distribution': seasonal_counts.to_dict()
        }
