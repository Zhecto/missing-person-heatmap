"""
Data preprocessing and cleaning module.
Handles missing values, date formatting, encoding, and standardization.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataCleaner:
    """Preprocesses and cleans missing persons data."""
    
    def __init__(self):
        """Initialize data cleaner with encoders and scalers."""
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.cleaning_report: List[str] = []
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'smart') -> pd.DataFrame:
        """
        Handle missing or inconsistent entries.
        
        Args:
            df: Input DataFrame
            strategy: 'drop', 'fill', or 'smart' (context-aware handling)
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        self.cleaning_report.append(f"Initial records: {initial_rows}")
        
        if strategy == 'drop':
            # Drop rows with any missing values
            df_clean = df_clean.dropna()
            dropped = initial_rows - len(df_clean)
            self.cleaning_report.append(f"Dropped {dropped} rows with missing values")
        
        elif strategy == 'smart':
            # Smart handling based on column importance
            
            # Critical columns: must have values
            critical_cols = ['Latitude', 'Longitude', 'Date Reported Missing']
            df_clean = df_clean.dropna(subset=critical_cols)
            
            # Fill missing Age with median
            if 'Age' in df_clean.columns:
                median_age = df_clean['Age'].median()
                age_missing = df_clean['Age'].isnull().sum()
                if age_missing > 0:
                    df_clean['Age'].fillna(median_age, inplace=True)
                    self.cleaning_report.append(f"Filled {age_missing} missing Age values with median: {median_age}")
            
            # Fill missing Gender with 'Unknown'
            if 'Gender' in df_clean.columns:
                gender_missing = df_clean['Gender'].isnull().sum()
                if gender_missing > 0:
                    df_clean['Gender'].fillna('Unknown', inplace=True)
                    self.cleaning_report.append(f"Filled {gender_missing} missing Gender values with 'Unknown'")
            
            # Fill missing Barangay District with 'Unknown'
            if 'Barangay District' in df_clean.columns:
                barangay_missing = df_clean['Barangay District'].isnull().sum()
                if barangay_missing > 0:
                    df_clean['Barangay District'].fillna('Unknown', inplace=True)
                    self.cleaning_report.append(f"Filled {barangay_missing} missing Barangay values")
            
            # Fill missing Time with '00:00'
            if 'Time Reported Missing' in df_clean.columns:
                time_missing = df_clean['Time Reported Missing'].isnull().sum()
                if time_missing > 0:
                    df_clean['Time Reported Missing'].fillna('00:00', inplace=True)
                    self.cleaning_report.append(f"Filled {time_missing} missing Time values with '00:00'")
            
            dropped = initial_rows - len(df_clean)
            self.cleaning_report.append(f"Dropped {dropped} rows with missing critical values")
        
        final_rows = len(df_clean)
        self.cleaning_report.append(f"Final records: {final_rows}")
        
        return df_clean
    
    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert date-related fields into usable datetime formats.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with parsed dates
        """
        df_parsed = df.copy()
        
        # Parse Date Reported Missing
        if 'Date Reported Missing' in df_parsed.columns:
            try:
                df_parsed['Date Reported Missing'] = pd.to_datetime(
                    df_parsed['Date Reported Missing'],
                    errors='coerce'
                )
                self.cleaning_report.append("✓ Parsed 'Date Reported Missing' column")
                
                # Extract useful features
                df_parsed['Year'] = df_parsed['Date Reported Missing'].dt.year
                df_parsed['Month'] = df_parsed['Date Reported Missing'].dt.month
                df_parsed['DayOfWeek'] = df_parsed['Date Reported Missing'].dt.dayofweek
                df_parsed['Quarter'] = df_parsed['Date Reported Missing'].dt.quarter
                
                self.cleaning_report.append("✓ Extracted temporal features (Year, Month, DayOfWeek, Quarter)")
            except Exception as e:
                self.cleaning_report.append(f"✗ Error parsing dates: {str(e)}")
        
        return df_parsed
    
    def encode_categorical(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encode categorical variables using Label Encoding.
        
        Args:
            df: Input DataFrame
            columns: List of columns to encode. If None, encodes Gender and Barangay District
            
        Returns:
            DataFrame with encoded columns
        """
        df_encoded = df.copy()
        
        if columns is None:
            columns = ['Gender', 'Barangay District']
        
        for col in columns:
            if col in df_encoded.columns:
                # Initialize encoder if not exists
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                # Handle any remaining NaN values
                df_encoded[col] = df_encoded[col].fillna('Unknown')
                
                # Fit and transform
                df_encoded[f'{col}_Encoded'] = self.label_encoders[col].fit_transform(
                    df_encoded[col].astype(str)
                )
                
                # Store mapping for reference
                mapping = dict(zip(
                    self.label_encoders[col].classes_,
                    self.label_encoders[col].transform(self.label_encoders[col].classes_)
                ))
                
                self.cleaning_report.append(f"✓ Encoded '{col}': {len(mapping)} unique values")
        
        return df_encoded
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate or irrelevant records.
        
        Args:
            df: Input DataFrame
            subset: Columns to check for duplicates. If None, checks Person ID
            
        Returns:
            DataFrame without duplicates
        """
        df_dedup = df.copy()
        initial_rows = len(df_dedup)
        
        if subset is None and 'Person ID' in df_dedup.columns:
            subset = ['Person ID']
        
        df_dedup = df_dedup.drop_duplicates(subset=subset, keep='first')
        
        duplicates_removed = initial_rows - len(df_dedup)
        self.cleaning_report.append(f"Removed {duplicates_removed} duplicate records")
        
        return df_dedup
    
    def standardize_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure latitude and longitude are in consistent format.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized coordinates
        """
        df_std = df.copy()
        
        # Ensure numeric types
        if 'Latitude' in df_std.columns:
            df_std['Latitude'] = pd.to_numeric(df_std['Latitude'], errors='coerce')
        
        if 'Longitude' in df_std.columns:
            df_std['Longitude'] = pd.to_numeric(df_std['Longitude'], errors='coerce')
        
        # Remove rows with invalid coordinates
        before = len(df_std)
        df_std = df_std.dropna(subset=['Latitude', 'Longitude'])
        after = len(df_std)
        
        removed = before - after
        if removed > 0:
            self.cleaning_report.append(f"Removed {removed} records with invalid coordinates")
        
        self.cleaning_report.append("✓ Standardized coordinate formats")
        
        return df_std
    
    def normalize_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize age values and create age groups.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with normalized age
        """
        df_norm = df.copy()
        
        if 'Age' in df_norm.columns:
            # Ensure numeric
            df_norm['Age'] = pd.to_numeric(df_norm['Age'], errors='coerce')
            
            # Create age groups
            df_norm['Age_Group'] = pd.cut(
                df_norm['Age'],
                bins=[0, 12, 17, 30, 50, 100],
                labels=['Child (0-12)', 'Teen (13-17)', 'Young Adult (18-30)', 
                       'Adult (31-50)', 'Senior (51+)']
            )
            
            self.cleaning_report.append("✓ Created age groups")
        
        return df_norm
    
    def preprocess_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run full preprocessing pipeline.
        
        Args:
            df: Raw input DataFrame
            
        Returns:
            Fully preprocessed DataFrame
        """
        self.cleaning_report = ["=== PREPROCESSING PIPELINE ==="]
        
        # Step 1: Remove duplicates
        df_clean = self.remove_duplicates(df)
        
        # Step 2: Handle missing values
        df_clean = self.handle_missing_values(df_clean, strategy='smart')
        
        # Step 3: Standardize coordinates
        df_clean = self.standardize_coordinates(df_clean)
        
        # Step 4: Parse dates
        df_clean = self.parse_dates(df_clean)
        
        # Step 5: Normalize age
        df_clean = self.normalize_age(df_clean)
        
        # Step 6: Encode categorical variables
        df_clean = self.encode_categorical(df_clean)
        
        self.cleaning_report.append("=== PIPELINE COMPLETE ===")
        
        return df_clean
    
    def get_cleaning_report(self) -> str:
        """
        Get formatted preprocessing report.
        
        Returns:
            Report string
        """
        return "\n".join(self.cleaning_report)
