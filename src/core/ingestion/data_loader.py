"""
Data loading module for missing persons dataset.
Handles CSV reading, validation, and initial data structure setup.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime


class DataLoader:
    """Loads and validates missing persons CSV data."""
    
    REQUIRED_COLUMNS = [
        'Person ID',
        'Gender',
        'Age',
        'Date Reported Missing',
        'Time Reported Missing',
        'Location last seen',
        'Latitude',
        'Longitude',
        'Barangay District',
        'Post URL'
    ]
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to the CSV file. If None, uses default data/ directory.
        """
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self.validation_errors: List[str] = []
    
    def load_csv(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load CSV file into a pandas DataFrame.
        
        Args:
            file_path: Path to CSV file. Uses instance data_path if not provided.
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV is invalid
        """
        path = file_path or self.data_path
        
        if path is None:
            raise ValueError("No file path provided")
        
        if not Path(path).exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        
        try:
            self.df = pd.read_csv(path)
            print(f"✓ Loaded {len(self.df)} records from {path}")
            return self.df
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {str(e)}")
    
    def validate_schema(self, df: Optional[pd.DataFrame] = None) -> bool:
        """
        Validate that the DataFrame has all required columns.
        
        Args:
            df: DataFrame to validate. Uses instance df if not provided.
            
        Returns:
            True if valid, False otherwise
        """
        data = df if df is not None else self.df
        
        if data is None:
            self.validation_errors.append("No data loaded")
            return False
        
        self.validation_errors = []
        missing_cols = set(self.REQUIRED_COLUMNS) - set(data.columns)
        
        if missing_cols:
            self.validation_errors.append(
                f"Missing required columns: {', '.join(missing_cols)}"
            )
            return False
        
        print("✓ Schema validation passed")
        return True
    
    def get_data_summary(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Get summary statistics of the loaded data.
        
        Args:
            df: DataFrame to summarize. Uses instance df if not provided.
            
        Returns:
            Dictionary with summary information
        """
        data = df if df is not None else self.df
        
        if data is None:
            return {"error": "No data loaded"}
        
        summary = {
            "total_records": len(data),
            "columns": list(data.columns),
            "missing_values": data.isnull().sum().to_dict(),
            "date_range": {
                "earliest": None,
                "latest": None
            },
            "unique_locations": data['Barangay District'].nunique() if 'Barangay District' in data.columns else 0,
            "gender_distribution": data['Gender'].value_counts().to_dict() if 'Gender' in data.columns else {},
            "age_stats": {
                "mean": float(data['Age'].mean()) if 'Age' in data.columns and pd.api.types.is_numeric_dtype(data['Age']) else None,
                "min": float(data['Age'].min()) if 'Age' in data.columns and pd.api.types.is_numeric_dtype(data['Age']) else None,
                "max": float(data['Age'].max()) if 'Age' in data.columns and pd.api.types.is_numeric_dtype(data['Age']) else None,
            }
        }
        
        return summary
    
    def validate_coordinates(self, df: Optional[pd.DataFrame] = None) -> bool:
        """
        Validate latitude and longitude values are within Metro Manila (NCR) bounds.
        Metro Manila approximate bounds: Lat 14.35-14.85, Lon 120.90-121.15
        
        Args:
            df: DataFrame to validate. Uses instance df if not provided.
            
        Returns:
            True if coordinates are valid, False otherwise
        """
        data = df if df is not None else self.df
        
        if data is None:
            self.validation_errors.append("No data loaded")
            return False
        
        # Metro Manila (NCR) bounds (approximate, with tolerance)
        LAT_MIN, LAT_MAX = 14.0, 15.0
        LON_MIN, LON_MAX = 120.5, 121.5
        
        lat_col = 'Latitude'
        lon_col = 'Longitude'
        
        if lat_col not in data.columns or lon_col not in data.columns:
            self.validation_errors.append("Missing coordinate columns")
            return False
        
        # Check for valid numeric values
        invalid_coords = (
            (data[lat_col] < LAT_MIN) | (data[lat_col] > LAT_MAX) |
            (data[lon_col] < LON_MIN) | (data[lon_col] > LON_MAX)
        )
        
        invalid_count = invalid_coords.sum()
        
        if invalid_count > 0:
            self.validation_errors.append(
                f"{invalid_count} records have coordinates outside Manila bounds"
            )
            print(f"⚠ {invalid_count} records with invalid coordinates")
            return False
        
        print("✓ Coordinate validation passed")
        return True
    
    def get_validation_report(self) -> str:
        """
        Get a formatted validation report.
        
        Returns:
            Formatted string with validation results
        """
        if not self.validation_errors:
            return "✓ All validations passed"
        
        report = "Validation Errors:\n"
        for i, error in enumerate(self.validation_errors, 1):
            report += f"{i}. {error}\n"
        
        return report


def load_and_validate(file_path: Path) -> tuple[pd.DataFrame, DataLoader]:
    """
    Convenience function to load and validate data in one call.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Tuple of (DataFrame, DataLoader instance)
    """
    loader = DataLoader(file_path)
    df = loader.load_csv()
    
    # Run validations
    loader.validate_schema()
    loader.validate_coordinates()
    
    # Print report
    print(loader.get_validation_report())
    
    return df, loader
