"""
Clustering analysis module for identifying hotspot patterns.
Implements K-means and DBSCAN for spatial clustering.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


class ClusteringModel:
    """Performs clustering analysis on missing persons data."""
    
    def __init__(self):
        """Initialize clustering model."""
        self.model: Optional[object] = None
        self.scaler = StandardScaler()
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.n_clusters: int = 0
        self.model_type: str = ''
        self.feature_columns: List[str] = []
    
    def fit_kmeans(
        self, 
        df: pd.DataFrame, 
        n_clusters: int = 5,
        features: Optional[List[str]] = None
    ) -> 'ClusteringModel':
        """
        Fit K-means clustering model.
        
        Args:
            df: Input DataFrame with coordinates
            n_clusters: Number of clusters to create
            features: Feature columns to use. Defaults to ['Latitude', 'Longitude']
            
        Returns:
            Self for method chaining
        """
        if features is None:
            features = ['Latitude', 'Longitude']
        
        self.feature_columns = features
        
        # Extract features
        X = df[features].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-means
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        self.labels_ = self.model.fit_predict(X_scaled)
        self.cluster_centers_ = self.scaler.inverse_transform(self.model.cluster_centers_)
        self.n_clusters = n_clusters
        self.model_type = 'kmeans'
        
        print(f"✓ K-means clustering complete: {n_clusters} clusters identified")
        
        return self
    
    def fit_dbscan(
        self,
        df: pd.DataFrame,
        eps: float = 0.01,
        min_samples: int = 5,
        features: Optional[List[str]] = None
    ) -> 'ClusteringModel':
        """
        Fit DBSCAN clustering model (density-based).
        
        Args:
            df: Input DataFrame with coordinates
            eps: Maximum distance between two samples for one to be considered in neighborhood
            min_samples: Minimum number of samples in a neighborhood for a point to be core
            features: Feature columns to use. Defaults to ['Latitude', 'Longitude']
            
        Returns:
            Self for method chaining
        """
        if features is None:
            features = ['Latitude', 'Longitude']
        
        self.feature_columns = features
        
        # Extract features
        X = df[features].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit DBSCAN
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels_ = self.model.fit_predict(X_scaled)
        
        # Calculate cluster centers (excluding noise points labeled as -1)
        self.n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        
        if self.n_clusters > 0:
            centers = []
            for i in range(self.n_clusters):
                mask = self.labels_ == i
                center = X[mask].mean(axis=0)
                centers.append(center)
            self.cluster_centers_ = np.array(centers)
        else:
            self.cluster_centers_ = None
        
        self.model_type = 'dbscan'
        
        noise_points = (self.labels_ == -1).sum()
        print(f"✓ DBSCAN clustering complete: {self.n_clusters} clusters identified, {noise_points} noise points")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            df: DataFrame with same features used for training
            
        Returns:
            Array of cluster labels
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit_kmeans or fit_dbscan first.")
        
        X = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'kmeans':
            return self.model.predict(X_scaled)
        else:
            # DBSCAN doesn't have predict method, use fit_predict
            return self.model.fit_predict(X_scaled)
    
    def add_cluster_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cluster labels to the original DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with 'Cluster' column added
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted yet")
        
        df_clustered = df.copy()
        df_clustered['Cluster'] = self.labels_
        
        return df_clustered
    
    def get_cluster_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistics for each cluster.
        
        Args:
            df: DataFrame with cluster labels
            
        Returns:
            DataFrame with cluster statistics
        """
        if 'Cluster' not in df.columns:
            df = self.add_cluster_labels(df)
        
        stats_list = []
        
        for cluster_id in sorted(df['Cluster'].unique()):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
            
            cluster_data = df[df['Cluster'] == cluster_id]
            
            stats = {
                'Cluster': cluster_id,
                'Size': len(cluster_data),
                'Percentage': f"{len(cluster_data) / len(df) * 100:.1f}%",
                'Center_Lat': cluster_data['Latitude'].mean() if 'Latitude' in cluster_data.columns else None,
                'Center_Lon': cluster_data['Longitude'].mean() if 'Longitude' in cluster_data.columns else None,
            }
            
            # Add demographic info if available
            if 'Gender' in cluster_data.columns:
                stats['Gender_Distribution'] = cluster_data['Gender'].value_counts().to_dict()
            
            if 'Age_Group' in cluster_data.columns:
                stats['Age_Distribution'] = cluster_data['Age_Group'].value_counts().to_dict()
            elif 'Age' in cluster_data.columns:
                stats['Avg_Age'] = cluster_data['Age'].mean()
            
            if 'Barangay District' in cluster_data.columns:
                top_location = cluster_data['Barangay District'].value_counts().index[0]
                stats['Top_Location'] = top_location
            
            stats_list.append(stats)
        
        return pd.DataFrame(stats_list)
    
    def evaluate_clustering(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate clustering quality using various metrics.
        
        Args:
            df: DataFrame with features used for clustering
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted yet")
        
        X = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Filter out noise points for DBSCAN
        mask = self.labels_ != -1
        X_filtered = X_scaled[mask]
        labels_filtered = self.labels_[mask]
        
        metrics = {}
        
        # Silhouette score (higher is better, range -1 to 1)
        if len(set(labels_filtered)) > 1:
            metrics['silhouette_score'] = silhouette_score(X_filtered, labels_filtered)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_filtered, labels_filtered)
        
        # Inertia (only for K-means)
        if self.model_type == 'kmeans':
            metrics['inertia'] = self.model.inertia_
        
        metrics['n_clusters'] = self.n_clusters
        metrics['n_samples'] = len(df)
        
        if self.model_type == 'dbscan':
            metrics['n_noise_points'] = (self.labels_ == -1).sum()
        
        return metrics
    
    def find_optimal_k(
        self,
        df: pd.DataFrame,
        k_range: Tuple[int, int] = (2, 10),
        features: Optional[List[str]] = None
    ) -> Dict[int, Dict[str, float]]:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            df: Input DataFrame
            k_range: Range of k values to test (min, max)
            features: Feature columns to use
            
        Returns:
            Dictionary mapping k to evaluation metrics
        """
        if features is None:
            features = ['Latitude', 'Longitude']
        
        X = df[features].values
        X_scaled = self.scaler.fit_transform(X)
        
        results = {}
        
        for k in range(k_range[0], k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            results[k] = {
                'inertia': kmeans.inertia_,
                'silhouette_score': silhouette_score(X_scaled, labels),
                'davies_bouldin_score': davies_bouldin_score(X_scaled, labels)
            }
        
        print(f"✓ Evaluated K-means for k={k_range[0]} to k={k_range[1]}")
        
        return results
    
    def identify_target_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify demographic patterns within each cluster.
        
        Args:
            df: DataFrame with cluster labels and demographic info
            
        Returns:
            DataFrame describing target groups for each cluster
        """
        if 'Cluster' not in df.columns:
            df = self.add_cluster_labels(df)
        
        target_groups = []
        
        for cluster_id in sorted(df['Cluster'].unique()):
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_data = df[df['Cluster'] == cluster_id]
            
            # Determine dominant characteristics
            group_profile = {
                'Cluster': cluster_id,
                'Size': len(cluster_data),
            }
            
            # Gender
            if 'Gender' in cluster_data.columns:
                gender_dist = cluster_data['Gender'].value_counts()
                dominant_gender = gender_dist.index[0]
                gender_pct = gender_dist.iloc[0] / len(cluster_data) * 100
                group_profile['Dominant_Gender'] = f"{dominant_gender} ({gender_pct:.0f}%)"
            
            # Age group
            if 'Age_Group' in cluster_data.columns:
                age_dist = cluster_data['Age_Group'].value_counts()
                dominant_age = age_dist.index[0]
                age_pct = age_dist.iloc[0] / len(cluster_data) * 100
                group_profile['Dominant_Age_Group'] = f"{dominant_age} ({age_pct:.0f}%)"
            
            # Create descriptive label
            description_parts = []
            if 'Dominant_Gender' in group_profile:
                description_parts.append(group_profile['Dominant_Gender'].split()[0])
            if 'Dominant_Age_Group' in group_profile:
                description_parts.append(group_profile['Dominant_Age_Group'].split('(')[0].strip())
            
            group_profile['Description'] = ' '.join(description_parts) if description_parts else f"Cluster {cluster_id}"
            
            target_groups.append(group_profile)
        
        return pd.DataFrame(target_groups)
