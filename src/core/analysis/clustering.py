"""
Clustering analysis module for identifying hotspot patterns.
Implements K-means and DBSCAN for spatial clustering with comprehensive evaluation.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns


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
    
    def find_optimal_dbscan_params(
        self,
        df: pd.DataFrame,
        eps_range: List[float] = None,
        min_samples_range: List[int] = None,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Find optimal DBSCAN parameters using grid search.
        
        Args:
            df: Input DataFrame
            eps_range: List of epsilon values to test
            min_samples_range: List of min_samples values to test
            features: Feature columns to use
            
        Returns:
            DataFrame with evaluation results for each parameter combination
        """
        if features is None:
            features = ['Latitude', 'Longitude']
        
        if eps_range is None:
            eps_range = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
        
        if min_samples_range is None:
            min_samples_range = [3, 5, 7, 10]
        
        X = df[features].values
        X_scaled = self.scaler.fit_transform(X)
        
        results = []
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X_scaled)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = (labels == -1).sum()
                
                result = {
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_percentage': (n_noise / len(labels)) * 100
                }
                
                # Calculate metrics only if we have valid clusters
                if n_clusters > 1 and n_noise < len(labels):
                    mask = labels != -1
                    X_filtered = X_scaled[mask]
                    labels_filtered = labels[mask]
                    
                    if len(set(labels_filtered)) > 1:
                        result['silhouette_score'] = silhouette_score(X_filtered, labels_filtered)
                        result['davies_bouldin_score'] = davies_bouldin_score(X_filtered, labels_filtered)
                    else:
                        result['silhouette_score'] = None
                        result['davies_bouldin_score'] = None
                else:
                    result['silhouette_score'] = None
                    result['davies_bouldin_score'] = None
                
                results.append(result)
        
        results_df = pd.DataFrame(results)
        print(f"✓ Evaluated {len(results)} DBSCAN parameter combinations")
        
        return results_df
    
    def compare_clustering_methods(
        self,
        df: pd.DataFrame,
        k: int = 5,
        eps: float = 0.01,
        min_samples: int = 5,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare K-Means and DBSCAN clustering performance.
        
        Args:
            df: Input DataFrame
            k: Number of clusters for K-Means
            eps: Epsilon parameter for DBSCAN
            min_samples: Min samples parameter for DBSCAN
            features: Feature columns to use
            
        Returns:
            DataFrame comparing both methods
        """
        if features is None:
            features = ['Latitude', 'Longitude']
        
        comparison = []
        
        # K-Means evaluation
        kmeans_model = ClusteringModel()
        kmeans_model.fit_kmeans(df, n_clusters=k, features=features)
        kmeans_metrics = kmeans_model.evaluate_clustering(df)
        
        comparison.append({
            'Method': 'K-Means',
            'n_clusters': kmeans_metrics['n_clusters'],
            'silhouette_score': kmeans_metrics.get('silhouette_score', None),
            'davies_bouldin_score': kmeans_metrics.get('davies_bouldin_score', None),
            'inertia': kmeans_metrics.get('inertia', None),
            'noise_points': 0,
            'parameters': f'k={k}'
        })
        
        # DBSCAN evaluation
        dbscan_model = ClusteringModel()
        dbscan_model.fit_dbscan(df, eps=eps, min_samples=min_samples, features=features)
        dbscan_metrics = dbscan_model.evaluate_clustering(df)
        
        comparison.append({
            'Method': 'DBSCAN',
            'n_clusters': dbscan_metrics['n_clusters'],
            'silhouette_score': dbscan_metrics.get('silhouette_score', None),
            'davies_bouldin_score': dbscan_metrics.get('davies_bouldin_score', None),
            'inertia': None,
            'noise_points': dbscan_metrics.get('n_noise_points', 0),
            'parameters': f'eps={eps}, min_samples={min_samples}'
        })
        
        comparison_df = pd.DataFrame(comparison)
        print("✓ Clustering methods comparison complete")
        
        return comparison_df
    
    def evaluate_cluster_quality(
        self,
        df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Comprehensive cluster quality evaluation with multiple metrics.
        
        Args:
            df: DataFrame with features used for clustering
            
        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted yet")
        
        X = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        # Filter out noise points
        mask = self.labels_ != -1
        X_filtered = X_scaled[mask]
        labels_filtered = self.labels_[mask]
        
        evaluation = {
            'model_type': self.model_type,
            'n_clusters': self.n_clusters,
            'n_samples': len(df),
            'n_noise_points': (self.labels_ == -1).sum() if self.model_type == 'dbscan' else 0
        }
        
        # Internal validation metrics
        if len(set(labels_filtered)) > 1:
            evaluation['silhouette_score'] = silhouette_score(X_filtered, labels_filtered)
            evaluation['davies_bouldin_score'] = davies_bouldin_score(X_filtered, labels_filtered)
            
            # Calculate per-cluster silhouette scores
            from sklearn.metrics import silhouette_samples
            silhouette_vals = silhouette_samples(X_filtered, labels_filtered)
            
            cluster_silhouettes = {}
            for cluster_id in set(labels_filtered):
                cluster_silhouettes[int(cluster_id)] = float(np.mean(silhouette_vals[labels_filtered == cluster_id]))
            
            evaluation['cluster_silhouette_scores'] = cluster_silhouettes
        
        # Cluster size distribution
        unique, counts = np.unique(labels_filtered, return_counts=True)
        evaluation['cluster_sizes'] = {int(u): int(c) for u, c in zip(unique, counts)}
        
        # Cluster balance metric (coefficient of variation)
        if len(counts) > 1:
            evaluation['cluster_balance'] = float(np.std(counts) / np.mean(counts))
        
        # Inertia for K-Means
        if self.model_type == 'kmeans':
            evaluation['inertia'] = float(self.model.inertia_)
        
        # Geographic spread of clusters
        if 'Latitude' in self.feature_columns and 'Longitude' in self.feature_columns:
            lat_idx = self.feature_columns.index('Latitude')
            lon_idx = self.feature_columns.index('Longitude')
            
            cluster_spreads = {}
            for cluster_id in set(labels_filtered):
                cluster_points = X[mask][labels_filtered == cluster_id]
                lat_std = np.std(cluster_points[:, lat_idx])
                lon_std = np.std(cluster_points[:, lon_idx])
                cluster_spreads[int(cluster_id)] = {
                    'lat_std': float(lat_std),
                    'lon_std': float(lon_std),
                    'avg_spread': float((lat_std + lon_std) / 2)
                }
            
            evaluation['cluster_geographic_spread'] = cluster_spreads
        
        return evaluation


# =============================================================================
# Helper Functions for Quick Evaluation
# =============================================================================

def run_kmeans(
    df: pd.DataFrame,
    n_clusters: int = 5,
    features: Optional[List[str]] = None
) -> Dict:
    """
    Quick wrapper to run K-Means clustering and get results.
    
    Args:
        df: Input DataFrame
        n_clusters: Number of clusters
        features: Feature columns to use
        
    Returns:
        Dictionary with model, labels, centers, and metrics
    """
    model = ClusteringModel()
    model.fit_kmeans(df, n_clusters=n_clusters, features=features)
    metrics = model.evaluate_clustering(df)
    
    return {
        'model': model,
        'labels': model.labels_,
        'centers': model.cluster_centers_,
        'n_clusters': model.n_clusters,
        'silhouette_score': metrics.get('silhouette_score'),
        'davies_bouldin_score': metrics.get('davies_bouldin_score'),
        'inertia': metrics.get('inertia')
    }


def run_dbscan(
    df: pd.DataFrame,
    eps: float = 0.01,
    min_samples: int = 5,
    features: Optional[List[str]] = None
) -> Dict:
    """
    Quick wrapper to run DBSCAN clustering and get results.
    
    Args:
        df: Input DataFrame
        eps: Epsilon parameter
        min_samples: Min samples parameter
        features: Feature columns to use
        
    Returns:
        Dictionary with model, labels, centers, and metrics
    """
    model = ClusteringModel()
    model.fit_dbscan(df, eps=eps, min_samples=min_samples, features=features)
    metrics = model.evaluate_clustering(df)
    
    return {
        'model': model,
        'labels': model.labels_,
        'centers': model.cluster_centers_,
        'n_clusters': model.n_clusters,
        'n_noise_points': metrics.get('n_noise_points', 0),
        'silhouette_score': metrics.get('silhouette_score'),
        'davies_bouldin_score': metrics.get('davies_bouldin_score')
    }


def find_optimal_clusters(
    df: pd.DataFrame,
    k_range: Tuple[int, int] = (2, 15),
    features: Optional[List[str]] = None,
    plot: bool = True
) -> pd.DataFrame:
    """
    Find optimal number of clusters using multiple metrics.
    
    Args:
        df: Input DataFrame
        k_range: Range of k values to test
        features: Feature columns to use
        plot: Whether to plot results
        
    Returns:
        DataFrame with evaluation results for each k
    """
    model = ClusteringModel()
    results = model.find_optimal_k(df, k_range=k_range, features=features)
    
    # Convert to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.index.name = 'k'
    results_df.reset_index(inplace=True)
    
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Elbow plot
        axes[0].plot(results_df['k'], results_df['inertia'], 'bo-')
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette score
        axes[1].plot(results_df['k'], results_df['silhouette_score'], 'go-')
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Score (Higher is Better)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Davies-Bouldin score
        axes[2].plot(results_df['k'], results_df['davies_bouldin_score'], 'ro-')
        axes[2].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[2].set_ylabel('Davies-Bouldin Score', fontsize=12)
        axes[2].set_title('Davies-Bouldin Score (Lower is Better)', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/outputs/optimal_k_evaluation.png', dpi=300, bbox_inches='tight')
        print("✓ Plot saved to data/outputs/optimal_k_evaluation.png")
        plt.close()
    
    # Recommend optimal k
    optimal_k_silhouette = results_df.loc[results_df['silhouette_score'].idxmax(), 'k']
    optimal_k_db = results_df.loc[results_df['davies_bouldin_score'].idxmin(), 'k']
    
    print(f"\n📊 Optimal k recommendations:")
    print(f"   - Silhouette Score: k={optimal_k_silhouette} (score: {results_df.loc[results_df['k']==optimal_k_silhouette, 'silhouette_score'].values[0]:.3f})")
    print(f"   - Davies-Bouldin: k={optimal_k_db} (score: {results_df.loc[results_df['k']==optimal_k_db, 'davies_bouldin_score'].values[0]:.3f})")
    
    return results_df


def find_optimal_dbscan(
    df: pd.DataFrame,
    eps_range: List[float] = None,
    min_samples_range: List[int] = None,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Find optimal DBSCAN parameters.
    
    Args:
        df: Input DataFrame
        eps_range: List of epsilon values to test
        min_samples_range: List of min_samples values to test
        features: Feature columns to use
        
    Returns:
        DataFrame with evaluation results sorted by quality
    """
    model = ClusteringModel()
    results = model.find_optimal_dbscan_params(
        df, 
        eps_range=eps_range,
        min_samples_range=min_samples_range,
        features=features
    )
    
    # Filter out results with no clusters or too much noise
    valid_results = results[
        (results['n_clusters'] > 1) & 
        (results['noise_percentage'] < 50)
    ].copy()
    
    if len(valid_results) > 0:
        # Sort by silhouette score (higher is better)
        valid_results_sorted = valid_results.sort_values('silhouette_score', ascending=False)
        
        print(f"\n📊 Top 5 DBSCAN parameter combinations:")
        print(valid_results_sorted.head(5).to_string(index=False))
        
        best = valid_results_sorted.iloc[0]
        print(f"\n✅ Recommended DBSCAN parameters:")
        print(f"   - eps: {best['eps']}")
        print(f"   - min_samples: {best['min_samples']}")
        print(f"   - n_clusters: {best['n_clusters']}")
        print(f"   - silhouette_score: {best['silhouette_score']:.3f}")
    else:
        print("⚠️ No valid DBSCAN configurations found. Try adjusting parameter ranges.")
    
    return results


def compare_all_methods(
    df: pd.DataFrame,
    optimal_k: int = None,
    eps: float = None,
    min_samples: int = None,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare K-Means and DBSCAN with optimal parameters.
    
    Args:
        df: Input DataFrame
        optimal_k: Optimal k for K-Means (if None, uses k=5)
        eps: Optimal eps for DBSCAN (if None, uses eps=0.01)
        min_samples: Optimal min_samples for DBSCAN (if None, uses min_samples=5)
        features: Feature columns to use
        
    Returns:
        DataFrame comparing both methods
    """
    if optimal_k is None:
        optimal_k = 5
    if eps is None:
        eps = 0.01
    if min_samples is None:
        min_samples = 5
    
    model = ClusteringModel()
    comparison = model.compare_clustering_methods(
        df,
        k=optimal_k,
        eps=eps,
        min_samples=min_samples,
        features=features
    )
    
    print("\n📊 Clustering Methods Comparison:")
    print(comparison.to_string(index=False))
    
    return comparison

