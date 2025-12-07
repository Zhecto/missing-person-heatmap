"""
Comprehensive Clustering Evaluation Script
Evaluates K-Means and DBSCAN to find optimal parameters and compare performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Add project paths
sys.path.append(str(Path(__file__).parent / "src"))

from core.ingestion.data_loader import DataLoader
from core.preprocessing.data_cleaner import DataCleaner
from core.analysis.clustering import (
    find_optimal_clusters,
    find_optimal_dbscan,
    compare_all_methods,
    run_kmeans,
    run_dbscan
)


def main():
    """Run comprehensive clustering evaluation."""
    
    print("=" * 80)
    print("CLUSTERING EVALUATION - Missing Person Hotspot Analysis")
    print("=" * 80)
    
    # Load data
    print("\n📥 Loading data...")
    loader = DataLoader()
    
    # Try to load uploaded data first, fall back to sample data
    try:
        df = loader.load_csv('data/uploaded_data.csv')
        print("✓ Loaded uploaded_data.csv")
    except:
        try:
            df = loader.load_csv('data/sample_data.csv')
            print("✓ Loaded sample_data.csv")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return
    
    print(f"   - Records: {len(df)}")
    print(f"   - Columns: {list(df.columns)}")
    
    # Preprocess data
    print("\n🧹 Preprocessing data...")
    cleaner = DataCleaner()
    df_clean = cleaner.preprocess_pipeline(df)
    
    # Remove any rows with missing coordinates
    df_clean = df_clean.dropna(subset=['Latitude', 'Longitude'])
    print(f"✓ Clean records with coordinates: {len(df_clean)}")
    
    # Define features for clustering
    features = ['Latitude', 'Longitude']
    
    # =============================================================================
    # PART 1: Find Optimal K for K-Means
    # =============================================================================
    print("\n" + "=" * 80)
    print("PART 1: K-MEANS OPTIMAL CLUSTER EVALUATION")
    print("=" * 80)
    
    k_results = find_optimal_clusters(
        df_clean,
        k_range=(2, 15),
        features=features,
        plot=True
    )
    
    # Save K-Means evaluation results
    k_results.to_csv('data/outputs/kmeans_evaluation.csv', index=False)
    print("✓ K-Means evaluation saved to data/outputs/kmeans_evaluation.csv")
    
    # Get recommended k
    optimal_k = int(k_results.loc[k_results['silhouette_score'].idxmax(), 'k'])
    print(f"\n✅ Selected optimal k: {optimal_k}")
    
    # =============================================================================
    # PART 2: Find Optimal Parameters for DBSCAN
    # =============================================================================
    print("\n" + "=" * 80)
    print("PART 2: DBSCAN OPTIMAL PARAMETER EVALUATION")
    print("=" * 80)
    
    dbscan_results = find_optimal_dbscan(
        df_clean,
        eps_range=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15],
        min_samples_range=[3, 5, 7, 10],
        features=features
    )
    
    # Save DBSCAN evaluation results
    dbscan_results.to_csv('data/outputs/dbscan_evaluation.csv', index=False)
    print("✓ DBSCAN evaluation saved to data/outputs/dbscan_evaluation.csv")
    
    # Get recommended DBSCAN parameters
    valid_dbscan = dbscan_results[
        (dbscan_results['n_clusters'] > 1) & 
        (dbscan_results['noise_percentage'] < 50) &
        (dbscan_results['silhouette_score'].notna())
    ]
    
    if len(valid_dbscan) > 0:
        best_dbscan = valid_dbscan.sort_values('silhouette_score', ascending=False).iloc[0]
        optimal_eps = best_dbscan['eps']
        optimal_min_samples = int(best_dbscan['min_samples'])
        print(f"\n✅ Selected DBSCAN parameters: eps={optimal_eps}, min_samples={optimal_min_samples}")
    else:
        # Try to find any configuration with clusters, even with higher noise
        any_clusters = dbscan_results[
            (dbscan_results['n_clusters'] > 0) &
            (dbscan_results['noise_percentage'] < 80)
        ]
        if len(any_clusters) > 0:
            best_dbscan = any_clusters.sort_values('n_clusters', ascending=False).iloc[0]
            optimal_eps = best_dbscan['eps']
            optimal_min_samples = int(best_dbscan['min_samples'])
            print(f"\n⚠️ DBSCAN found clusters but with high noise: eps={optimal_eps}, min_samples={optimal_min_samples}")
            print(f"   (Noise: {best_dbscan['noise_percentage']:.1f}%, Clusters: {best_dbscan['n_clusters']})")
        else:
            optimal_eps = 0.05
            optimal_min_samples = 3
            print(f"\n⚠️ No valid DBSCAN configuration found. Using fallback: eps={optimal_eps}, min_samples={optimal_min_samples}")
    
    # =============================================================================
    # PART 3: Compare Methods with Optimal Parameters
    # =============================================================================
    print("\n" + "=" * 80)
    print("PART 3: CLUSTERING METHODS COMPARISON")
    print("=" * 80)
    
    comparison = compare_all_methods(
        df_clean,
        optimal_k=optimal_k,
        eps=optimal_eps,
        min_samples=optimal_min_samples,
        features=features
    )
    
    # Save comparison results
    comparison.to_csv('data/outputs/clustering_comparison.csv', index=False)
    print("✓ Comparison saved to data/outputs/clustering_comparison.csv")
    
    # =============================================================================
    # PART 4: Detailed Evaluation of Best Model
    # =============================================================================
    print("\n" + "=" * 80)
    print("PART 4: DETAILED EVALUATION OF BEST MODEL")
    print("=" * 80)
    
    # Run both models with optimal parameters
    print("\n🔍 Running K-Means with optimal k...")
    kmeans_result = run_kmeans(df_clean, n_clusters=optimal_k, features=features)
    
    print("\n🔍 Running DBSCAN with optimal parameters...")
    dbscan_result = run_dbscan(
        df_clean,
        eps=optimal_eps,
        min_samples=optimal_min_samples,
        features=features
    )
    
    # Get comprehensive quality evaluation
    print("\n📊 K-Means Comprehensive Evaluation:")
    kmeans_quality = kmeans_result['model'].evaluate_cluster_quality(df_clean)
    print(json.dumps(kmeans_quality, indent=2, default=str))
    
    print("\n📊 DBSCAN Comprehensive Evaluation:")
    dbscan_quality = dbscan_result['model'].evaluate_cluster_quality(df_clean)
    print(json.dumps(dbscan_quality, indent=2, default=str))
    
    # Save detailed evaluations
    with open('data/outputs/kmeans_detailed_evaluation.json', 'w') as f:
        json.dump(kmeans_quality, f, indent=2, default=str)
    
    with open('data/outputs/dbscan_detailed_evaluation.json', 'w') as f:
        json.dump(dbscan_quality, f, indent=2, default=str)
    
    print("\n✓ Detailed evaluations saved to data/outputs/")
    
    # =============================================================================
    # PART 5: Get Cluster Statistics
    # =============================================================================
    print("\n" + "=" * 80)
    print("PART 5: CLUSTER STATISTICS")
    print("=" * 80)
    
    # K-Means statistics
    print("\n📊 K-Means Cluster Statistics:")
    df_kmeans = kmeans_result['model'].add_cluster_labels(df_clean)
    kmeans_stats = kmeans_result['model'].get_cluster_statistics(df_kmeans)
    print(kmeans_stats.to_string(index=False))
    kmeans_stats.to_csv('data/outputs/kmeans_cluster_stats.csv', index=False)
    
    # DBSCAN statistics
    print("\n📊 DBSCAN Cluster Statistics:")
    df_dbscan = dbscan_result['model'].add_cluster_labels(df_clean)
    dbscan_stats = dbscan_result['model'].get_cluster_statistics(df_dbscan)
    print(dbscan_stats.to_string(index=False))
    dbscan_stats.to_csv('data/outputs/dbscan_cluster_stats.csv', index=False)
    
    # =============================================================================
    # PART 6: Final Recommendations
    # =============================================================================
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)
    
    # Determine best method
    kmeans_silhouette = kmeans_result['silhouette_score']
    dbscan_silhouette = dbscan_result['silhouette_score']
    
    print(f"\n🎯 Model Performance Summary:")
    print(f"   K-Means (k={optimal_k}):")
    print(f"     - Silhouette Score: {kmeans_silhouette:.3f}" if kmeans_silhouette else "     - Silhouette Score: N/A")
    print(f"     - Davies-Bouldin: {kmeans_result['davies_bouldin_score']:.3f}" if kmeans_result['davies_bouldin_score'] else "     - Davies-Bouldin: N/A")
    print(f"     - Inertia: {kmeans_result['inertia']:.2f}" if kmeans_result['inertia'] else "     - Inertia: N/A")
    print(f"     - Clusters: {kmeans_result['n_clusters']}")
    
    print(f"\n   DBSCAN (eps={optimal_eps}, min_samples={optimal_min_samples}):")
    print(f"     - Silhouette Score: {dbscan_silhouette:.3f}" if dbscan_silhouette else "     - Silhouette Score: N/A")
    print(f"     - Davies-Bouldin: {dbscan_result['davies_bouldin_score']:.3f}" if dbscan_result['davies_bouldin_score'] else "     - Davies-Bouldin: N/A")
    print(f"     - Clusters: {dbscan_result['n_clusters']}")
    print(f"     - Noise Points: {dbscan_result['n_noise_points']}")
    
    if kmeans_silhouette is not None and dbscan_silhouette is not None:
        if kmeans_silhouette > dbscan_silhouette:
            print(f"\n✅ RECOMMENDED: K-Means with k={optimal_k}")
            print(f"   Reason: Higher silhouette score ({kmeans_silhouette:.3f} vs {dbscan_silhouette:.3f})")
        else:
            print(f"\n✅ RECOMMENDED: DBSCAN with eps={optimal_eps}, min_samples={optimal_min_samples}")
            print(f"   Reason: Higher silhouette score ({dbscan_silhouette:.3f} vs {kmeans_silhouette:.3f})")
    elif kmeans_silhouette is not None:
        print(f"\n✅ RECOMMENDED: K-Means with k={optimal_k}")
        print(f"   Reason: DBSCAN failed to produce valid clusters for this dataset")
        print(f"   Note: Geographic data may be too sparse for density-based clustering")
    else:
        print(f"\n⚠️ Both methods encountered issues. Manual review recommended.")
    
    print("\n💡 Usage Recommendations:")
    print("   - K-Means: Best for identifying fixed number of geographic regions")
    print("   - DBSCAN: Best for finding natural density-based clusters")
    print("   - For thesis: Use K-Means k=5 for consistent, interpretable results")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE! All results saved to data/outputs/")
    print("=" * 80)
    
    # Create summary file
    summary = {
        'evaluation_date': pd.Timestamp.now().isoformat(),
        'dataset_records': len(df_clean),
        'optimal_kmeans': {
            'k': optimal_k,
            'silhouette_score': float(kmeans_silhouette) if kmeans_silhouette else None,
            'davies_bouldin_score': float(kmeans_result['davies_bouldin_score']) if kmeans_result['davies_bouldin_score'] else None,
            'inertia': float(kmeans_result['inertia'])
        },
        'optimal_dbscan': {
            'eps': float(optimal_eps),
            'min_samples': int(optimal_min_samples),
            'n_clusters': int(dbscan_result['n_clusters']),
            'silhouette_score': float(dbscan_silhouette) if dbscan_silhouette else None,
            'davies_bouldin_score': float(dbscan_result['davies_bouldin_score']) if dbscan_result['davies_bouldin_score'] else None,
            'n_noise_points': int(dbscan_result['n_noise_points'])
        }
    }
    
    with open('data/outputs/evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✓ Summary saved to data/outputs/evaluation_summary.json")


if __name__ == "__main__":
    main()
