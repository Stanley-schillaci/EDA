import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def coverage(clustered_data, total_data_length):
    cluster_sizes = clustered_data['cluster'].value_counts()
    return cluster_sizes / total_data_length

def diversity(clustered_data, column_name):
    return clustered_data.groupby('cluster')[column_name].std()

def diversity_with_cluster(clustered_data, column_name):
    return clustered_data.groupby('cluster')[column_name].agg(std='std', cluster_number='first')

def frequency_with_cluster(clustered_data, column_name):
    return clustered_data.groupby('cluster')[column_name].agg(mean='mean', cluster_number='first')

def frequency(clustered_data, column_name):
    return clustered_data.groupby('cluster')[column_name].mean()

def cluster_size(clustered_data):
    return clustered_data['cluster'].value_counts()

def compute_quality_metrics(user_features, df, coverage_col, diversity_col, frequency_col, user_features_scaled):
    sizes_cluster = cluster_size(user_features)
    coverage_cluster = coverage(user_features, len(df[coverage_col].unique()))
    diversity_cluster = diversity(user_features, diversity_col)
    frequency_cluster = frequency(user_features, frequency_col)

    cluster_diversity_mapping = dict(zip(diversity_cluster.index, diversity_cluster.values))
    cluster_frequency_mapping = dict(zip(frequency_cluster.index, frequency_cluster.values))

    metrics_df = pd.DataFrame({
        'Cluster_ID': sizes_cluster.index,
        'Cluster_Size': sizes_cluster.values
    })
    
    metrics_df['Coverage'] = coverage_cluster.values
    metrics_df['Diversity'] = metrics_df['Cluster_ID'].map(cluster_diversity_mapping)
    metrics_df['Frequency'] = metrics_df['Cluster_ID'].map(cluster_frequency_mapping)

    sil_score = silhouette_score(user_features_scaled, user_features['cluster'])
    return metrics_df, sil_score