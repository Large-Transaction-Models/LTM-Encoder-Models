#!/usr/bin/env python3
"""
Converted from present_results.Rmd
Generates survival prediction model performance heatmaps across feature sets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pickle
import os
import pyreadr

def load_rds_file(filepath):
    """
    Load RDS file using pyreadr
    """
    try:
        if filepath.endswith('.rds'):
            result = pyreadr.read_r(filepath)
            # pyreadr returns a dict with keys as table names
            # Usually the first (and only) key contains the data
            if result:
                key = list(result.keys())[0]
                return result[key]
            else:
                print(f"No data found in {filepath}")
                return None
        else:
            return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_benchmark_results():
    """Load benchmark results from CSV"""
    filepath = "../Results/Survival_Prediction/originalBenchmark/final_results.csv"
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # Remove unwanted columns
        columns_to_remove = ['GBM', 'DeepHit', 'DeepSurv']
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Transform Cox values: 1 - original_value
        if 'Cox' in df.columns:
            df['Cox'] = 1 - df['Cox']
        
        # Add features column
        df['features'] = 'benchmark'
        
        # Clean outcome event names
        df['outcomeEvent'] = df['outcomeEvent'].replace('account liquidated', 'liquidated')
        
        # Create dataset column
        df['dataset'] = df['indexEvent'] + '-' + df['outcomeEvent']
        
        # Remove unwanted columns
        df = df.drop(columns=['indexEvent', 'outcomeEvent'])
        
        # Add mean row
        mean_row = df.groupby('features').agg({
            'XGBoost': 'mean',
            'Cox': 'mean', 
            'AFT': 'mean'
        }).reset_index()
        mean_row['dataset'] = 'mean'
        
        df = pd.concat([df, mean_row], ignore_index=True)
        
        return df
    else:
        print(f"File not found: {filepath}")
        return None

def load_raw_results():
    """Load raw results from RDS file"""
    filepath = "../Results/Survival_Prediction/raw_seqLen10/results.rds"
    if os.path.exists(filepath):
        df = load_rds_file(filepath)
        if df is not None:
            # Add features column
            df['features'] = 'raw'
            
            # Clean outcome event names
            df['outcomeEvent'] = df['outcomeEvent'].replace('Account Liquidated', 'liquidated')
            df['outcomeEvent'] = df['outcomeEvent'].str.lower()
            
            # Create dataset column
            df['dataset'] = df['indexEvent'].str.lower() + '-' + df['outcomeEvent']
            
            # Remove unwanted columns
            df = df.drop(columns=['indexEvent', 'outcomeEvent'])
            
            # Clean model names (remove suffixes)
            df['model'] = df['model'].str.replace('_.*$', '', regex=True)
            
            # Transform Cox values: 1 - original_value (before pivoting)
            df.loc[df['model'] == 'Cox', 'c_index'] = 1 - df.loc[df['model'] == 'Cox', 'c_index']
            
            # Pivot to wide format
            df = df.pivot_table(index=['dataset', 'features'], 
                              columns='model', 
                              values='c_index', 
                              aggfunc='mean').reset_index()
            
            # Add mean row
            mean_row = df.groupby('features').agg({
                'XGBoost': 'mean',
                'Cox': 'mean', 
                'AFT': 'mean'
            }).reset_index()
            mean_row['dataset'] = 'mean'
            
            df = pd.concat([df, mean_row], ignore_index=True)
            
            return df
        else:
            print(f"Failed to load RDS file: {filepath}")
            return None
    else:
        print(f"File not found: {filepath}")
        return None

def load_ltm_results():
    """Load LTM results from RDS file"""
    filepath = "../Results/Survival_Prediction/ltm_seqLen10/results.rds"
    if os.path.exists(filepath):
        df = load_rds_file(filepath)
        if df is not None:
            # Add features column
            df['features'] = 'LTM'
            
            # Clean outcome event names
            df['outcomeEvent'] = df['outcomeEvent'].replace('Account Liquidated', 'liquidated')
            df['outcomeEvent'] = df['outcomeEvent'].str.lower()
            
            # Create dataset column
            df['dataset'] = df['indexEvent'].str.lower() + '-' + df['outcomeEvent']
            
            # Remove unwanted columns
            df = df.drop(columns=['indexEvent', 'outcomeEvent'])
            
            # Clean model names (remove suffixes)
            df['model'] = df['model'].str.replace('_.*$', '', regex=True)
            
            # Transform Cox values: 1 - original_value (before pivoting)
            df.loc[df['model'] == 'Cox', 'c_index'] = 1 - df.loc[df['model'] == 'Cox', 'c_index']
            
            # Pivot to wide format
            df = df.pivot_table(index=['dataset', 'features'], 
                              columns='model', 
                              values='c_index', 
                              aggfunc='mean').reset_index()
            
            # Add mean row
            mean_row = df.groupby('features').agg({
                'XGBoost': 'mean',
                'Cox': 'mean', 
                'AFT': 'mean'
            }).reset_index()
            mean_row['dataset'] = 'mean'
            
            df = pd.concat([df, mean_row], ignore_index=True)
            
            return df
        else:
            print(f"Failed to load RDS file: {filepath}")
            return None
    else:
        print(f"File not found: {filepath}")
        return None

def create_heatmap(data, title, filename, metric_name="C-index"):
    """Create and save heatmap"""
    
    # Convert to long format
    df_long = pd.melt(data, 
                     id_vars=['dataset', 'features'], 
                     value_vars=['XGBoost', 'AFT', 'Cox'],
                     var_name='model', 
                     value_name=metric_name.lower())
    
    # Set factor levels (order)
    df_long['model'] = pd.Categorical(df_long['model'], categories=['XGBoost', 'AFT', 'Cox'])
    
    # Set dataset order with mean at the end
    unique_datasets = df_long['dataset'].unique()
    datasets_without_mean = [d for d in unique_datasets if d != 'mean']
    dataset_order = datasets_without_mean + ['mean']
    df_long['dataset'] = pd.Categorical(df_long['dataset'], categories=dataset_order)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Create pivot table for heatmap
    pivot_data = df_long.pivot_table(values=metric_name.lower(), 
                                   index='dataset', 
                                   columns=['model', 'features'], 
                                   aggfunc='mean')
    
    # Create custom colormap (RdYlBu equivalent) - REVERSED for proper color mapping
    # Blue for low values, red/orange for high values
    colors = ['#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', 
              '#fee090', '#fdae61', '#f46d43', '#d73027']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('RdYlBu_reversed', colors, N=n_bins)
    
    # Create heatmap
    sns.heatmap(pivot_data, 
                annot=True, 
                fmt='.2f',
                cmap=cmap,
                cbar_kws={'label': metric_name},
                linewidths=0.5,
                ax=ax)
    
    # Customize plot
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('', fontsize=12)  # Remove x-label since we have spanning headers
    ax.set_ylabel('Dataset', fontsize=12)
    
    # Rotate x-axis labels (sub-column labels)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to make room for headers
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """Main function to generate heatmaps"""
    print("Loading data...")
    
    # Load all results
    benchmark_results = load_benchmark_results()
    raw_results = load_raw_results()
    ltm_results = load_ltm_results()
    
    if benchmark_results is not None:
        print("✓ Benchmark results loaded")
    else:
        print("✗ Failed to load benchmark results")
    
    if raw_results is not None:
        print("✓ Raw results loaded")
    else:
        print("✗ Failed to load raw results")
        
    if ltm_results is not None:
        print("✓ LTM results loaded")
    else:
        print("✗ Failed to load LTM results")
    
    # Combine all results
    if all(df is not None for df in [benchmark_results, raw_results, ltm_results]):
        all_results = pd.concat([benchmark_results, raw_results, ltm_results], ignore_index=True)
        
        print("Generating survival prediction heatmap...")
        fig = create_heatmap(all_results, 
                           "Survival Prediction Model Performance Across Feature Sets",
                           "prediction_results_corrected.pdf",
                           "C-index")
        print("✓ Heatmap saved as 'prediction_results_corrected.pdf'")
        
    else:
        print("Error: Could not load all required data files")
        print("Note: RDS files need to be converted to CSV format for Python processing")
        print("You can use R to export them:")
        print("  readRDS('file.rds') %>% write.csv('file.csv', row.names=FALSE)")

if __name__ == "__main__":
    main()
