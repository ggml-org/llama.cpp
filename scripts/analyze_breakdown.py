#!/usr/bin/env python3
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import argparse
from collections import defaultdict

def parse_csv_file(file_path):
    """Parse a breakdown CSV file and return a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        
        # Filter out anomalous data (sync_nsec = 0 and unreasonably large proc_nsec)
        anomalies = df[(df['sync_nsec'] == 0) & (df['proc_nsec'] > 1e12)].index
        if len(anomalies) > 0:
            print(f"Filtered out {len(anomalies)} anomalous data points with sync_nsec=0")
            print(f"Anomalies: {df.loc[anomalies].to_string()}")
            df = df.drop(anomalies)
        
        # Convert nanoseconds to milliseconds
        for col in ['proc_nsec', 'sync_nsec', 'total_nsec']:
            if col in df.columns:
                df[f'{col}_ms'] = df[col] / 1_000_000
        
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def analyze_operators(df):
    """Group and sum operators by type."""
    if df is None or df.empty:
        return None
    
    # Group by op_name and sum the time values
    op_summary = df.groupby('op_name').agg({
        'proc_nsec_ms': 'sum',
        'sync_nsec_ms': 'sum', 
        'total_nsec_ms': 'sum'
    }).reset_index()
    
    # Sort by total time in descending order
    op_summary = op_summary.sort_values('total_nsec_ms', ascending=False)
    
    return op_summary

def visualize_breakdown(op_summary, output_path, title):
    """Create and save breakdown visualization."""
    if op_summary is None or op_summary.empty:
        print(f"No data to visualize for {title}")
        return
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot proc time and sync time as stacked bars
    ax = op_summary.plot(
        kind='bar', 
        x='op_name', 
        y=['proc_nsec_ms', 'sync_nsec_ms'],
        stacked=True,
        color=['#3498db', '#e74c3c'],
        title=f"Operator Breakdown - {title}"
    )
    
    # Customize plot
    plt.xlabel('Operator')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(['Processing Time', 'Sync Time'])
    plt.tight_layout()
    
    # Add total time values on top of each bar
    for i, (_, row) in enumerate(op_summary.iterrows()):
        plt.text(
            i, 
            row['total_nsec_ms'] + 0.5, 
            f"{row['total_nsec_ms']:.1f}",
            ha='center', 
            va='bottom', 
            rotation=0, 
            size=8
        )
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")

def visualize_ops_pie(op_summary, output_path, title, top_n=10):
    """Create a pie chart of the top N operators."""
    if op_summary is None or op_summary.empty:
        return
    
    # Take top N operators
    top_ops = op_summary.head(top_n).copy()
    
    # Add "Others" category for the rest
    if len(op_summary) > top_n:
        others_sum = op_summary.iloc[top_n:]['total_nsec_ms'].sum()
        others = pd.DataFrame({
            'op_name': ['Others'],
            'proc_nsec_ms': [0],  # We won't show breakdown for Others
            'sync_nsec_ms': [0],
            'total_nsec_ms': [others_sum]
        })
        top_ops = pd.concat([top_ops, others])
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.pie(
        top_ops['total_nsec_ms'], 
        labels=top_ops['op_name'], 
        autopct='%1.1f%%',
        startangle=90,
        shadow=False,
    )
    plt.axis('equal')
    plt.title(f"Top {top_n} Operators - {title}")
    
    # Save the figure
    pie_path = output_path.replace('.png', '_pie.png')
    plt.savefig(pie_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Pie chart saved to {pie_path}")

def get_file_type_from_path(file_path):
    """Extract type (prefill/decode) from file path."""
    base_name = os.path.basename(file_path)
    if 'prefill_' in base_name:
        return 'prefill'
    elif 'decode_' in base_name:
        return 'decode'
    else:
        # Try to identify from file content or context
        return 'breakdown'

def get_depth_from_path(file_path):
    """Extract depth from file path."""
    base_name = os.path.basename(file_path)
    
    # Handle different naming patterns
    if '_' in base_name:
        # Try to find a number after an underscore and before another underscore or period
        parts = base_name.split('_')
        for part in parts:
            if part.isdigit():
                return part
    
    # Default if no depth found
    return "unknown"

def process_file(file_path):
    """Process a single CSV file and generate visualizations."""
    print(f"Processing {file_path}...")
    
    # Get file type and depth
    file_type = get_file_type_from_path(file_path)
    depth = get_depth_from_path(file_path)
    
    # Parse CSV
    df = parse_csv_file(file_path)
    if df is None:
        return
    
    # Analyze operators
    op_summary = analyze_operators(df)
    
    # Create output path
    output_dir = os.path.dirname(file_path)
    base_name = os.path.basename(file_path).replace('.csv', '')
    output_path = os.path.join(output_dir, f"{base_name}_breakdown.png")
    
    # Visualize
    title = f"{file_type.title()} (Depth {depth})"
    visualize_breakdown(op_summary, output_path, title)
    visualize_ops_pie(op_summary, output_path, title)
    
    # Also generate a text summary
    summary_path = output_path.replace('.png', '.txt')
    with open(summary_path, 'w') as f:
        total_time = op_summary['total_nsec_ms'].sum()
        f.write(f"Operator Breakdown - {title}\n")
        f.write(f"Total time: {total_time:.2f} ms\n\n")
        f.write(f"{'Operator':<20} {'Processing (ms)':<15} {'Sync (ms)':<15} {'Total (ms)':<15} {'Percentage':<10}\n")
        f.write('-' * 80 + '\n')
        
        for _, row in op_summary.iterrows():
            percentage = (row['total_nsec_ms'] / total_time) * 100
            f.write(f"{row['op_name']:<20} {row['proc_nsec_ms']:<15.2f} {row['sync_nsec_ms']:<15.2f} "
                   f"{row['total_nsec_ms']:<15.2f} {percentage:<10.2f}%\n")
    
    return op_summary

def main():
    parser = argparse.ArgumentParser(description='Analyze operator breakdown from CSV files')
    parser.add_argument('--dir', help='Directory containing CSV files to analyze', default=None)
    parser.add_argument('--file', help='Specific CSV file to analyze', default=None)
    parser.add_argument('--compare', help='Generate comparison charts across depths', action='store_true')
    args = parser.parse_args()
    
    files_to_process = []
    
    if args.file:
        files_to_process = [args.file]
    elif args.dir:
        files_to_process = glob.glob(os.path.join(args.dir, '*.csv'))
    else:
        # Try to find CSV files in current directory
        files_to_process = glob.glob('*.csv')
        if not files_to_process:
            print("No CSV files found. Please specify a file or directory.")
            return
    
    # Process all files
    summaries = {}
    for file_path in files_to_process:
        summary = process_file(file_path)
        if summary is not None:
            file_type = get_file_type_from_path(file_path)
            depth = get_depth_from_path(file_path)
            key = f"{file_type}_{depth}"
            summaries[key] = summary
    
    # Generate comparison charts if requested
    if args.compare and len(summaries) > 1:
        compare_across_depths(summaries, os.path.dirname(files_to_process[0]))

def compare_across_depths(summaries, output_dir):
    """Generate comparison charts across different depths."""
    # Group by file type (prefill/decode)
    prefill_summaries = {k.split('_')[1]: v for k, v in summaries.items() if k.startswith('prefill_')}
    decode_summaries = {k.split('_')[1]: v for k, v in summaries.items() if k.startswith('decode_')}
    
    # Compare prefill
    if prefill_summaries:
        compare_operator_times(prefill_summaries, output_dir, 'prefill')
    
    # Compare decode
    if decode_summaries:
        compare_operator_times(decode_summaries, output_dir, 'decode')

def compare_operator_times(summaries_by_depth, output_dir, file_type):
    """Create charts comparing operator times across depths."""
    if not summaries_by_depth:
        return
    
    # Get all unique operators across all depths
    all_ops = set()
    for summary in summaries_by_depth.values():
        all_ops.update(summary['op_name'].tolist())
    
    # Create a DataFrame for comparison
    compare_data = {}
    depths = []
    
    for depth, summary in sorted(summaries_by_depth.items(), key=lambda x: int(x[0]) if x[0].isdigit() else float('inf')):
        depths.append(depth)
        
        # Create mapping of op_name to total_time for this depth
        op_times = {}
        for _, row in summary.iterrows():
            op_times[row['op_name']] = row['total_nsec_ms']
            
        # Add to compare data
        compare_data[depth] = op_times
    
    # Convert to DataFrame with ops as rows and depths as columns
    compare_df = pd.DataFrame(index=sorted(all_ops))
    
    for depth in depths:
        compare_df[depth] = compare_df.index.map(lambda op: compare_data[depth].get(op, 0))
    
    # Sort by average time across all depths
    compare_df['avg'] = compare_df.mean(axis=1)
    compare_df = compare_df.sort_values('avg', ascending=False)
    compare_df = compare_df.drop('avg', axis=1)
    
    # Take top 10 ops
    top_ops = compare_df.head(10)
    
    # Plot stacked bar chart
    plt.figure(figsize=(14, 10))
    top_ops.T.plot(kind='bar', stacked=True, figsize=(14, 10))
    plt.title(f'{file_type.title()} Time Comparison Across Different Depths')
    plt.xlabel('Depth')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45)
    plt.legend(title='Operator', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{file_type}_depth_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison chart saved to {output_path}")
    
    # Also create a line chart showing how total time increases with depth
    plt.figure(figsize=(10, 6))
    total_times = [compare_df[depth].sum() for depth in depths]
    
    # Convert depths to integers if possible
    try:
        x_vals = [int(d) for d in depths]
    except ValueError:
        x_vals = list(range(len(depths)))
        plt.xticks(x_vals, depths)
    
    plt.plot(x_vals, total_times, marker='o', linestyle='-', linewidth=2)
    plt.title(f'{file_type.title()} Total Time vs Depth')
    plt.xlabel('Depth')
    plt.ylabel('Total Time (ms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add data labels
    for i, (x, y) in enumerate(zip(x_vals, total_times)):
        plt.text(x, y + max(total_times)*0.02, f"{y:.1f}", ha='center')
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{file_type}_total_time_by_depth.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Total time chart saved to {output_path}")

if __name__ == "__main__":
    main() 