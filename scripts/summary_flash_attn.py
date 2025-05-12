#!/usr/bin/env python3

import os
import re
import sys
import json
import pandas as pd
from volcenginesdkarkruntime import Ark

def extract_flash_attn_results(file_path):
    """Extract Flash Attention benchmark results from the output file."""
    results = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove ANSI color codes
    content = re.sub(r'\x1b\[[0-9;]*m', '', content)
    
    # Fix line breaks within benchmark lines
    content = re.sub(r'(\n\s+)', ' ', content)
    
    # Extract all benchmark lines
    pattern = r'FLASH_ATTN_EXT\((.*?)\):\s+(\d+) runs -\s+([\d.]+) us/run -\s+([\d.]+) MFLOP/run -\s+([\d.]+) GFLOPS'
    matches = re.findall(pattern, content)
    
    for match in matches:
        params_str, runs, us_per_run, mflop_per_run, gflops = match
        
        # Parse parameters
        param_dict = {}
        param_pattern = r'(\w+)=([^,\]]+)'
        param_matches = re.findall(param_pattern, params_str)
        
        for param_name, param_value in param_matches:
            # Convert numeric values
            try:
                if param_value.isdigit():
                    param_dict[param_name] = int(param_value)
                elif param_value.replace('.', '', 1).isdigit():
                    param_dict[param_name] = float(param_value)
                else:
                    param_dict[param_name] = param_value
            except ValueError:
                param_dict[param_name] = param_value
        
        # Extract permute values separately (they're in a list format)
        permute_match = re.search(r'permute=\[([\d,]+)\]', params_str)
        if permute_match:
            permute_str = permute_match.group(1)
            param_dict['permute'] = [int(x) for x in permute_str.split(',')]
        
        # Add performance metrics
        result = {
            **param_dict,
            'runs': int(runs),
            'us_per_run': float(us_per_run),
            'mflop_per_run': float(mflop_per_run),
            'gflops': float(gflops)
        }
        
        results.append(result)
    
    return results

def results_to_dataframe(results):
    """Convert extracted results to a pandas DataFrame."""
    df = pd.DataFrame(results)
    
    # Convert permute list to a string for easier display
    if 'permute' in df.columns:
        df['permute'] = df['permute'].apply(lambda x: str(x) if isinstance(x, list) else x)
    
    return df

def summarize_with_llm(df):
    """Use LLM to summarize performance patterns in the data."""
    # Check for API key
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        print("Error: ARK_API_KEY environment variable not set.")
        print("Please set your API key with: export ARK_API_KEY='your_api_key'")
        sys.exit(1)
    
    # Initialize Ark client
    client = Ark(api_key=api_key)
    
    # Create pivot tables for easier analysis
    pivot_by_type = pd.pivot_table(
        df, 
        values='gflops',
        index=['hsk', 'hsv', 'nr', 'kv'],
        columns=['type_KV'],
        aggfunc='mean'
    )
    
    pivot_by_dim = pd.pivot_table(
        df, 
        values='gflops',
        index=['type_KV', 'nr'],
        columns=['hsk', 'kv'],
        aggfunc='mean'
    )
    
    # Create a summary table showing performance for different configurations
    best_configs = df.sort_values('gflops', ascending=False).head(10)
    worst_configs = df.sort_values('gflops', ascending=True).head(5)
    
    # Create a comparison table for quantization types
    quant_comparison = pd.pivot_table(
        df,
        values='gflops',
        index=['hsk', 'nr', 'kv'],
        columns=['type_KV'],
        aggfunc='mean'
    ).reset_index()
    
    # Add comparison columns
    if 'f16' in quant_comparison.columns and 'q8_0' in quant_comparison.columns:
        quant_comparison['f16_vs_q8_ratio'] = quant_comparison['f16'] / quant_comparison['q8_0']
    
    if 'q8_0' in quant_comparison.columns and 'q4_0' in quant_comparison.columns:
        quant_comparison['q8_vs_q4_ratio'] = quant_comparison['q8_0'] / quant_comparison['q4_0']
    
    # Prepare prompt for LLM
    prompt = f"""
Analyze this FLASH_ATTN_EXT benchmark data and create a summary of performance patterns. 

The key parameters in the data are:
- hsk: Key head size (dimensionality of keys)
- hsv: Value head size (dimensionality of values)
- nh: Number of heads
- nr: Repeat factor (for grouped-query attention)
- kv: KV sequence length (context length for the keys and values)
- nb: Batch size
- type_KV: Data type used for K and V matrices (f16, q8_0, q4_0)
- gflops: Performance in GFLOPS (higher is better)

Pivot table by quantization type:
{pivot_by_type.to_string()}

Pivot table by dimensions:
{pivot_by_dim.to_string()}

Top 10 performing configurations:
{best_configs[['hsk', 'nr', 'kv', 'type_KV', 'gflops']].to_string()}

Bottom 5 performing configurations:
{worst_configs[['hsk', 'nr', 'kv', 'type_KV', 'gflops']].to_string()}

Quantization comparison:
{quant_comparison.to_string()}

Please provide:
1. A comprehensive analysis of how different parameters affect performance
2. Observations about quantization impact (f16 vs q8_0 vs q4_0)
3. Insights about how head size, context length, and nr (grouped-query factor) affect throughput
4. Recommendations for optimal configurations based on the data
5. A detailed comparison table showing performance across different configurations

Format your response as markdown with tables where appropriate.
"""
    
    # Call LLM for analysis
    try:
        completion = client.chat.completions.create(
            model="ep-m-20250510005507-ptq82",
            messages=[
                {"role": "system", "content": "You are a performance analysis expert specializing in ML acceleration. Analyze benchmark data and provide clear, insightful summaries with quantitative comparisons."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Get the LLM response
        summary = completion.choices[0].message.content
        return summary
        
    except Exception as e:
        print(f"Error calling LLM API: {str(e)}")
        return "Failed to generate summary with LLM."

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Analyze Flash Attention benchmark results')
    parser.add_argument('--input', default='flash_attn_benchmark.txt', help='Path to benchmark output file')
    # parser.add_argument('--output', default='flash_attn_summary.md', help='Output markdown file for the summary')
    parser.add_argument('--csv', default='flash_attn_results.csv', help='Output CSV file for the raw results')
    args = parser.parse_args()
    
    # Extract results
    results = extract_flash_attn_results(args.input)
    if not results:
        print(f"No benchmark results found in {args.input}")
        sys.exit(1)
    
    print(f"Extracted {len(results)} benchmark results.")
    
    # Convert to DataFrame
    df = results_to_dataframe(results)
    
    # Save raw results to CSV
    df.to_csv(args.csv, index=False)
    print(f"Raw results saved to {args.csv}")
    
    # # Generate summary with LLM
    # summary = summarize_with_llm(df)
    
    # # Save summary to file
    # with open(args.output, 'w') as f:
    #     f.write(summary)
    
    # print(f"Summary saved to {args.output}")
    # print("Summary:")
    # print("=" * 40)
    # print(summary)

if __name__ == "__main__":
    main() 