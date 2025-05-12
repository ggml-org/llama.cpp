#!/usr/bin/env python3

import os
import glob
import re
import sys
import json
import pandas as pd
from volcenginesdkarkruntime import Ark

def get_markdown_files(directory):
    """Get all markdown benchmark files in the specified directory."""
    # 尝试通过相对或绝对路径查找文件
    files = glob.glob(f"{directory}/prefill_decode_*.md")
    
    # 如果没有找到文件，检查是否需要添加repo根目录
    if not files:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        
        # 尝试几种可能的路径
        possible_paths = [
            os.path.join(repo_root, directory),
            os.path.join(script_dir, directory),
            directory
        ]
        
        for path in possible_paths:
            files = glob.glob(f"{path}/prefill_decode_CPU_*.md")
            if files:
                print(f"Found files in {path}")
                break
    
    return sorted(files)  # Sort files by name for consistent processing

def clean_json_response(text):
    """Clean up JSON response from LLM to ensure it's valid JSON."""
    # Remove markdown code fences if present
    if '```' in text:
        # Extract content between code fences
        match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # If we're here, either there were no code fences or the regex didn't match
    # Try to find where the JSON object starts and ends
    start_idx = text.find('{')
    if start_idx != -1:
        # Find the matching closing brace
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found the closing brace
                    return text[start_idx:i+1]
    
    # If we couldn't find valid JSON, return the original text
    return text

def extract_data_with_llm(markdown_files):
    """Use LLM to extract benchmark data from markdown files."""
    
    # Check for API key
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        print("Error: ARK_API_KEY environment variable not set.")
        print("Please set your API key with: export ARK_API_KEY='your_api_key'")
        sys.exit(1)
    
    # Initialize Ark client
    client = Ark(api_key=api_key)
    
    all_results = []
    
    for i, file_path in enumerate(markdown_files):
        print(f"Processing file {i+1}/{len(markdown_files)}: {file_path}")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Get timestamp from filename
        timestamp_match = re.search(r'CPU_(\d+)\.md', file_path)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
        else:
            timestamp = "unknown"
        
        # 从文件中提取模型名称
        model_name_match = re.search(r'Model: (.*)', content)
        if model_name_match:
            model_file_path = model_name_match.group(1).strip()
            model_name_from_file = os.path.basename(model_file_path)
            # 去除扩展名
            model_name_from_file = os.path.splitext(model_name_from_file)[0]
        else:
            model_name_from_file = "unknown"
        
        # Prepare prompt for LLM
        prompt = f"""
Extract structured data from this benchmark markdown file. For each prefill depth section in the markdown, extract the following fields from the table:
- model_name: The model name (e.g., "llama 8B Q8_0")
- model_size: The model size in GiB (e.g., "7.95 GiB")
- params: The number of parameters (e.g., "8.03 B")
- backend: The backend used (e.g., "Metal,BLAS")
- threads: Number of threads (e.g., "12")
- tokens_per_second: The performance in tokens per second (e.g., "12.44 ± 0.00")
- prefill_depth: The prefill depth from the section header (e.g., "1024", "2048", etc.)

Return a JSON object with this format:
{{
  "results": [
    {{
      "model_name": "...",
      "model_size": "...",
      "params": "...",
      "backend": "...",
      "threads": "...",
      "tokens_per_second": "...",
      "prefill_depth": "..."
    }},
    ...
  ]
}}

IMPORTANT: Return ONLY the raw JSON object with no additional text, markdown code blocks, or fences.

Markdown content:
{content}
"""
        
        # Call LLM to extract data
        try:
            completion = client.chat.completions.create(
                model="ep-m-20250510005507-ptq82",
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant. Extract structured data precisely from the provided text. Return only JSON with no markdown formatting."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse LLM response
            response_content = completion.choices[0].message.content
            
            # Clean and parse the JSON
            json_content = clean_json_response(response_content)
            
            # Debug the actual content before parsing
            print(f"First 100 chars of processed JSON: {json_content[:100]}...")
            
            # Try to parse the JSON
            try:
                results = json.loads(json_content)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Attempting manual parsing...")
                
                # Manual parsing as fallback
                if "results" in json_content and "model_name" in json_content and "prefill_depth" in json_content:
                    # Try to extract data using regex
                    model_matches = re.findall(r'"model_name":\s*"([^"]+)"', json_content)
                    size_matches = re.findall(r'"model_size":\s*"([^"]+)"', json_content)
                    params_matches = re.findall(r'"params":\s*"([^"]+)"', json_content)
                    backend_matches = re.findall(r'"backend":\s*"([^"]+)"', json_content)
                    threads_matches = re.findall(r'"threads":\s*"([^"]+)"', json_content)
                    tps_matches = re.findall(r'"tokens_per_second":\s*"([^"]+)"', json_content)
                    depth_matches = re.findall(r'"prefill_depth":\s*"([^"]+)"', json_content)
                    
                    # If we have matches for all fields and the same number of each
                    if (model_matches and size_matches and params_matches and backend_matches and 
                        threads_matches and tps_matches and depth_matches and
                        len(model_matches) == len(depth_matches)):
                        
                        # Construct results manually
                        results = {"results": []}
                        for i in range(len(model_matches)):
                            results["results"].append({
                                "model_name": model_matches[i],
                                "model_size": size_matches[i],
                                "params": params_matches[i],
                                "backend": backend_matches[i],
                                "threads": threads_matches[i],
                                "tokens_per_second": tps_matches[i],
                                "prefill_depth": depth_matches[i]
                            })
                        print(f"Manually parsed {len(results['results'])} results")
                    else:
                        raise Exception("Manual parsing failed - field count mismatch")
                else:
                    raise Exception("Manual parsing failed - required fields not found")
            
            # Add timestamp and file info to each result
            for result in results.get('results', []):
                result['timestamp'] = timestamp
                result['source_file'] = os.path.basename(file_path)
                
                # 添加从文件中提取的模型名
                result['model_file'] = model_name_from_file
                
                # Convert string values to appropriate types where possible
                try:
                    result['prefill_depth'] = int(result['prefill_depth'])
                    result['threads'] = int(result['threads'])
                    # Extract just the number from tokens_per_second
                    tps_match = re.search(r'(\d+\.\d+)', result['tokens_per_second'])
                    if tps_match:
                        result['tokens_per_second'] = float(tps_match.group(1))
                except ValueError:
                    pass
                
                all_results.append(result)
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            print(f"Response content: {response_content}")
    
    return all_results

def save_to_csv(results, output_file):
    """Save extracted results to CSV file."""
    if not results:
        print("No results to save.")
        return False
    
    df = pd.DataFrame(results)
    
    # 只保留指定的列
    keep_columns = [
        'model_name', 'model_file', 'prefill_depth', 'tokens_per_second', 
        'threads', 'backend', 'model_size', 'params'
    ]
    
    # 只保留存在于DataFrame中的列
    keep_columns = [col for col in keep_columns if col in df.columns]
    
    # 如果这些列中有不存在的，给出警告
    base_columns = ['model_name', 'prefill_depth', 'tokens_per_second', 
                  'threads', 'backend', 'model_size', 'params']
    missing_columns = [col for col in base_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: The following requested columns are missing: {', '.join(missing_columns)}")
    
    # 筛选列
    df = df[keep_columns]
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Also create a pivot table for easier comparison of different prefill depths
    try:
        pivot_df = df.pivot_table(
            index=['model_name', 'threads', 'backend', 'model_size', 'params'],
            columns='prefill_depth',
            values='tokens_per_second',
            aggfunc='mean'
        )
        
        # Rename columns for clarity
        pivot_df.columns = [f"depth_{col}_tps" for col in pivot_df.columns]
        
        pivot_file = output_file.replace('.csv', '_pivot.csv')
        pivot_df.to_csv(pivot_file)
        print(f"Pivot table saved to {pivot_file}")
    except Exception as e:
        print(f"Could not create pivot table: {str(e)}")
    
    return True

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Extract benchmark data from markdown files using LLM')
    parser.add_argument('--dir', default='bench_results', help='Directory containing benchmark markdown files')
    parser.add_argument('--output', default=None, help='Output CSV file path (defaults to <dir>/benchmark_summary.csv)')
    parser.add_argument('--test', action='store_true', help='Process only one file for testing')
    args = parser.parse_args()
    
    # 设置默认输出文件
    if args.output is None:
        args.output = os.path.join(args.dir, "benchmark_summary.csv")
    
    # Get all markdown benchmark files
    markdown_files = get_markdown_files(args.dir)
    if not markdown_files:
        print(f"No benchmark files found in {args.dir}")
        sys.exit(1)
    
    # For testing, use only one file
    if args.test and markdown_files:
        markdown_files = [markdown_files[0]]
        
    print(f"Found {len(markdown_files)} benchmark files.")
    
    # Extract data using LLM
    results = extract_data_with_llm(markdown_files)
    print(f"Extracted {len(results)} benchmark results.")
    
    # Save results to CSV
    success = save_to_csv(results, args.output)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 