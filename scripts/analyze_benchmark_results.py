#!/usr/bin/env python3

import os
import glob
import pandas as pd
import re
import argparse

def extract_model_name(filename):
    """Extract model name from the file path"""
    # 从文件名中提取模型名称
    match = re.search(r'([^/]+)_[qf][0-9]+_[0-9]+\.csv$', filename)
    if match:
        return match.group(1)
    return "unknown"

def extract_model_params(row):
    """Extract model parameters in billions from model_n_params column"""
    if 'model_n_params' in row:
        # Convert parameters from string to numeric and then to billions
        try:
            return float(row['model_n_params']) / 1e9
        except (ValueError, TypeError):
            return None
    return None

def process_csv_files(directory):
    """Process all CSV files in the given directory"""
    # 获取目录下所有CSV文件
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # 创建空DataFrame来存储所有数据
    all_data = pd.DataFrame()
    
    # 处理每个CSV文件
    for file_path in csv_files:
        print(f"Processing {file_path}")
        
        # 从文件名中提取KV类型
        kv_type_match = re.search(r'prefill_decode_([^_]+)_', os.path.basename(file_path))
        kv_type = kv_type_match.group(1) if kv_type_match else "unknown"
        
        # 读取CSV
        try:
            df = pd.read_csv(file_path)
            
            # 添加额外的列
            df['file_name'] = os.path.basename(file_path)
            df['kv_type'] = kv_type
            df['model_name'] = extract_model_name(file_path)
            
            # 添加模型参数量（B）列
            df['model_params_B'] = df.apply(extract_model_params, axis=1)
            
            # 确保类型正确
            if 'n_gen' in df.columns:
                df['n_gen'] = pd.to_numeric(df['n_gen'], errors='coerce')
            if 'n_depth' in df.columns:
                df['n_depth'] = pd.to_numeric(df['n_depth'], errors='coerce')
            if 'avg_ts' in df.columns:
                df['avg_ts'] = pd.to_numeric(df['avg_ts'], errors='coerce')
            
            # 合并到主数据框
            all_data = pd.concat([all_data, df], ignore_index=True)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if all_data.empty:
        print("No data found in CSV files")
        return
    
    # 基于n_gen字段拆分为prefill和decode数据
    # prefill通常没有生成tokens (n_gen = 0)
    # decode通常有生成tokens (n_gen > 0)
    prefill_data = all_data[all_data['n_gen'] == 0].copy()
    decode_data = all_data[all_data['n_gen'] > 0].copy()
    
    # 生成交叉表分析 - Prefill数据
    if not prefill_data.empty:
        try:
            print("Generating prefill pivot table...")
            
            # 添加K缓存和V缓存组合列
            prefill_data['k_cache'] = prefill_data['type_k'].astype(str)
            prefill_data['v_cache'] = prefill_data['type_v'].astype(str)
            # 创建缓存类型的组合键，用于排序
            prefill_data['cache_key'] = prefill_data['k_cache'] + '_' + prefill_data['v_cache']
            
            # 创建模型名称和KV类型的交叉表，显示prefill性能
            pivot_prefill = pd.pivot_table(
                prefill_data, 
                values='avg_ts',
                index=['model_name', 'n_depth', 'n_prompt', 'model_params_B', 'k_cache', 'v_cache'],
                aggfunc='mean'
            )
            
            # 重置索引，将索引列变成常规列
            pivot_prefill_reset = pivot_prefill.reset_index()
            
            # 按cache_key排序，保证相同类型的缓存在一起
            pivot_prefill_reset['cache_key'] = pivot_prefill_reset['k_cache'] + '_' + pivot_prefill_reset['v_cache']
            pivot_prefill_reset = pivot_prefill_reset.sort_values(by=['cache_key', 'n_depth'])
            pivot_prefill_reset = pivot_prefill_reset.drop(columns=['cache_key'])  # 删除辅助排序列
            
            # 保存到文件
            output_path = os.path.join(directory, "prefill_performance_pivot.csv")
            pivot_prefill_reset.to_csv(output_path, index=False)
            print(f"Prefill pivot table saved to {output_path}")
            
            # 额外创建按深度分组的透视表
            prefill_depth_data = prefill_data.copy()
            
            # 按缓存类型分组
            cache_groups = []
            for cache_type, group in prefill_depth_data.groupby(['k_cache', 'v_cache']):
                k_type, v_type = cache_type
                
                # 为每种缓存类型创建透视表
                depth_pivot = pd.pivot_table(
                    group,
                    values='avg_ts',
                    index=['model_name', 'model_params_B'],
                    columns=['n_depth'],
                    aggfunc='mean'
                )
                
                # 重命名列以便更清晰
                depth_pivot.columns = [f'depth_{col}_tps' for col in depth_pivot.columns]
                
                # 添加缓存类型列
                depth_pivot = depth_pivot.reset_index()
                depth_pivot['k_cache'] = k_type
                depth_pivot['v_cache'] = v_type
                
                cache_groups.append(depth_pivot)
            
            # 合并所有缓存类型结果
            if cache_groups:
                combined_depth_pivot = pd.concat(cache_groups)
                # 调整列顺序，确保缓存类型在前面
                cols = combined_depth_pivot.columns.tolist()
                depth_cols = [col for col in cols if col.startswith('depth_')]
                other_cols = [col for col in cols if not col.startswith('depth_')]
                final_cols = ['model_name', 'model_params_B', 'k_cache', 'v_cache'] + [col for col in other_cols if col not in ['model_name', 'model_params_B', 'k_cache', 'v_cache']] + depth_cols
                combined_depth_pivot = combined_depth_pivot[final_cols]
                
                # 按缓存类型排序
                combined_depth_pivot['cache_key'] = combined_depth_pivot['k_cache'] + '_' + combined_depth_pivot['v_cache']
                combined_depth_pivot = combined_depth_pivot.sort_values(by=['cache_key'])
                combined_depth_pivot = combined_depth_pivot.drop(columns=['cache_key'])  # 删除辅助排序列
                
                # 保存到文件
                depth_output = os.path.join(directory, "prefill_by_depth_pivot.csv")
                combined_depth_pivot.to_csv(depth_output, index=False)
                print(f"Prefill by depth pivot table saved to {depth_output}")
            
        except Exception as e:
            print(f"Error creating prefill pivot table: {e}")
    
    # 生成交叉表分析 - Decode数据
    if not decode_data.empty:
        try:
            print("Generating decode pivot table...")
            
            # 添加K缓存和V缓存组合列
            decode_data['k_cache'] = decode_data['type_k'].astype(str)
            decode_data['v_cache'] = decode_data['type_v'].astype(str)
            # 创建缓存类型的组合键，用于排序
            decode_data['cache_key'] = decode_data['k_cache'] + '_' + decode_data['v_cache']
            
            # 创建模型名称和KV类型的交叉表，显示decode性能
            pivot_decode = pd.pivot_table(
                decode_data, 
                values='avg_ts',
                index=['model_name', 'n_depth', 'n_prompt', 'model_params_B', 'n_gen', 'k_cache', 'v_cache'],
                aggfunc='mean'
            )
            
            # 重置索引，将索引列变成常规列
            pivot_decode_reset = pivot_decode.reset_index()
            
            # 按cache_key排序，保证相同类型的缓存在一起
            pivot_decode_reset['cache_key'] = pivot_decode_reset['k_cache'] + '_' + pivot_decode_reset['v_cache']
            pivot_decode_reset = pivot_decode_reset.sort_values(by=['cache_key', 'n_depth'])
            pivot_decode_reset = pivot_decode_reset.drop(columns=['cache_key'])  # 删除辅助排序列
            
            # 保存到文件
            output_path = os.path.join(directory, "decode_performance_pivot.csv")
            pivot_decode_reset.to_csv(output_path, index=False)
            print(f"Decode pivot table saved to {output_path}")
            
            # 额外创建按深度分组的透视表
            decode_depth_data = decode_data.copy()
            
            # 按缓存类型分组
            cache_groups = []
            for cache_type, group in decode_depth_data.groupby(['k_cache', 'v_cache']):
                k_type, v_type = cache_type
                
                # 为每种缓存类型创建透视表
                depth_pivot = pd.pivot_table(
                    group,
                    values='avg_ts',
                    index=['model_name', 'model_params_B'],
                    columns=['n_depth'],
                    aggfunc='mean'
                )
                
                # 重命名列以便更清晰
                depth_pivot.columns = [f'depth_{col}_tps' for col in depth_pivot.columns]
                
                # 添加缓存类型列
                depth_pivot = depth_pivot.reset_index()
                depth_pivot['k_cache'] = k_type
                depth_pivot['v_cache'] = v_type
                
                cache_groups.append(depth_pivot)
            
            # 合并所有缓存类型结果
            if cache_groups:
                combined_depth_pivot = pd.concat(cache_groups)
                # 调整列顺序，确保缓存类型在前面
                cols = combined_depth_pivot.columns.tolist()
                depth_cols = [col for col in cols if col.startswith('depth_')]
                other_cols = [col for col in cols if not col.startswith('depth_')]
                final_cols = ['model_name', 'model_params_B', 'k_cache', 'v_cache'] + [col for col in other_cols if col not in ['model_name', 'model_params_B', 'k_cache', 'v_cache']] + depth_cols
                combined_depth_pivot = combined_depth_pivot[final_cols]
                
                # 按缓存类型排序
                combined_depth_pivot['cache_key'] = combined_depth_pivot['k_cache'] + '_' + combined_depth_pivot['v_cache']
                combined_depth_pivot = combined_depth_pivot.sort_values(by=['cache_key'])
                combined_depth_pivot = combined_depth_pivot.drop(columns=['cache_key'])  # 删除辅助排序列
                
                # 保存到文件
                depth_output = os.path.join(directory, "decode_by_depth_pivot.csv")
                combined_depth_pivot.to_csv(depth_output, index=False)
                print(f"Decode by depth pivot table saved to {depth_output}")
            
        except Exception as e:
            print(f"Error creating decode pivot table: {e}")
    
    print("Processing complete!")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Process benchmark CSV files')
    parser.add_argument('--dir', required=True, help='Directory containing benchmark CSV files')
    args = parser.parse_args()
    
    # 处理CSV文件
    process_csv_files(args.dir)

if __name__ == "__main__":
    main() 