#!/usr/bin/env python3

import re
import sys
import os

def fix_tensor_data_in_file(filepath):
    """Fix tensor->data references in a file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Fix simple data access patterns (but not assignments)
        # Pattern: something->data (but not = something->data)
        content = re.sub(r'(\w+)->data(?!\s*=)', r'tensor_data(\1)', content)
        
        # Fix assignments: tensor->data = value -> tensor_set_data(tensor, value)
        content = re.sub(r'(\w+)->data\s*=\s*([^;]+);', r'tensor_set_data(\1, \2);', content)
        
        # Fix GGML_ASSERT patterns
        content = re.sub(r'GGML_ASSERT\(tensor_data\(([^)]+)\)\s*!=\s*NULL', r'GGML_ASSERT(tensor_data(\1) != NULL', content)
        content = re.sub(r'GGML_ASSERT\(tensor_data\(([^)]+)\)\s*==\s*NULL', r'GGML_ASSERT(tensor_data(\1) == NULL', content)
        content = re.sub(r'GGML_ASSERT\(tensor_data\(([^)]+)\)', r'GGML_ASSERT(tensor_data(\1)', content)
        
        # Fix memcpy patterns
        content = re.sub(r'memcpy\(tensor_data\(([^)]+)\),', r'memcpy(tensor_data(\1),', content)
        content = re.sub(r'memcpy\(([^,]+),\s*tensor_data\(([^)]+)\),', r'memcpy(\1, tensor_data(\2),', content)
        
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True
        else:
            print(f"No changes: {filepath}")
            return False
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python fix_tensor_data.py <file_or_directory>")
        sys.exit(1)
    
    target = sys.argv[1]
    
    if os.path.isfile(target):
        fix_tensor_data_in_file(target)
    elif os.path.isdir(target):
        for root, dirs, files in os.walk(target):
            for file in files:
                if file.endswith(('.c', '.cpp', '.h', '.hpp')):
                    filepath = os.path.join(root, file)
                    fix_tensor_data_in_file(filepath)
    else:
        print(f"Error: {target} is not a valid file or directory")
        sys.exit(1)

if __name__ == "__main__":
    main()