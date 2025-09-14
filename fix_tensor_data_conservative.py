#!/usr/bin/env python3

import re
import sys
import os

def fix_tensor_data_in_file(filepath):
    """Fix tensor->data references in a file, but only for actual tensor variables"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # More conservative approach - only fix patterns where we're confident it's a tensor
        # Look for common tensor variable names and patterns
        
        # Fix: tensor->data -> tensor_data(tensor) 
        content = re.sub(r'\btensor->data\b(?!\s*=)', r'tensor_data(tensor)', content)
        content = re.sub(r'\bsrc->data\b(?!\s*=)', r'tensor_data(src)', content)
        content = re.sub(r'\bdst->data\b(?!\s*=)', r'tensor_data(dst)', content)
        content = re.sub(r'\bsrc0->data\b(?!\s*=)', r'tensor_data(src0)', content)
        content = re.sub(r'\bsrc1->data\b(?!\s*=)', r'tensor_data(src1)', content)
        content = re.sub(r'\bnode->data\b(?!\s*=)', r'tensor_data(node)', content)
        content = re.sub(r'\bt->data\b(?!\s*=)', r'tensor_data(t)', content)
        content = re.sub(r'\bleaf->data\b(?!\s*=)', r'tensor_data(leaf)', content)
        content = re.sub(r'\bview_src->data\b(?!\s*=)', r'tensor_data(view_src)', content)
        content = re.sub(r'\bgrad_acc->data\b(?!\s*=)', r'tensor_data(grad_acc)', content)
        content = re.sub(r'\binput->data\b(?!\s*=)', r'tensor_data(input)', content)
        content = re.sub(r'\bparent->data\b(?!\s*=)', r'tensor_data(parent)', content)
        content = re.sub(r'\bids->data\b(?!\s*=)', r'tensor_data(ids)', content)
        
        # Fix assignments: tensor->data = value -> tensor_set_data(tensor, value)
        content = re.sub(r'\btensor->data\s*=\s*([^;]+);', r'tensor_set_data(tensor, \1);', content)
        content = re.sub(r'\bsrc->data\s*=\s*([^;]+);', r'tensor_set_data(src, \1);', content)
        content = re.sub(r'\bdst->data\s*=\s*([^;]+);', r'tensor_set_data(dst, \1);', content)
        content = re.sub(r'\bnode->data\s*=\s*([^;]+);', r'tensor_set_data(node, \1);', content)
        content = re.sub(r'\bt->data\s*=\s*([^;]+);', r'tensor_set_data(t, \1);', content)
        content = re.sub(r'\bnew_tensor->data\s*=\s*([^;]+);', r'tensor_set_data(new_tensor, \1);', content)
        
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