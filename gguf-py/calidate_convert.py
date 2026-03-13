from gguf import GGUFReader

def validate_gguf_file(filepath):
    """验证GGUF文件结构是否正确"""
    try:
        reader = GGUFReader(filepath)
        print("文件头信息:")
        print(f"魔数: {reader.fields.get('magic', '未知')}")
        print(f"版本: {reader.fields.get('version', '未知')}")
        print(f"张量数量: {len(reader.tensors)}")
        
        # 检查每个张量的偏移量
        for tensor in reader.tensors:
            print(f"张量: {tensor.name}, 偏移量: {tensor.data_offset}")
            
        return True
    except Exception as e:
        print(f"文件验证失败: {e}")
        return False

# 使用验证函数
validate_gguf_file("ifairy.gguf")