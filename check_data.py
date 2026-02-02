import os
import json
from glob import glob

# 三个文件夹的路径
paths = {
    "Arxiv": "datasets_dumps/arxiv",
    "News": "datasets_dumps/news",
    "Conversation": "datasets_dumps/conversation"
}

def check_folder(name, path):
    print(f"--- 正在检查 {name} ---")
    if not os.path.exists(path):
        print(f"? 错误: 文件夹不存在: {path}")
        return

    # 1. 检查文件是否存在
    if name == "Conversation":
        # 对话数据特殊检查：文件名必须匹配
        target = os.path.join(path, 'conversation_2k.json')
        if os.path.exists(target):
            files = [target]
        else:
            print(f"? 错误: 在 {path} 里没找到 'conversation_2k.json'")
            print("   (提示: 请检查文件名，或者修改代码里的文件名)")
            return
    else:
        # 其他两个检查任意 .json
        files = glob(os.path.join(path, "*.json"))
    
    if not files:
        print(f"? 错误: 文件夹里没有找到 JSON 文件")
        return
    
    print(f"? 发现 {len(files)} 个文件。")
    
    # 2. 检查文件内容格式 (读取第一个文件)
    try:
        with open(files[0], 'r', encoding='utf-8', errors='ignore') as f:
            # 兼容 JSONL (一行一个JSON) 和 普通 JSON List
            first_line = f.readline()
            try:
                data = json.loads(first_line)
            except:
                # 也许是整个文件是一个大 JSON List
                f.seek(0)
                full_data = json.load(f)
                data = full_data[0] if isinstance(full_data, list) else full_data

            print(f"   文件示例 Keys: {list(data.keys())}")
            
            # 3. 验证关键字段
            if name == "News":
                if 'content' in data or 'text' in data:
                    print("? 格式正确: 包含 content/text 字段")
                else:
                    print("?? 警告: 没找到 content 或 text 字段，代码可能会读到空内容！")
            
            elif name == "Arxiv":
                 if 'text' in data:
                    print("? 格式正确: 包含 text 字段")
                 else:
                    print("?? 警告: 没找到 text 字段")

    except Exception as e:
        print(f"? 读取失败: {e}")
    print("\n")

# 运行检查
for name, path in paths.items():
    check_folder(name, path)
