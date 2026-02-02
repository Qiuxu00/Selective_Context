# -*- coding: utf-8 -*-
import os
import json
from datasets import load_dataset

# 1. 设置镜像加速 (非常重要，防止超时)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("🚀 正在尝试从镜像站下载 BBC News 数据集...")

try:
    # 2. 下载数据集 (只下载训练集)
    ds = load_dataset('liyucheng/bbc_new_2303', split='train')
    
    # 3. 目标路径
    target_dir = "datasets_dumps/news"
    output_path = os.path.join(target_dir, "bbc_real_data.json")
    
    # 确保文件夹存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(f"📥 数据集加载成功，共 {len(ds)} 条数据。")
    print(f"💾 正在转换为标准 JSON 列表并保存到 {output_path} ...")

    # === 核心修改点 ===
    # 不使用 ds.to_json()，而是先转为 Python 列表，再用 json.dump 写入
    # 这样能保证生成的是标准的 [{}, {}, ...] 格式
    data_list = list(ds) 
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # ensure_ascii=False 保证中文正常显示（虽然BBC是英文，是个好习惯）
        # indent=4 让文件排版更漂亮，方便你查看
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    # ================
    
    print("✅ 下载并保存成功！格式已修正为标准 JSON 列表。")
    print("👉 现在你可以运行 main.py 了。")

except Exception as e:
    print(f"❌ 下载或保存失败: {e}")