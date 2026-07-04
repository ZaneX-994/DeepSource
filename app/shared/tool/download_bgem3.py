"""
工具脚本，用于处理 download bgem3 相关的辅助任务。
"""
import os
from modelscope.hub.snapshot_download import snapshot_download
from dotenv import load_dotenv


load_dotenv()

# 下载模型到当前目录下的 models/bge-m3 文件夹
model_dir = snapshot_download('BAAI/bge-m3', cache_dir=os.getenv("BGE_M3_PATH"))
print(f"模型已下载到: {model_dir}")




