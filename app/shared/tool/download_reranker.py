"""
工具脚本，用于处理 download reranker 相关的辅助任务。
"""
import os

from modelscope.hub.snapshot_download import snapshot_download
from dotenv import load_dotenv


load_dotenv()

local_dir = os.getenv("BGE_RERANKER_LARGE")

snapshot_download(
    model_id="BAAI/bge-reranker-large",
    cache_dir=local_dir,
)

print("下载完成，模型目录：", local_dir)