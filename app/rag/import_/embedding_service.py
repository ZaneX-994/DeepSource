import json
from pathlib import Path
from typing import Any

from app.infra.llm.providers import llm_provider
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import EMBEDDING_BATCH_SIZE
from app.shared.runtime.logger import logger, step_log

@step_log("get_and_validate")
def get_and_validate(state: ImportGraphState) -> tuple[list[dict[str, Any]], str]:

    # 1.获取相关参数
    md_path = state.get("md_path")
    item_name = state.get("item_name")
    chunks = state.get("chunks")

    # 2.参数校验
    if not item_name:
        if md_path and Path(md_path).exists():
            item_name = Path(md_path).stem
        else:
            item_name = "default_item_name"
        logger.warning(f"没有item_name，给予默认值{item_name}")

    if not chunks:
        if md_path:
            # 获取备份json数据
            chunks_json_obj = Path(md_path).with_name(f"{Path(md_path).stem}.json")
            chunks = json.loads(chunks_json_obj.read_text(encoding="utf-8"))
        if not chunks:
            logger.error(f"chunks为空，读取本地备份json文件后依旧为空，业务无法继续！")
            raise ValueError(f"chunks为空，读取本地备份json文件后依旧为空，业务无法继续！")

    return chunks, item_name

@step_log("batch_generate_embeddings")
def batch_generate_embeddings(chunks: list[dict[str, Any]], item_name: str, batch_size: int=EMBEDDING_BATCH_SIZE) -> list[dict[str, Any]]:

    logger.info(f"未生成向量之前的示例: {chunks[0]}")


    # 1.获取chunks长度
    length = len(chunks)

    # 2.循环遍历
    for index in range(0, length, batch_size):
        batch = chunks[index:index + batch_size]
        # 拼接item_name + content
        current_content_list = [ f"主体: {item_name}, 内容: {chunk.get("content")}" for chunk in batch ]
        # 批量生成向量
        result = llm_provider.generate_embeddings(current_content_list)
        # 回填数据
        for idx, chunk in enumerate(batch):
            chunk["dense_vector"] = result.get("dense")[idx]
            chunk["sparse_vector"] = result.get("sparse")[idx]

    logger.info(f"生成向量之后的示例: {chunks[0]}")


    return chunks






@step_log("generate_chunk_embeddings")
def generate_chunk_embeddings(state: ImportGraphState) -> ImportGraphState:
    """
    向量化服务：
    1. 读取 chunks
    2. 生成 dense_vector / sparse_vector
    3. 将向量结果补充回 chunks
    """

    # 1.获取并校验参数
    chunks, item_name = get_and_validate(state)

    # 2.批量生产embeddings
    embedding_content = batch_generate_embeddings(chunks, item_name)

    # 3.更新state
    state["embedding_content"] = embedding_content

    return state