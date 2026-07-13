from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger
from app.infra.llm.providers import llm_provider
from app.infra.vector_store.milvus_gateway import milvus_gateway

def validate_and_get_data(state: QueryGraphState) -> tuple[list[str], str]:

    item_names = state.get("item_names", [])
    rewritten_query = state.get("rewritten_query", "")

    if len(item_names) == 0 or not rewritten_query:
        logger.error("关联主体或重写的问题为空，业务无法继续！")
        raise ValueError("关联主体或重写的问题为空，业务无法继续！")

    return item_names, rewritten_query


def search_by_milvus(item_names: list[str], rewritten_query: str) -> list:

    query_embedding = llm_provider.generate_embeddings([rewritten_query])
    reqs = milvus_gateway.create_request(
        dense_vector=query_embedding['dense'][0],
        sparse_vector=query_embedding['sparse'][0],
        expr=f"item_name in {item_names}",
        limit=5,
    )

    milvus_result = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name,
        reqs=reqs,
        ranker_weights=(0.5, 0.5),
        norm_score=True,
        limit=5,
        output_fields=[
            "chunk_id",
            "file_title",
            "item_name",
            "title",
            "parent_title",
            "part",
            "content",
        ]
    )

    return milvus_result[0] if milvus_result and len(milvus_result) > 0 else []


def deal_milvus_list(milvus_list: list):
    embedding_chunks = []

    for item in milvus_list:
        entity = item.get("entity", {})
        embedding_chunks.append({
            "id": entity.get("chunk_id"),
            "score": item.get("score", 0.0),
            "file_title": entity.get("file_title", ""),
            "item_name": entity.get("item_name", ""),
            "title": entity.get("title", ""),
            "parent_title": entity.get("parent_title", ""),
            "part": entity.get("part", ""),
            "content": entity.get("content", ""),
            "source": "milvus",
            "url": ""
        })

    return embedding_chunks


def search_by_embedding(state: QueryGraphState) -> QueryGraphState:
    """
    向量检索服务：
    1. 根据改写后的问题和限定的商品范围
    2. 利用 BGEM3 混合检索（稠密+稀疏）技术
    3. 从 Milvus 向量数据库中召回 Top-K 最相关的知识切片
    4. 回写 embedding_chunks
    """

    # 校验并获取参数
    item_names, rewritten_query = validate_and_get_data(state)

    # 进行向量混合搜索
    milvus_list = search_by_milvus(item_names, rewritten_query)

    # 进行结果统一格式化处理
    embedding_chunks = deal_milvus_list(milvus_list)
    
    return embedding_chunks