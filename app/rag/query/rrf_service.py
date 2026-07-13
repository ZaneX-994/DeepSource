from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import NODE_RRF_K, NODE_RRF_LIMIT_TOP
from app.shared.runtime.logger import logger

def validate_and_get_data(state: QueryGraphState):
    embedding_chunks = state.get("embedding_chunks", [])
    hyde_embedding_chunks = state.get("hyde_embedding_chunks", [])

    if len(embedding_chunks) == 0 or len(hyde_embedding_chunks) == 0:
        logger.error(f"embedding_chunks或者hyde_embedding_chunks数据为空，业务无法继续！")
        raise ValueError(f"embedding_chunks或者hyde_embedding_chunks数据为空，业务无法继续！")

    return embedding_chunks, hyde_embedding_chunks


def use_by_rrf(rrf_list: list, top: int = NODE_RRF_LIMIT_TOP, k: int = NODE_RRF_K):
    """
        进行权重排名计算
    :param rrf_list:
    :param top:
    :param k:
    :return:
    """
    score_dict: dict[str, float] = {}
    chunk_dict: dict[str, dict] = {}

    # 循环每一路
    for weight, current_chunks in rrf_list:
        for rank, chunk in enumerate(current_chunks, start=1):
            # chunk -> {id, title, content...}
            # 上一次计算得分 + 权重 * 1 / ( k + rank)
            score_dict[chunk.get("chunk_id")] = score_dict.get(chunk.get("chunk_id"), 0.0) + weight / (k + rank)
            chunk_dict.setdefault(chunk.get("chunk_id"), chunk) # 每次检查只保留第一次数据

    # score_dict = {chunk_id1: 分数1, chunk_id2, 分数2}
    # chunk_dict = {chunk_id: chunk, score: 向量数据库打分}
    chunk_list = []
    for chunk_id, score in score_dict.items():
        chunk = chunk_dict.get(chunk_id)
        chunk['score'] = score # milvus打的分替换成rrf排序的分
        chunk_list.append(chunk)
    # 排序处理
    chunk_list.sort(key = lambda chunk : chunk.get("score", 0), reverse = True)

    # 截取topk
    return chunk_list[:top]


def fuse_by_rrf(state: QueryGraphState) -> QueryGraphState:
    """
    RRF 融合服务：
    1. 合并来自不同检索源的文档列表
    2. 应用 RRF 算法消除分数差异
    3. 给出综合排名最高的文档列表（Top 10）
    4. 回写 rrf_chunks
    """

    embedding_chunks, hyde_embedding_chunks = validate_and_get_data(state)

    rrf_list = [(1.0, embedding_chunks), (1.0, hyde_embedding_chunks)]

    rrf_chunks = use_by_rrf(rrf_list)

    state["rrf_chunks"] = rrf_chunks
    
    return state