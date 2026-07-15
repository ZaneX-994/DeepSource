from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import RERANKER_MAX_INPUT_TOKENS, RERANKER_SUMMARY_CHAR_RATIO, RERANKER_MIN_SUMMARY_CHARS, \
    RERANKER_MAX_TOPK, RERANKER_MIN_TOPK, RERANKER_GAP_RATIO, RERANKER_GAP_ABS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger
from app.infra.llm.providers import llm_provider

def validate_and_get_data(state: QueryGraphState) -> tuple:
    rewritten_query = state.get("rewritten_query")
    rrf_chunks = state.get("rrf_chunks", [])
    web_search_docs = state.get("web_search_docs", [])

    if not rewritten_query or len(rrf_chunks) == 0 or len(web_search_docs) == 0:
        logger.error("rewritten_query, rrf_chunks, web_search_docs可能为空，业务无法继续！")
        raise ValueError("rewritten_query, rrf_chunks, web_search_docs可能为空，业务无法继续！")


    return rewritten_query, rrf_chunks, web_search_docs


def deal_rrf_and_web_result(rrf_chunks, web_search_docs):
    reranker_docs = []

    for chunk in rrf_chunks:
        reranker_docs.append({
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("content"),
            "title": chunk.get("title"),
            "score": 0,
            "type": "milvus",
            "url": None
        })


    for doc in web_search_docs:
        reranker_docs.append({
            "chunk_id": None,
            "text": doc.get("snippet"),
            "title": doc.get("title"),
            "score": 0,
            "type": "web",
            "url": doc.get("url")
        })

    return reranker_docs


def create_question_answer_pair_list(rewritten_query, reranker_docs):
    question_answer_pair_list = []

    # 获取问题并计算token数量
    reranker_model = llm_provider.reranker_model()
    tokenizer = reranker_model.tokenizer
    rewritten_query_token_list = tokenizer.encode(rewritten_query, add_special_tokens=False)
    rewritten_query_token_length = len(rewritten_query_token_list)

    # 循环获取答案
    for doc in reranker_docs:
        # 答案的长度判断
        answer = doc.get("text")
        answer_length = len(tokenizer.encode(answer, add_special_tokens=False))
        # 超长处理
        # reranker固定4个分隔符号
        if rewritten_query_token_length + answer_length + 4 > RERANKER_MAX_INPUT_TOKENS:
            # 调用模型进行压缩
            limit = max(
                RERANKER_MIN_SUMMARY_CHARS,
                int((RERANKER_MAX_INPUT_TOKENS - 4 - rewritten_query_token_length) / RERANKER_SUMMARY_CHAR_RATIO)
            )
            rerank_text_refine_str = load_prompt("rerank_text_refine", question=rewritten_query, answer=answer, limit=limit)

            messages = [
                HumanMessage(
                    content=rerank_text_refine_str,
                )
            ]

            chains = llm_provider.chat() | StrOutputParser()
            answer = chains.invoke(messages)

        question_answer_pair_list.append((rewritten_query, answer))

    return question_answer_pair_list


def use_reranker_deal_score(question_answer_pair_list, reranker_docs):
    # 调用reranker打分
    reranker_model = llm_provider.reranker_model()
    score_list = reranker_model.compute_score(question_answer_pair_list, normalize=True)
    # 倒序排序
    for score, doc in zip(score_list, reranker_docs):
        doc["score"] = score

    reranker_docs.sort(key = lambda doc: doc.get("score"), reverse = True)


def dynamic_truncate_reranker_docs(reranker_docs):
    """
        动态截取topk个
            RERANKER_MAX_TOPK = 6 -> 最多6个
            RERANKER_MIN_TOPK = 2 -> 最少2个
            RERANKER_GAP_RATIO = 0.2 -> 断崖百分比
            RERANKER_GAP_ABS = 0.2 -> 断崖分差值 0.8， 0.5（差值大于0.2，跳过）
    :param reranker_docs:
    :return:
    """
    top_max = RERANKER_MAX_TOPK
    top_min = RERANKER_MIN_TOPK
    gap_ratio = RERANKER_GAP_RATIO
    gap_abs = RERANKER_GAP_ABS

    top_max = min(top_max, len(reranker_docs))


    topk = top_max


    if top_max > top_min:

        start_score = reranker_docs[top_min - 1].get("score", 0.0)

        for pre in range(top_min - 1, top_max - 1):
            # 获取前指针对应的分数
            pre_score = reranker_docs[pre].get("score", 0.0)
            next_score = reranker_docs[pre + 1].get("score", 0.0)
            abs_diff = pre_score - next_score
            ratio_diff = abs_diff / pre_score

            # 判断断崖
            if abs_diff > gap_abs or ratio_diff > gap_ratio:
                # 截取到前置指针位置
                topk = pre + 1
                break

            # 判断累计断崖
            cul_diff_ratio = (start_score - next_score) / start_score
            if cul_diff_ratio > gap_ratio * 2:
                topk = pre + 1
                break

    final_reranker_docs = reranker_docs[:topk]

    return final_reranker_docs


def rerank_documents(state: QueryGraphState) -> QueryGraphState:
    """
    重排序服务：
    1. 合并 RRF 和 Web Search 的文档
    2. 使用 BGE Reranker 模型计算相关性得分
    3. 根据得分动态截断，智能截取 TopK
    4. 回写 reranked_docs
    """

    rewritten_query, rrf_chunks, web_search_docs = validate_and_get_data(state)

    # 数据格式化处理
    reranker_docs = deal_rrf_and_web_result(rrf_chunks, web_search_docs)

    # 组装问题和答案
    question_answer_pair_list: list[list[tuple[str, str]]] = create_question_answer_pair_list(rewritten_query, reranker_docs)

    logger.info(f"排序和打分前的数据: {reranker_docs}")
    # 打分 + 排序
    use_reranker_deal_score(question_answer_pair_list, reranker_docs)
    logger.info(f"排序和打分后的数据: {reranker_docs}")

    # 动态截取数据
    reranker_docs = dynamic_truncate_reranker_docs(reranker_docs)

    state["reranked_docs"] = reranker_docs

    return state