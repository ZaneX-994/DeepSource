from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger
from app.infra.persistence.history_repository import history_repository
from app.infra.vector_store.milvus_gateway import milvus_gateway

def validate_and_get_data(state: QueryGraphState) -> tuple[str, str]:
    # 1.获取数据
    session_id = state.get('session_id')
    original_query = state.get('original_query')

    # 2.进行数据校验
    if not session_id or not original_query:
        logger.error(f"session_id 或者 original_query 为空，业务无法继续！")
        raise ValueError(f"session_id 或者 original_query 为空，业务无法继续！")

    # 3.返回结果
    return session_id, original_query


def get_history_messages_and_context(session_id:str) -> str:
    # 获取最近10条历史记录
    message_list:list[dict] = history_repository.list_recent(session_id=session_id, limit=10)

    if not message_list or len(message_list) == 0:
        logger.warning(f"当前会话：{session_id} 没有历史对话记录！history_context为空！")
        return "无对话记录"

    final_message_list = [
        item
        for item in message_list if len(item.get("item_names", [])) > 0
    ]

    if not final_message_list or len(final_message_list) == 0:
        logger.warning(f"当前会话：{session_id} 没有有效历史对话记录！history_context为空！")
        return "无有效对话记录"

    # 拼接history_context
    history_context = ""
    for index, item in enumerate(final_message_list):
        history_context += (f"序号:{index}，{'提问: ' if item.get('role') == 'user' else '回答: '}"
                            f"{item.get('rewritten_query') if item.get('role') == 'user' else item.get('text')[:50]}"
                            f"本次关联主体: {','.join(item.get('item_names', []))} \n")
        # 序号1: 提问：xxx, 关联主体 a,b,c
        # 序号2: 回答：xxx, 关联主体 a,b,c

    return history_context


def call_llm_item_name_and_rewritten_query(history_context: str, original_query: str) -> dict:
    # 加载模型对象
    json_llm_client = llm_provider.chat(json_mode=True)
    # 加载提示词
    history_prompt_text = load_prompt("rewritten_query_and_itemnames", query=original_query, history_text=history_context)
    # 包装提示词对象
    history_prompt_messages = [
        HumanMessage(
            content=history_prompt_text,
        )
    ]

    # 创建调用链
    chains = json_llm_client | JsonOutputParser()
    # 调用获取结果
    result = chains.invoke(history_prompt_messages)

    # 参数校验赋予默认值
    if "item_names" not in result:
        result["item_names"] = []

    if "rewritten_query" not in result:
        result["rewritten_query"] = original_query

    return result


def search_by_item_names(item_names: list[str]) -> dict[str, list[dict]]: # -> {"reqst_item_name": [{"item1": "score1"}, {"item1": "score1"}]}

    final_result = {}

    # 批量把item_names转化成稀疏和稠密向量
    # {dense: [[],[]], sparse: [[],[]]
    item_names_vectors = llm_provider.generate_embeddings(item_names)
    for index in range(0, len(item_names)):
        # 获取item_name对应的稠密稀疏向量
        item_name = item_names[index]
        item_name_dense = item_names_vectors['dense'][index]
        item_name_sparse = item_names_vectors['sparse'][index]

        # 稀疏和稠密向量混合检索

        # 1.创建annSearchRequest
        reqs = milvus_gateway.create_request(item_name_dense, item_name_sparse, limit=5*2)

        # 2.创建reranker排序器

        # 3.调用混合检索
        results = milvus_gateway.hybrid_search(
            collection_name=milvus_gateway.item_name_collection_name,
            reqs=reqs,
            ranker_weights=(0.5, 0.5),
            norm_score=True,
            output_fields=['item_name']
        )
        # results = [[{id: 主键, distance: 0.9, entity: {item_name: 具体名字}}, {}]]
        # 处理检索结果
        item_name_milvus_list = []
        if len(results[0]) > 0:
            for item in results[0]:
                item_name_milvus_list.append({
                    "item_name": item.get("entity").get("item_name"),
                    "score": item.get("distance")
                })


        final_result[item_name] = item_name_milvus_list

    return final_result


def select_item_names(milvus_result: dict[str, list[dict]]) -> dict[str, list]:

    """
    根据分数明确确定和可选的item_name列表
    :param milvus_result:
    :return:
    """

    confirmed_list = []
    optional_list = []

    # item_name: [{item_name: xxx, score: xxx}, {...},..]
    for item_name, milvus_result_list_dict in milvus_result.items():
        # confirmed_list -> [ 稠密向量满分1 * 0.5 + 稀疏向量满分 0.75 * 0.5 ] = 0.875 -> 0.8+ 确认
        # optional_list -> 0.6 - 0.8
        high_score_list = [ item for item in milvus_result_list_dict if item.get("score", 0) >= 0.73 ]
        md_score_list = [ item for item in milvus_result_list_dict if 0.6 <= item.get("score", 0) < 0.73 ]

        if len(high_score_list) > 0:
            confirmed_list.append(high_score_list[0])
            logger.info(f"模型识别的item_name: {item_name}，有对应向量数据库中确认的item_name: {high_score_list[0].get("item_name")}")
            continue

        if len(md_score_list) > 0:
            optional_list.extend(md_score_list[:2])
            logger.info(f"模型识别的item_name: {item_name}，有对应向量数据库中没有确认的item_name，但是有可选的 {','.join([ item.get("item_name") for item in md_score_list[:2]  ])}")
            continue

    return {
        "confirmed_list": confirmed_list,
        "optional_list": optional_list
    }


def apply_item_name_result(state: QueryGraphState, list_dict: dict[str, list], rewritten_query: str):
    """
    修改state
    :param state:
    :param list_dict:
    :param rewritten_query:
    :return:
    """
    confirmed_list = list_dict.get("confirmed_list", [])
    optional_list = list_dict.get("optional_list", [])

    if len(confirmed_list) > 0:
        state["item_names"] = [ item.get("item_name") for item in confirmed_list ]
        state["rewritten_query"] = rewritten_query
        if "answer" in state:
            state["answer"] = None
        return

    if len(optional_list) > 0:
        # 没有确认的 但有可选的
        state["answer"] = f"本次提问没有确认的主体，但是有相似可选的{','.join([item.get("item_name") for item in optional_list])}，请再次确认"
        return

    state["answer"] = "本次问题没有关联到任何主体，且没有相似可选的主体！请先明确主体！"


def save_history_message(state: QueryGraphState):
    history_repository.save_message(
        session_id=state.get("session_id"),
        role="user",
        text=state.get("original_query"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names", []),
        image_urls=[]
    )


def confirm_item_name(state: QueryGraphState) -> QueryGraphState:
    """
    意图确认服务：
    1. 结合历史对话提取商品名
    2. 将模糊问题改写为完整独立的精准问题
    3. 在 Milvus 向量库中进行混合搜索
    4. 根据评分高低自动对齐标准型号，或生成反问让用户手动确认
    5. 同步历史记录到 MongoDB
    """

    session_id, original_query = validate_and_get_data(state)

    # 获取历史信息和拼接上下文
    history_context = get_history_messages_and_context(session_id)

    # 调用模型识别item_names和重写的问题
    result: dict = call_llm_item_name_and_rewritten_query(history_context, original_query)
    list_dict = {}
    if len(result.get("item_names", [])) > 0:
        # 进行向量数据库的搜索 llm - 分析 -> item_names -> milvus中查询 -> 打分
        milvus_result: dict[str, list[dict]] = search_by_item_names(result.get("item_names", []))
        # 根据打分结果确认 1.确认列表 2.可选列表
        # dict{"confirmed_list":[], "optional_list": []}
        list_dict: dict[str, list] = select_item_names(milvus_result)

    # 修改state
    apply_item_name_result(state, list_dict, result.get("rewritten_query", ""))

    # 记录提问和聊天记录
    save_history_message(state)

    return state