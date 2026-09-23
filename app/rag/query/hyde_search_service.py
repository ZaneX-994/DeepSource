from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.rag.query.embedding_search_service import validate_and_get_data, search_by_milvus, deal_milvus_list
from app.shared.runtime.load_prompt import load_prompt


def call_llm_answer(rewritten_query: str) -> str:
    # 获取模型客户端对象
    llm_client = llm_provider.chat()
    # 加载提示词
    hyde_prompt_text = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
    # 封装提示词
    messages = [
        HumanMessage(
            content=hyde_prompt_text,
        )
    ]
    # 创建调用链
    chains = llm_client | StrOutputParser()
    # 获取结果
    answer = chains.invoke(messages)

    return answer

def search_by_hyde(state: QueryGraphState) -> QueryGraphState:
    """
    HyDE 检索服务：
    1. 让 LLM 基于问题虚构一个"理想答案"
    2. 对这个假设性答案进行向量化
    3. 用答案向量在 Milvus 中检索真实文档
    4. 回写 hyde_embedding_chunks
    """

    # 校验并获取参数
    item_names, rewritten_query = validate_and_get_data(state)

    # 模型获取答案
    llm_answer = call_llm_answer(rewritten_query)

    # 进行向量混合搜索
    milvus_list = search_by_milvus(item_names, rewritten_query, llm_answer)

    # 进行结果统一格式化处理
    hyde_embedding_chunks = deal_milvus_list(milvus_list)


    return hyde_embedding_chunks