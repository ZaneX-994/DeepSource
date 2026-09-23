import re

from langchain_core.messages import HumanMessage
from sqlalchemy import false

from app.infra.llm.providers import llm_provider
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import SUPPORTED_IMAGE_EXTENSIONS
from app.rag.query.item_name_confirm_service import get_history_messages_and_context
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.task_utils import add_done_task,add_running_task,push_to_session
from app.shared.utils.sse_utils import SSEEvent
from app.shared.runtime.logger import logger
from app.infra.persistence.history_repository import history_repository
import time
import sys


def state_exist_answer(state: QueryGraphState) -> bool:
    answer = state.get("answer")
    if not answer:
        logger.info("answer内容为空，业务正常进行查询！跳入回答环节！")
        return False
    is_stream = state.get("is_stream")
    if is_stream:
        # 流式：现在就要把结果推到队列中
        push_to_session(session_id=state.get("session_id"), event=SSEEvent.DELTA, data={"delta": answer})
    else:
        return True


def load_answer_prompt(state: QueryGraphState) -> str:
    """
        加载提示词 用于模型润色answer回答
    :param state:
    :return:
    """

    question = state.get("rewritten_query")
    reranked_docs = state.get("reranked_docs", [])
    context = ""
    # 第1部分，标题：xx，来源：网络｜向量数据库，置信度：xx，内容：xx\n

    for index, doc in enumerate(reranked_docs, start=1):
        context += (f"第{index}部分, 标题:{doc.get("title")}, 来源:{'网络搜索' if doc.get('type') == 'web' else '向量库'}, "
                    f"置信度:{doc.get('score')}, 内容:{doc.get('text')}\n")

    item_names = f"{','.join(state.get("item_names", []))}"

    # [{_id, role, text, rewritten_query, item_names, image_urls, ts}]
    history_text = get_history_messages_and_context(state.get("session_id"))

    # 加载提示词
    answer_out_prompt= load_prompt("answer_out", question=question, context=context, item_names=item_names, history=history_text)

    return answer_out_prompt


def call_llm_deal_anwser(state: QueryGraphState, answer_prompt_text: str):
    """
        处理answer
    :param state:
    :param answer_prompt_text:
    :return:
    """

    is_stream = state.get("is_stream", False)
    llm_client = llm_provider.chat()
    messages = [
        HumanMessage(
            content=answer_prompt_text,
        )
    ]
    answer = ""

    if is_stream:
        # 流式
        stream = llm_client.stream(messages)
        for chunk in stream:
            data = chunk.content # 增量数据delta -> 推送到队列
            push_to_session(session_id=state.get("session_id"), event=SSEEvent.DELTA, data={"delta": data})
            answer += data
    else:
        # 非流式
        response = llm_client.invoke(messages) # AIMessage
        answer = response.content

    state["answer"] = answer


def extract_text_image_urls(state):

    """
        处理图片
            text
            url
    :param state:
    :return:
    """

    # 获取reranked_docs text url
    reranked_docs = state.get("reranked_docs", [])

    # 定义正则
    image_re = re.compile(r"\!\[.*?\]\((.*?)\)")
    image_urls = []

    for doc in reranked_docs:
        url: str = doc.get("url")
        text: str = doc.get("text")
        if url and url.endswith(SUPPORTED_IMAGE_EXTENSIONS):
            image_urls.append(url)

        url_list = image_re.findall(text)
        if url_list and len(url_list) > 0:
            image_urls.extend(url_list)

    state["image_urls"] = image_urls


def save_answer_message_history(state: QueryGraphState):
    history_repository.save_message(
        session_id=state.get("session_id"),
        role="assistant",
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names", []),
        image_urls=state.get("image_urls", []),
    )

def generate_answer(state: QueryGraphState) -> QueryGraphState:
    """
    答案生成服务：
    1. 检查前置答案（如有追问或拒绝回答，直接输出）
    2. 构建 Prompt（用户问题 + 历史对话 + TopK 文档）
    3. 调用 LLM 生成最终答案（支持流式推送）
    4. 从引用文档中提取图片 URL
    5. 写入 MongoDB 历史记录
    6. 回写 answer 和 image_urls
    """
    ""

    # 检查state["answer"]
    has_answer: bool = state_exist_answer(state)

    if not has_answer:
        # 没有answer 模型润色answer 提取image_urls
        answer_prompt_text: str = load_answer_prompt(state)
        # 调用llm处理
        call_llm_deal_anwser(state, answer_prompt_text)
        # 使用正则匹配image_urls
        extract_text_image_urls(state)

    # 保存记录
    save_answer_message_history(state)

    return state