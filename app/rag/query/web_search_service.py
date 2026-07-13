import asyncio
import json

from agents.mcp import MCPServerStreamableHttp

from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger
from app.infra.config.providers import infra_config


def validate_and_get_query(state: QueryGraphState) -> str:

    rewritten_query = state.get("rewritten_query", "")

    if not rewritten_query:
        logger.error("重写的问题为空，业务无法继续！")
        raise ValueError("重写的问题为空，业务无法继续！")

    return rewritten_query

async def open_ai_mcp(rewritten_query: str):
    # 连接mcp服务端
    mcp_server = MCPServerStreamableHttp(
        name="Streamable HTTP Python Server",
        params={
            "url": infra_config.mcp_config.mcp_base_url,
            "headers": {"Authorization": f"Bearer {infra_config.mcp_config.api_key}"},
            "timeout": 50,
        },
        cache_tools_list=True,
        max_retry_attempts=3,
    )


    try:
        # mcp服务连接
        await mcp_server.connect()
        # mcp工具调用
        mcp_result = await mcp_server.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": rewritten_query,
                "count": 5
            }
        )

        return mcp_result
    except Exception as e:
        logger.error(f"mcp调用出错，问题: {str(e)}")
    finally:
        # 清空连接
        await mcp_server.cleanup()

def search_by_web(state: QueryGraphState) -> QueryGraphState:
    """
    网络搜索服务：
    1. 通过 MCP 协议异步调用百炼联网搜索接口
    2. 将用户的查询转化为实时的、结构化的网络搜索结果
    3. 包含标题、链接和摘要
    4. 回写 web_search_docs
    """

    rewritten_query = validate_and_get_query(state)

    # async使用openai提供的mcp调用方式
    mcp_result = asyncio.run(open_ai_mcp(rewritten_query))

    # 结果解析
    text = mcp_result.content[0].text
    text_dict = json.loads(text)
    web_search_docs = text_dict.get("pages", [])

    return web_search_docs