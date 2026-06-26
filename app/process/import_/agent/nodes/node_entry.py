import json

from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState, create_default_state
from app.rag.import_.entry_service import resolve_input_file
from app.shared.runtime.logger import logger

@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    """
    add_running_task(state["task_id"], "node_entry")
    # 调用业务逻辑
    state = resolve_input_file(state)
    add_done_task(state["task_id"], "node_entry")
    return state

if __name__ == "__main__":

    # 单元测试：覆盖不支持类型、MD、PDF三种场景
    logger.info("====== 开始node_entry单元测试 ======")

    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="测试文档.txt"
    )
    result1 = node_entry(test_state1)
    print(f"第一次测试结果：\n {json.dumps(result1, indent=4, ensure_ascii=False)}")

    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="测试文档.md"
    )
    result2 = node_entry(test_state2)
    print(f"第二次测试结果：\n {json.dumps(result2, indent=4, ensure_ascii=False)}")

    test_state3 = create_default_state(
        task_id="test_task_003",
        local_file_path="测试文档.pdf"
    )
    result3 = node_entry(test_state3)
    print(f"第三次测试结果：\n {json.dumps(result3, indent=4, ensure_ascii=False)}")

    logger.info("====== 结束node_entry单元测试 ======")