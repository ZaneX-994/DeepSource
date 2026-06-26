from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger
from pathlib import Path


def resolve_input_file(state: ImportGraphState) -> ImportGraphState:
    """
    入口识别服务：
    1. 校验 local_file_path
    2. 识别文件类型（PDF / Markdown）
    3. 回写 is_pdf_read_enabled / is_md_read_enabled
    4. 回写 pdf_path / md_path / file_title
    """

    local_file_path = state.get("local_file_path")

    if not local_file_path:
        logger.error(f"local_file_path为空，无法继续业务，提前终止！")
        raise FileNotFoundError(f"local_file_path为空，无法继续业务，提前终止！")

    if local_file_path.endswith(".md"):
        state["md_path"] = local_file_path
        state["is_md_read_enabled"] = True
        state["is_pdf_read_enabled"] = False
    elif local_file_path.endswith(".pdf"):
        state["pdf_path"] = local_file_path
        state["is_md_read_enabled"] = False
        state["is_pdf_read_enabled"] = True
    # 扩展文件类型
    # elif...
    else:
        logger.warning(f"{local_file_path}对应文件类型无法解析，只支持md/pdf格式，提前终止！")
        state["is_md_read_enabled"] = False
        state["is_pdf_read_enabled"] = False

    file_title = Path(local_file_path).stem
    state["file_title"] = file_title


    return state

