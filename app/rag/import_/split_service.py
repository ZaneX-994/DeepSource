import json
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MIN, CHUNK_MAX_SIZE
from app.shared.runtime.logger import logger, step_log

@step_log("load_markdown_content")
def load_markdown_content(state: ImportGraphState) -> tuple[str, str]:

    md_content = state.get("md_content")
    file_title = state.get("file_title")
    md_path = state.get("md_path")

    # md_content校验
    if not md_content:
        if md_path and Path(md_path).exists():
            logger.warning(f"md_content内容为空，从备份地址:{md_path}再次读取数据")
            md_content = Path(md_path).read_text(encoding="utf-8")

        if not md_content:
            logger.error(f"md_content为空，尝试从{md_path}再次读取，结果依然为空，业务无法继续，提前终止！")
            raise ValueError(f"md_content为空，尝试从{md_path}再次读取，结果依然为空，业务无法继续，提前终止！")

    if not file_title:
        if md_path and Path(md_path).exists():
            file_title = Path(md_path).stem

        if not file_title:
            file_title = "default"

        logger.warning(f"file_title为空，启动默认值机制，赋值后:{file_title}")
        state["file_title"] = file_title

    # 数据清洗 统一换行符
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

    state['md_content'] = md_content

    return md_content, file_title

@step_log("split_chunks")
def split_chunks(md_content: str, file_title: str) -> list[dict[str, Any]]:
    """
    根据标题完成语意切割

    md_content
    ## 标题1\n
    文本内容\n
    文本内容\n
    ![]()...
    ```
    Python
    xxxx
    xxxx
    ```
    文本内容
    文本内容

    """

    # 1. md_content按行切割
    md_content_lines:list[str] = md_content.split("\n")

    # 2. 准备数据（记录当前标题和标题行以及代码块 历史chunks）
    chunks:list[dict[str, Any]] = [] # [{title, content, file_title}]
    current_title: str = None
    current_title_lines: list[str] = []
    is_code_block = False

    # 3. 循环 -> 判断是不是标题 -> 是不是代码块 -> 是不是普通行
    title_reg = re.compile(r"^\s*#{1,6}\s.+")
    for line in md_content_lines:
        line_strip = line.strip()
        if not line_strip:
            # 空行
            logger.warning(f"处理空行，跳过本次")
            continue
        # 判断是不是代码块
        if line_strip.startswith("```") or line_strip.startswith("~~~"):
            is_code_block = not is_code_block
            current_title_lines.append(line_strip)
            continue

        # 判断是不是有效标题
        if not is_code_block and title_reg.match(line_strip):
            # 情况1. 下一个标题开启 结算上一个标题
            # 情况2. 第一个标题
            if current_title and len(current_title_lines) > 1:
                chunks.append(
                    {
                        'content': '\n'.join(current_title_lines),
                        'title': current_title,
                        'file_title': file_title,
                    }
                )
            current_title = line_strip
            current_title_lines = [line_strip] # 会覆盖标题前内容：此处认为标题出现前的文字为无效文字

        else:
            current_title_lines.append(line_strip)

    # 最后一次可能没结算
    if current_title and len(current_title_lines) > 1:
        chunks.append(
            {
                'content': '\n'.join(current_title_lines),
                'title': current_title,
                'file_title': file_title,
            }
        )

    # 全文无标题
    if len(chunks) == 0 and len(current_title_lines) > 0:
        chunks.append({
            'content': '\n'.join(current_title_lines),
            'title': 'default',
            'file_title': file_title,
        })

    logger.info(f"完成了根据语意标题切割，切块数量为:{len(chunks)}")

    return chunks

@step_log("_split_long_chunk")
def _split_long_chunk(chunk) -> list[dict[str, Any]]:
    """
        chunk
            title # xxx
            file_title hk180烫金机
            content  ## xxx 内容1 ｜ 内容2
        注意不能只让第一部分有标题
            1. 先把标题移除
            2. 定义标准标题 prefix title\n
    """

    # 1. 清洗原有的content，去掉标题
    content = chunk.get("content")
    title = chunk.get("title")
    file_title = chunk.get("file_title")
    clear_content = content[len(title)+1:]

    # 2. 定义公共标准标签前缀
    sub_content_prefix = title + "\n"

    # 3. 定义递归切割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE - len(sub_content_prefix), #去除标题
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "：", ".", "!", "?", ";"]
    )

    sub_chunk_list = []

    # 4. 使用递归切割器进行content内容切割
    for part_index, split_text in enumerate(splitter.split_text(clear_content), start=1):
        # 5. 拼接每个子chunk内容
        split_text = split_text.strip()
        content_new = sub_content_prefix + split_text
        # title: 标题信息，以及是内部第几块 title_part
        sub_chunk_list.append({
            "title": f"{title}_第{part_index}部分",
            "content": content_new,
            "file_title": file_title,
            "parent_title": title,
            "part": part_index,
        })

    # 6. 返回最终结果
    logger.info(f"进入标题:{title}，完成切割后，切成:{len(sub_chunk_list)}块")
    return sub_chunk_list

step_log("_merge_short_chunks_with_same_parent_title")
def _merge_short_chunks_with_same_parent_title(refine_list) -> list[dict[str, Any]]:

    """
    合并同一个标题下过短的块（400），合并后不超过（1000）
    :param refine_list:
    :return:
    """


    """
        先指向一个基础（pre）作为参照
        如果base小于400，尝试将后面的合并入
        合并的前提是：base < 400 & 同一个parent_title & 合并后小于1000
    """

    merged_chunks = []
    base_chunk = None # 要合并入的

    for next_chunk in refine_list:
        if not base_chunk:
            base_chunk = next_chunk
            logger.info(f"设置base_chunk内容！")
            continue

        # base_chunk -> content < 400
        is_short_chunk = len(base_chunk.get("content")) < CHUNK_MIN
        # next_chunk -> parent_title = base_chunk -> parent_title and parent_title 非空
        is_same_parent = base_chunk.get("parent_title") and base_chunk.get("parent_title") == next_chunk.get("parent_title")

        if is_short_chunk and is_same_parent:

            # next_chunk [400, 600] 不能合并
            if len(next_chunk.get("content")) > CHUNK_MIN and len(next_chunk.get("content")) < CHUNK_MAX_SIZE:
                merged_chunks.append(base_chunk)
                merged_chunks.append(next_chunk)
                base_chunk = None
                continue
            # base_chunk + next_chunk   content <= 1000
            next_content = next_chunk.get("content")[len(next_chunk.get("parent_title")) + 1:]
            is_too_long = (len(base_chunk.get("content")) + len(next_content)) > CHUNK_MAX_SIZE
            if not is_too_long:
                base_chunk['content'] = base_chunk.get("content") + "\n" + next_content
                continue

        merged_chunks.append(base_chunk)
        base_chunk = next_chunk

    if base_chunk:
        merged_chunks.append(base_chunk)

    logger.info(f"进行短合并，合并之前:{len(refine_list)}，合并之后:{len(merged_chunks)}")

    return merged_chunks

@step_log("refine_chunks")
def refine_chunks(chunks) -> list[dict[str,Any]]:
    """
    进行精细切割!
      长 -> 600 -> 短切
      短 -> 400 -> 合并 -> 1000
    :param chunks:
    :return:
    """
    # 1. 定义接收最终结果的list
    refine_list = []
    # 2. 进行循环处理查看是否过长
    for chunk in chunks:
        content = chunk.get("content")
        if len(content) > CHUNK_SIZE:
            long_chunk_list = _split_long_chunk(chunk)
            refine_list.extend(long_chunk_list) #[{}.{}]
        else:
            refine_list.append(chunk)
    # 3. 进行短合并处理
    refine_list = _merge_short_chunks_with_same_parent_title(refine_list)
    # 4. 补全属性..
    for chunk in refine_list:
        if "parent_title" not in chunk:
            chunk['parent_title'] = chunk.get('title',"default_title")
        if "part" not in chunk:
            chunk['part'] = 1
    # 5. 最终返回结果
    logger.info(f"完成chunks的精细处理! 进入切块数量:{len(chunks)},处理后:{len(refine_list)}")
    return refine_list

@step_log("backup_chunks_json")
def backup_chunks_json(refine_chunks_list, file_path):
    """
    数据备份
    :param refine_chunks_list:
    :param file_path:
    :return:
    """
    # 1.获取目标地址
    json_path_obj: Path = Path(file_path).with_name(f"{Path(file_path).stem}.json")
    # 2.写入字符串
    json_path_obj.write_text(json.dumps(refine_chunks_list, indent=4, ensure_ascii=False), encoding="utf-8")
    logger.info(f"已经将切片数据备份到:{json_path_obj}")


@step_log("split_document")
def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    文档切分服务：
    1. 按标题层级做一级粗切
    2. 对超长文本做二次细切
    3. 构造 chunks 列表
    4. 回写 chunks
    """

    # 1. 参数校验
    md_content, file_title  = load_markdown_content(state)

    # 2. 确保语意切割，根据标题切割
    chunks: list[dict[str, Any]] = split_chunks(md_content, file_title)

    # 3. 进行精细切割处理
    refine_chunks_list = refine_chunks(chunks)

    # 4. 修改state
    state['chunks'] = refine_chunks_list

    # 5. 备份refine_chunks_list -> [{}, {}..] -> json string
    backup_chunks_json(refine_chunks_list, state['md_path'])

    return state