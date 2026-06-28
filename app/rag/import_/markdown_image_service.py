import re
from pathlib import Path

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import SUPPORTED_IMAGE_EXTENSIONS
from app.shared.runtime.logger import logger

def validate_and_get_data(state: ImportGraphState):

    # 1.获取md_path
    md_path = state.get('md_path')

    # 2.判断存在性
    if not md_path:
        logger.error(f"核心参数md_path为空，业务无法继续！")
        raise ValueError(f"核心参数md_path为空，业务无法继续！")

    md_path_obj = Path(md_path)
    if not md_path_obj.exists():
        logger.error(f"md_path: {md_path}，但是没有真实的文件，业务无法继续！")
        raise ValueError(f"md_path: {md_path}，但是没有真实的文件，业务无法继续！")

    # 3.读取md_content
    md_content = md_path_obj.read_text(encoding='utf-8')
    if not md_content:
        logger.error(f"md_path: {md_path}，内容为空，业务无法继续！")
        raise ValueError(f"md_path: {md_path}，内容为空，业务无法继续！")
    state['md_content'] = md_content

    # 4.获取images文件夹地址
    images_path_obj: Path = md_path_obj.parent / 'images'

    return md_content, images_path_obj, md_path_obj

def scan_images(image_path_obj: Path, md_content: str) -> list[tuple[str, str, tuple[str, str]]] :
    # 创建容器
    image_context = []

    # 遍历图片文件夹images获取每一个文件
    for image_obj in image_path_obj.iterdir():
        image_name = image_obj.name
        # 判断文件是图片
        if image_obj.suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            logger.warn(f"当前文件名: {image_name} 不是图片，无需处理，跳过本次！")
            continue
        image_reg = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")

        match = image_reg.search(md_content)

        if not match:
            logger.warning(f"{image_name}图片没有被md文件引用，跳过本次！")
            continue

        start = match.start()
        end = match.end()

        pre_context = md_content[max(0, start-100):start]
        post_context = md_content[end:min(end+100, len(md_content))]

        image_context.append(
            (
                image_name, str(image_obj),
                (pre_context, post_context),
            )
        )

    return image_context

def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:
    """
    Markdown 图片增强服务：
    1. 扫描 Markdown 中的图片
    2. 调用多模态模型生成图片说明
    3. 上传图片到 MinIO
    4. 替换 Markdown 图片地址并回写 md_content
    """
    # 1.参数校验和获取
    md_content, img_path_obj, md_path_obj = validate_and_get_data(state)

    # 2.没有文件提前终止
    if (not img_path_obj.exists()) or img_path_obj.is_file() or len(list(img_path_obj.iterdir())) == 0:
        logger.info(f"{md_path_obj}文档对应的images为空或者没有图片，不需要图片识别，提前结束当前节点！")
        return state


    # 3. 获取每张图的信息
    image_content: list[tuple[str, str, tuple[str, str]]] = scan_images(img_path_obj, md_content)


    # 4. 使用视觉模型识别图形意图


    return state