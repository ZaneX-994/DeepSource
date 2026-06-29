import re
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject
from app.infra.llm.providers import llm_provider
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import SUPPORTED_IMAGE_EXTENSIONS
from app.shared.runtime.logger import logger, step_log
from app.shared.runtime.load_prompt import load_prompt
import base64
from mimetypes import guess_type
from app.shared.utils.rate_limit_utils import apply_api_rate_limit
from app.infra.object_storage.minio_gateway import minio_gateway

@step_log("validate_and_get_data")
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

@step_log("scan_images")
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


@step_log("summarize_image")
def summarize_image(image_content: list[tuple[str, str, tuple[str, str]]], stem: str) -> dict[str, str] :

    # 获取模型对象
    vision_client = llm_provider.chat_vision()

    image_summaries = {}

    for image_name, image_path_str, image_context in image_content:
        # 添加访问限制
        apply_api_rate_limit()

        # 导入文本提示词
        image_text = load_prompt("image_summary", root_folder=stem, image_content=image_context)
        # 处理图片的base64字符串
        image_path_obj = Path(image_path_str)
        image_base64_str = base64.b64encode(image_path_obj.read_bytes()).decode("utf-8")

        # 封装提示词（图片/文本）
        human_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": image_text,
                 },
                {
                    "type": "image_url",
                    # url: "图片网络地址" ｜ base64字符串
                    "image_url": {
                        "url": f"data:{guess_type(image_name)[0]};base64,{image_base64_str}"
                    },
                }

            ]
        )

        # 封装一个调用chain
        chain = vision_client | StrOutputParser()

        # 调用chain
        image_summary = chain.invoke([human_message])

        image_summaries[image_name] = image_summary
        logger.debug(f"完成{image_name}图片意图识别，识别内容：{image_summary}")

    return image_summaries

@step_log("upload_images_and_replace")
def upload_images_and_replace(md_content: str, image_summaries: dict[str, str],
                              image_content: list[tuple[str, str, tuple[str, str]]], stem: str) -> str:
    """
    进行文件上传和md_content替换
    :param md_content: 旧md_content
    :param image_summaries: 图片描述 {图片名.jpg = 描述内容}
    :param image_content: [(图片名，图片地址，（上，下）]
    :param stem: 文件名
    :return: 新的md_content
    """

    # 获取minio客户端对象
    minio_client = minio_gateway.minio_client

    # 先删除当前文件图片（minion）
    """
        bucket_name
            /upload-images
                /hak180产品安全手册
                    /xxx.jpg
    """
    # 先查询再删除
    select_object_list = minio_client.list_objects(
        bucket_name=minio_gateway.bucket_name,
        # prefix查询的时候前面一定不能加 /
        prefix=minio_gateway.image_minio_dir[1:] + "/" + stem,
        recursive=True, # 获取前缀下的文件夹中的所有对象
    )

    # 再删除
    # select_object_list -> delete_object_list
    delete_object_list = [ DeleteObject(select_object.object_name) for select_object in select_object_list ]

    errors = minio_client.remove_objects(minio_gateway.bucket_name, delete_object_list=delete_object_list)

    for error in errors:
        logger.warning(f"文件删除状态: {error}")

    # 上传文件
    image_urls = {}
    for image_name, image_path_str, _ in image_content:
        # 网络请求可能失败，不应影响整体
        try:

            # 图片上传
            # fput
            object_name = minio_gateway.image_minio_dir + "/" + stem + "/" + image_name

            minio_client.fput_object(
                bucket_name=minio_gateway.bucket_name,
                object_name=object_name,
                file_path=image_path_str,
                content_type=guess_type(image_path_str)[0],
            )

            # 图片访问地址
            # http or https :// 端点:9000/桶名/对象名
            image_url = minio_gateway.builde_image_url(stem, image_name)
            image_urls[image_name] = image_url
            logger.info(f"{image_name} 已经上传到minio服务器，访问地址：{image_url}")

        except Exception as e:
            logger.warning(f"{image_name}图片上传到minio失败，跳过本次，继续运行")
            continue

    if not image_urls:
        logger.warning(f"图片上传全部失败！直接使用原md_content处理即可！")
        return md_content

    # 进行md_content内容替换
    for image_name, image_url in image_urls.items():
        image_summary = image_summaries.get(image_name)
        # ![](image_name) -> ![image_name](image_url)
        # 定义正则
        reg = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")

        md_content = reg.sub(lambda _: f"![{image_summary}]({image_url})", md_content)

    return md_content

@step_log("backup_new_md_content")
def backup_new_md_content(md_content_new, md_path_obj) -> str:
    # 获取目标的path
    md_new_path_obj: Path = md_path_obj.with_name(f"{md_path_obj.stem}_new{md_path_obj.suffix}")
    md_new_path_obj.write_text(md_content_new, encoding="utf-8")
    logger.info(f"将md_content_new进行备份，备份地址为{md_new_path_obj}")

    return str(md_new_path_obj)


@step_log("enrich_markdown_images")
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
    image_summaries:dict[str, str] = summarize_image(image_content, md_path_obj.stem)

    # 5. 文件上传和md_content内容转换
    md_content_new: str = upload_images_and_replace(md_content, image_summaries, image_content, md_path_obj.stem)

    # 6. 修改state
    state["md_content"] = md_content_new

    # 7. 备份md_content
    md_path_new: str = backup_new_md_content(md_content_new, md_path_obj)
    state["md_path"] = md_path_new

    return state