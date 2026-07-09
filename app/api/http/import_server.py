"""
导入服务 HTTP 入口模块，直接承载导入接口和相关接口业务逻辑
"""
import shutil
import sys
import uuid
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.schema.import_schema import UploadResponseSchema, StatusResponseSchema
from app.shared.runtime.logger import logger, PROJECT_ROOT
from app.process.import_.agent.main_graph import graph
from app.process.import_.agent.state import get_default_state, create_default_state, ImportGraphState
from app.infra.config.providers import infra_config

from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_PENDING,
    get_done_task_list,
    get_running_task_list,
    get_task_result,
    update_task_status, get_task_status, add_running_task, add_done_task
)
import uvicorn

app = FastAPI(
    title=infra_config.settings.import_app_name,
    description="企业化RAG导入服务，负责文件上传、导入执行与状态查询。",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(infra_config.settings.cors_origins) or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 接口1：返回html文件
@app.get("/import/html")
def import_html():
    import_html_path_obj = PROJECT_ROOT / "app" / "resources" / "html" / "import.html"

    return FileResponse(
        path=str(import_html_path_obj),
        media_type=guess_type(str(import_html_path_obj.name))[0],
    )

# 接口2: 导入文件
# 异步接口
def invoke_graph(task_id: str, local_file_path: str, local_dir: str):
    # 执行必须要 task_id | local_file_path | local_dir
    try:
        # processing
        update_task_status(task_id, status_name=TASK_STATUS_PROCESSING)

        state: ImportGraphState = create_default_state(
            task_id=task_id,
            local_file_path=local_file_path,
            local_dir=local_dir,
        )
        graph.invoke(state)

        # completed
        update_task_status(task_id, status_name=TASK_STATUS_COMPLETED)

    except Exception as e:
        # failed
        update_task_status(task_id, status_name=TASK_STATUS_FAILED)
        logger.exception(f"导入模块执行发生异常！{str(e)}")


@app.post("/upload")
def uploads(background_tasks: BackgroundTasks, files: list[UploadFile]):
    """
    异步调用图对象，解析本次上传的文件
    1.定义执行graph的函数
    2.凑齐参数 task_id | local_file_path | local_dir
    3.异步调用 执行graph
    4.返回前端数据
    :param background_tasks:
    :param files:
    :return:
    """
    # uuid4 -> 时区 ｜ 时间戳 ｜ ip地址 ｜ mac地址
    task_id: str = str(uuid.uuid4())
    # local_dir -> 项目根目录/output/日期/task_id
    local_dir: Path = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d") / task_id
    local_dir.mkdir(parents=True, exist_ok=True)
    upload_file = files[0]
    local_file_path: Path = local_dir / upload_file.filename

    # 存储文件
    # local_file_path.write_bytes(upload_file.file.read())
    # 优化：部分读取
    add_running_task(task_id, "upload_file")

    with local_file_path.open("wb") as file_buffer:
        """
        默认一次读64kb写64kb，循环读取写入
        不会阻塞服务器，支持高并发
        """
        shutil.copyfileobj(upload_file.file, file_buffer)
    add_done_task(task_id, "upload_file")

    # 异步执行
    background_tasks.add_task(
        invoke_graph,
        task_id=task_id,
        local_file_path=str(local_file_path),
        local_dir=str(local_dir),
    )


    return UploadResponseSchema(
        code=200,
        message=f"{upload_file.filename}文件上传成功!",
        task_ids=[task_id],
    )

# 接口3: 获取请求状态
@app.get("/status/{task_id}")
def task_status(task_id: str):

    done_list = get_done_task_list(task_id)
    running_list = get_running_task_list(task_id)

    task_status = get_task_status(task_id)


    return StatusResponseSchema(
        code=200,
        task_id=task_id,
        status=task_status,
        done_list=done_list,
        running_list=running_list,
    )

if __name__ == "__main__":
    uvicorn.run(app, host=infra_config.settings.app_host, port=infra_config.settings.import_app_port)