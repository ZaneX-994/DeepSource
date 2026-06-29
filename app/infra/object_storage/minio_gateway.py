from minio import Minio
from app.shared.clients.minio_utils import get_minio_client
from app.infra.config.providers import infra_config

class MinioGateway:

    # 提供获取桶名称的函数
    @property
    def bucket_name(self):
        return infra_config.minio_config.bucket_name

    # 提供获取图片前缀的函数
    @property
    def image_minio_dir(self):
        return infra_config.minio_config.minio_img_dir

    # 提供获取minio_client的函数
    @property
    def minio_client(self):
        return get_minio_client()

    # minio上传文件不会返回访问地址 -> 手动拼接访问地址
    def builde_image_url(self, stem: str, object_name: str):

        image_url = "https://" if infra_config.minio_config.minio_secure else "http://" + (f"{infra_config.minio_config.endpoint}"
                           f"/{infra_config.minio_config.bucket_name}{infra_config.minio_config.minio_img_dir}/{stem}/{object_name}")

        return image_url


minio_gateway = MinioGateway()

