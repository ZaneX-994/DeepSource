from dataclasses import dataclass

from app.shared.config.common import env_str


@dataclass
class RedisConfig:
    redis_host: str
    redis_port: str
    redis_password: str
    redis_db: str


redis_config = RedisConfig(
    redis_host=env_str("REDIS_HOST"),
    redis_port=env_str("REDIS_PORT"),
    redis_password=env_str("REDIS_PASSWORD"),
    redis_db=env_str("REDIS_DB"),
)


