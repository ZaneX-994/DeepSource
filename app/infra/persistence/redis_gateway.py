"""
Redis Gateway - 基于连接池的 Redis 访问网关
提供统一的 Redis 访问入口，内置连接池管理与健康检查
"""

import os
import logging
from typing import Optional, Any
from contextlib import contextmanager

import redis
from redis.connection import ConnectionPool

from app.shared.config.redis_config import redis_config

logger = logging.getLogger(__name__)


class RedisGateway:
    """Redis 访问网关，封装连接池与常用操作"""

    _instance: Optional["RedisGateway"] = None
    _pool: Optional[ConnectionPool] = None

    def __new__(cls, *args, **kwargs):
        """单例模式：确保全局共享同一个连接池"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = None,
        password: str = None,
        max_connections: int = 20,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 3.0,
        decode_responses: bool = True,
    ):
        # 避免重复初始化连接池
        if self._pool is not None:
            return

        self.host = host or os.getenv("REDIS_HOST", "127.0.0.1")
        self.port = port or int(os.getenv("REDIS_PORT", 6379))
        self.db = db if db is not None else int(os.getenv("REDIS_DB", 0))
        self.password = password or os.getenv("REDIS_PASSWORD", None)

        logger.info(
            f"Initializing Redis connection pool: {self.host}:{self.port}/{self.db}, "
            f"max_connections={max_connections}"
        )

        self._pool = ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            decode_responses=decode_responses,
            retry_on_timeout=True,
        )

    @property
    def client(self) -> redis.Redis:
        """获取一个 Redis 客户端实例（从连接池中借用连接）"""
        return redis.Redis(connection_pool=self._pool)

    @contextmanager
    def get_connection(self):
        """
        上下文管理器：安全地获取和释放连接
        用法:
            with gateway.get_connection() as conn:
                conn.set("key", "value")
        """
        conn = self._pool.get_connection("_")
        try:
            yield conn
        finally:
            self._pool.release(conn)

    def ping(self) -> bool:
        """健康检查"""
        try:
            return self.client.ping()
        except redis.ConnectionError as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    # ==================== 常用操作快捷方法 ====================

    def get(self, key: str) -> Optional[str]:
        return self.client.get(key)

    def set(self, key: str, value: Any, ex: int = None) -> bool:
        """设置键值对，ex 为过期时间（秒）"""
        return self.client.set(key, value, ex=ex)

    def delete(self, *keys: str) -> int:
        return self.client.delete(*keys)

    def exists(self, key: str) -> bool:
        return bool(self.client.exists(key))

    def expire(self, key: str, seconds: int) -> bool:
        return self.client.expire(key, seconds)

    def close(self):
        """关闭连接池（应用退出时调用）"""
        if self._pool:
            self._pool.disconnect()
            self._pool = None
            RedisGateway._instance = None
            logger.info("Redis connection pool closed.")

    def __repr__(self):
        status = "connected" if self.ping() else "disconnected"
        return f"<RedisGateway {self.host}:{self.port}/{self.db} [{status}]>"

# 首次创建时初始化连接池，后续调用复用同一连接池
redis_gateway = RedisGateway(
    host=redis_config.redis_host,
    port=int(redis_config.redis_port),
    password=redis_config.redis_password,
    db=int(redis_config.redis_db),
)