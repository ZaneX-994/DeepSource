"""
文档哈希注册表 - 支持同名文档多Hash存储、Hash存在性查询与异名同内容检测
"""

import logging
from typing import Set
from app.infra.persistence.redis_gateway import RedisGateway, redis_gateway

logger = logging.getLogger(__name__)

# Redis Key 前缀，避免与其他业务冲突
KEY_DOC_HASHES = "doc:hashes:"      # Set: doc:hashes:{filename} -> {hash1, hash2, ...}
KEY_HASH_FILES = "hash:files:"      # Set: hash:files:{hash_value} -> {filename1, filename2, ...}

# 默认过期时间：1天（秒），可根据业务需要调整（如 30天 = 30*24*3600）
DEFAULT_TTL = 30*24*3600


class DocRegistry:
    """文档哈希注册表"""

    def __init__(self, gateway: RedisGateway = None):
        self.gw = gateway or RedisGateway()

    # ==================== 核心操作 ====================

    def register(self, filename: str, file_hash: str, ttl: int = DEFAULT_TTL) -> bool:
        """
        注册文档：将 hash 加入同名文档集合，并建立反向索引
        - ttl: 过期时间（秒），默认1天
        - 返回 True 表示新注册，False 表示该 hash+filename 组合已存在
        """
        client = self.gw.client
        hashes_key = f"{KEY_DOC_HASHES}{filename}"
        files_key = f"{KEY_HASH_FILES}{file_hash}"

        # 检查该 hash+filename 组合是否已存在（避免重复写入和无效刷新TTL）
        if client.sismember(files_key, filename):
            logger.info(f"Already registered: {filename} -> {file_hash[:16]}...")
            return False

        # 使用 pipeline 保证原子性 + 批量写入
        pipe = client.pipeline(transaction=True)

        # 正向索引：文件名 -> hash集合
        pipe.sadd(hashes_key, file_hash)
        pipe.expire(hashes_key, ttl)

        # 反向索引：hash -> 文件名集合
        pipe.sadd(files_key, filename)
        pipe.expire(files_key, ttl)

        pipe.execute()

        logger.info(f"Registered: {filename} -> {file_hash[:16]}... (TTL={ttl}s)")
        return True

    def hash_exists(self, file_hash: str) -> bool:
        """O(1) 查询某个 hash 是否已存在（全局去重）"""
        return bool(self.gw.client.exists(f"{KEY_HASH_FILES}{file_hash}"))

    def get_hashes(self, filename: str) -> Set[str]:
        """获取某个文件名下所有已注册的 hash 集合"""
        members = self.gw.client.smembers(f"{KEY_DOC_HASHES}{filename}")
        return {m.decode('utf-8') if isinstance(m, bytes) else m for m in members} if members else set()

    def get_files_by_hash(self, file_hash: str) -> Set[str]:
        """🆕 根据 hash 查询所有关联的文件名（含不同文件名）"""
        members = self.gw.client.smembers(f"{KEY_HASH_FILES}{file_hash}")
        return {m.decode('utf-8') if isinstance(m, bytes) else m for m in members} if members else set()

    def has_different_filename(self, filename: str, file_hash: str) -> bool:
        """🆕 判断是否存在相同内容（同hash）但不同文件名的文档"""
        files = self.get_files_by_hash(file_hash)
        # 去掉自身后，如果还有剩余，说明存在异名同内容
        other_files = files - {filename}
        return len(other_files) > 0

    def is_duplicate(self, filename: str, file_hash: str) -> bool:
        """判断该文件名的某个 hash 是否已注册过"""
        return bool(self.gw.client.sismember(
            f"{KEY_DOC_HASHES}{filename}", file_hash
        ))

    def unregister(self, filename: str, file_hash: str) -> bool:
        """
        注销文档：从正向集合移除 hash，并从反向集合移除 filename
        """
        client = self.gw.client
        hashes_key = f"{KEY_DOC_HASHES}{filename}"
        files_key = f"{KEY_HASH_FILES}{file_hash}"

        # 从正向索引移除 hash
        removed_from_hashes = client.srem(hashes_key, file_hash)
        # 从反向索引移除 filename
        removed_from_files = client.srem(files_key, filename)

        if removed_from_hashes or removed_from_files:
            logger.info(f"Unregistered: {filename} -> {file_hash[:16]}...")
            return True

        return False

    def count(self, filename: str) -> int:
        """获取某文件名下的文档版本数"""
        return self.gw.client.scard(f"{KEY_DOC_HASHES}{filename}")


# 模块级单例（可选，方便外部直接 import registry 使用）
registry = DocRegistry(redis_gateway)