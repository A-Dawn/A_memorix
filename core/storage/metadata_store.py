"""
元数据存储模块

基于SQLite的元数据管理，存储段落、实体、关系等信息。
"""

import sqlite3
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple

from src.common.logger import get_logger
from ..utils.hash import compute_hash, normalize_text

logger = get_logger("A_Memorix.MetadataStore")


class MetadataStore:
    """
    元数据存储类

    功能：
    - SQLite数据库管理
    - 段落/实体/关系元数据存储
    - 增删改查操作
    - 事务支持
    - 索引优化

    参数：
        data_dir: 数据目录
        db_name: 数据库文件名（默认metadata.db）
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        db_name: str = "metadata.db",
    ):
        """
        初始化元数据存储

        Args:
            data_dir: 数据目录
            db_name: 数据库文件名
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.db_name = db_name
        self._conn: Optional[sqlite3.Connection] = None
        self._is_initialized = False

        logger.info(f"MetadataStore 初始化: db={db_name}")

    def connect(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        """
        连接到数据库

        Args:
            data_dir: 数据目录（默认使用初始化时的目录）
        """
        if data_dir is None:
            data_dir = self.data_dir

        if data_dir is None:
            raise ValueError("未指定数据目录")

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        db_path = data_dir / self.db_name

        # 连接数据库
        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row  # 使用字典式访问

        # 优化性能
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-64000")  # 64MB缓存
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.execute("PRAGMA foreign_keys = ON") # 开启外键约束支持级联删除

        logger.info(f"连接到数据库: {db_path}")

        # 初始化表结构
        if not self._is_initialized:
            self._initialize_tables()
            self._is_initialized = True

    def close(self) -> None:
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("数据库连接已关闭")

    def _initialize_tables(self) -> None:
        """初始化数据库表结构"""
        cursor = self._conn.cursor()

        # 段落表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paragraphs (
                hash TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                vector_index INTEGER,
                created_at REAL,
                updated_at REAL,
                metadata TEXT,
                source TEXT,
                word_count INTEGER,
                knowledge_type TEXT DEFAULT 'mixed'
            )
        """)

        # 实体表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                hash TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                vector_index INTEGER,
                appearance_count INTEGER DEFAULT 1,
                created_at REAL,
                metadata TEXT
            )
        """)

        # 关系表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                hash TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                vector_index INTEGER,
                confidence REAL DEFAULT 1.0,
                created_at REAL,
                source_paragraph TEXT,
                metadata TEXT,
                UNIQUE(subject, predicate, object)
            )
        """)

        # 三元组与段落的关联表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paragraph_relations (
                paragraph_hash TEXT NOT NULL,
                relation_hash TEXT NOT NULL,
                PRIMARY KEY (paragraph_hash, relation_hash),
                FOREIGN KEY (paragraph_hash) REFERENCES paragraphs(hash) ON DELETE CASCADE,
                FOREIGN KEY (relation_hash) REFERENCES relations(hash) ON DELETE CASCADE
            )
        """)

        # 实体与段落的关联表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paragraph_entities (
                paragraph_hash TEXT NOT NULL,
                entity_hash TEXT NOT NULL,
                mention_count INTEGER DEFAULT 1,
                PRIMARY KEY (paragraph_hash, entity_hash),
                FOREIGN KEY (paragraph_hash) REFERENCES paragraphs(hash) ON DELETE CASCADE,
                FOREIGN KEY (entity_hash) REFERENCES entities(hash) ON DELETE CASCADE
            )
        """)

        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_paragraphs_vector
            ON paragraphs(vector_index)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_vector
            ON entities(vector_index)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relations_vector
            ON relations(vector_index)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relations_subject
            ON relations(subject)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relations_object
            ON relations(object)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_name
            ON entities(name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_paragraphs_source
            ON paragraphs(source)
        """)

        self._conn.commit()
        logger.debug("数据库表结构初始化完成")
        
        # 执行schema迁移
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """执行数据库schema迁移"""
        cursor = self._conn.cursor()
        
        # 检查paragraphs表是否有knowledge_type列
        cursor.execute("PRAGMA table_info(paragraphs)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "knowledge_type" not in columns:
            logger.info("检测到旧版schema，正在迁移添加knowledge_type字段...")
            try:
                cursor.execute("""
                    ALTER TABLE paragraphs 
                    ADD COLUMN knowledge_type TEXT DEFAULT 'mixed'
                """)
                self._conn.commit()
                logger.info("Schema迁移完成：已添加knowledge_type字段")
            except sqlite3.OperationalError as e:
                logger.warning(f"Schema迁移失败（可能已存在）: {e}")

        # 检查paragraphs表是否有is_permanent列
        cursor.execute("PRAGMA table_info(paragraphs)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "is_permanent" not in columns:
            logger.info("正在迁移: 添加记忆动态字段...")
            try:
                # 段落表
                cursor.execute("ALTER TABLE paragraphs ADD COLUMN is_permanent BOOLEAN DEFAULT 0")
                cursor.execute("ALTER TABLE paragraphs ADD COLUMN last_accessed REAL")
                cursor.execute("ALTER TABLE paragraphs ADD COLUMN access_count INTEGER DEFAULT 0")
                
                # 关系表
                cursor.execute("ALTER TABLE relations ADD COLUMN is_permanent BOOLEAN DEFAULT 0")
                cursor.execute("ALTER TABLE relations ADD COLUMN last_accessed REAL")
                cursor.execute("ALTER TABLE relations ADD COLUMN access_count INTEGER DEFAULT 0")
                
                self._conn.commit()
                logger.info("Schema迁移完成：已添加记忆动态字段")
            except sqlite3.OperationalError as e:
                logger.warning(f"Schema迁移失败: {e}")

        # 检查relations表是否有is_inactive列 (V5 Memory System)
        cursor.execute("PRAGMA table_info(relations)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "is_inactive" not in columns:
            logger.info("正在迁移: 添加V5记忆动态字段 (inactive, protected)...")
            try:
                # 关系表 V5 新增字段
                cursor.execute("ALTER TABLE relations ADD COLUMN is_inactive BOOLEAN DEFAULT 0")
                cursor.execute("ALTER TABLE relations ADD COLUMN inactive_since REAL")
                cursor.execute("ALTER TABLE relations ADD COLUMN is_pinned BOOLEAN DEFAULT 0")
                cursor.execute("ALTER TABLE relations ADD COLUMN protected_until REAL")
                cursor.execute("ALTER TABLE relations ADD COLUMN last_reinforced REAL")
                
                # 为回收站创建 deleted_relations 表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS deleted_relations (
                        hash TEXT PRIMARY KEY,
                        subject TEXT NOT NULL,
                        predicate TEXT NOT NULL,
                        object TEXT NOT NULL,
                        vector_index INTEGER,
                        confidence REAL DEFAULT 1.0,
                        created_at REAL,
                        source_paragraph TEXT,
                        metadata TEXT,
                        is_permanent BOOLEAN DEFAULT 0,
                        last_accessed REAL,
                        access_count INTEGER DEFAULT 0,
                        is_inactive BOOLEAN DEFAULT 0,
                        inactive_since REAL,
                        is_pinned BOOLEAN DEFAULT 0,
                        protected_until REAL,
                        last_reinforced REAL,
                        deleted_at REAL  -- 用于记录删除时间的额外列
                    )
                """)
                
                self._conn.commit()
                logger.info("Schema迁移完成：已添加V5记忆动态字段及回收站表")
            except sqlite3.OperationalError as e:
                logger.warning(f"Schema迁移失败 (V5): {e}")

        # 检查 entities 表是否有 is_deleted 列 (Soft Delete System)
        cursor.execute("PRAGMA table_info(entities)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if "is_deleted" not in columns:
            logger.info("正在迁移: 添加软删除字段 (Soft Delete)...")
            try:
                # 实体表
                cursor.execute("ALTER TABLE entities ADD COLUMN is_deleted INTEGER DEFAULT 0")
                cursor.execute("ALTER TABLE entities ADD COLUMN deleted_at REAL")
                
                # 段落表
                cursor.execute("ALTER TABLE paragraphs ADD COLUMN is_deleted INTEGER DEFAULT 0")
                cursor.execute("ALTER TABLE paragraphs ADD COLUMN deleted_at REAL")
                
                self._conn.commit()
                logger.info("Schema迁移完成：已添加软删除字段")
            except sqlite3.OperationalError as e:
                logger.warning(f"Schema迁移失败 (Soft Delete): {e}")

        # 数据修复: 检查是否存在 source/vector_index 列错位的情况
        # 症状: vector_index (本应是int) 变成了文件名字符串, source (本应是文件名) 变成了类型字符串
        try:
            cursor.execute("""
                SELECT count(*) FROM paragraphs 
                WHERE typeof(vector_index) = 'text' 
                AND source IN ('mixed', 'factual', 'narrative', 'structured', 'auto')
            """)
            count = cursor.fetchone()[0]
            if count > 0:
                logger.warning(f"检测到 {count} 条数据存在列错位（文件名误存入vector_index），正在自动修复...")
                cursor.execute("""
                    UPDATE paragraphs
                    SET 
                        knowledge_type = source,
                        source = vector_index,
                        vector_index = NULL
                    WHERE typeof(vector_index) = 'text' 
                    AND source IN ('mixed', 'factual', 'narrative', 'structured', 'auto')
                """)
                self._conn.commit()
                logger.info(f"自动修复完成: 已校正 {cursor.rowcount} 条数据")
        except Exception as e:
            logger.error(f"数据自动修复失败: {e}")

    def add_paragraph(
        self,
        content: str,
        vector_index: Optional[int] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        knowledge_type: str = "mixed",
    ) -> str:
        """
        添加段落

        Args:
            content: 段落内容
            vector_index: 向量索引
            source: 来源
            metadata: 额外元数据
            knowledge_type: 知识类型 (structured/narrative/factual/mixed)

        Returns:
            段落哈希值
        """
        content_normalized = normalize_text(content)
        hash_value = compute_hash(content_normalized)

        now = datetime.now().timestamp()
        word_count = len(content_normalized.split())

        cursor = self._conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO paragraphs
                (hash, content, vector_index, created_at, updated_at, metadata, source, word_count, knowledge_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                hash_value,
                content,
                vector_index,
                now,
                now,
                pickle.dumps(metadata or {}),
                source,
                word_count,
                knowledge_type,
            ))
            self._conn.commit()
            logger.debug(f"添加段落: hash={hash_value[:16]}..., words={word_count}, type={knowledge_type}")
            return hash_value
        except sqlite3.IntegrityError:
            logger.debug(f"段落已存在: {hash_value[:16]}...")
            # 尝试复活
            self.revive_if_deleted(paragraph_hashes=[hash_value])
            return hash_value

    def _canonicalize_name(self, name: str) -> str:
        """
        规范化名称 (统一小写并去除首尾空格)
        
        Args:
            name: 原始名称
            
        Returns:
            规范化后的名称
        """
        if not name:
            return ""
        return name.strip().lower()

    def add_entity(
        self,
        name: str,
        vector_index: Optional[int] = None,
        source_paragraph: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        添加实体
        
        Args:
            name: 实体名称
            vector_index: 向量索引
            source_paragraph: 来源段落哈希 (如果提供，将建立关联)
            metadata: 额外元数据
            
        Returns:
            实体哈希值
        """
        # 1. 规范化名称
        name_normalized = self._canonicalize_name(name)
        if not name_normalized:
            raise ValueError("Entity name cannot be empty")
            
        hash_value = compute_hash(name_normalized)
        now = datetime.now().timestamp()

        cursor = self._conn.cursor()
        
        # 2. 插入实体 (INSERT OR IGNORE)
        # 注意：这里我们保留原有的 name 字段存储，可以是 display name，
        # 但 hash 必须由 canonical name 生成。
        # 如果实体已存在，我们其实不一定要更新 name (保留第一次的 display name 往往更好)
        # 或者我们也可以选择不作为唯一键冲突，而是逻辑判断。
        # 考虑到 entities.hash 是主键，entities.name 是 UNIQUE。
        # 如果 name 大小写不同但 hash 相同 (冲突)，或者 name 不同但 canonical name 相同?
        # 由于 hash 是由 canonical name 算出来的，所以 hash 相同意味着 canonical name 相同。
        # 如果 db 中已存在的 name 是 "Apple"，新来的 name 是 "apple"，它们 canonical name 都是 "apple"，hash 一样。
        # 此时 INSERT OR IGNORE 会忽略。
        
        try:
            cursor.execute("""
                INSERT INTO entities
                (hash, name, vector_index, appearance_count, created_at, metadata)
                VALUES (?, ?, ?, 1, ?, ?)
            """, (
                hash_value,
                name,
                vector_index,
                now,
                pickle.dumps(metadata or {}),
            ))
            
            logger.debug(f"添加实体: {name} ({hash_value[:8]})")
            self._conn.commit()
            
            # 3. 建立来源关联
            if source_paragraph:
                self.link_paragraph_entity(source_paragraph, hash_value)
                
            return hash_value
            
        except sqlite3.IntegrityError:
            # 实体已存在
            # 1. 尝试复活 (自动复活)
            self.revive_if_deleted(entity_hashes=[hash_value])
            
            # 2. 更新计数
            cursor.execute("""
                UPDATE entities
                SET appearance_count = appearance_count + 1
                WHERE hash = ?
            """, (hash_value,))
            self._conn.commit()
            
            logger.debug(f"实体已存在(复活/计数+1): {name}")
            
            # 3. 建立来源关联
            if source_paragraph:
                self.link_paragraph_entity(source_paragraph, hash_value)
                
            return hash_value

    def add_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        vector_index: Optional[int] = None,
        confidence: float = 1.0,
        source_paragraph: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        添加关系
        
        Args:
            subject: 主语
            predicate: 谓语
            obj: 宾语
            vector_index: 向量索引
            confidence: 置信度
            source_paragraph: 来源段落哈希
            metadata: 额外元数据
            
        Returns:
            关系哈希值
        """
        # 1. 规范化输入
        s_canon = self._canonicalize_name(subject)
        p_canon = self._canonicalize_name(predicate)
        o_canon = self._canonicalize_name(obj)
        
        if not all([s_canon, p_canon, o_canon]):
             raise ValueError("Relation components cannot be empty")

        # 2. 计算组合哈希
        # 公式: md5(s|p|o)
        relation_key = f"{s_canon}|{p_canon}|{o_canon}"
        hash_value = compute_hash(relation_key)

        now = datetime.now().timestamp()
        
        # 记录原始 display name 到 metadata (如果需要的话，或者直接存到 DB 字段)
        # 这里我们直接存入 subject, predicate, object 字段，
        # 注意：如果 DB 里已存在该关系 (hash 相同)，则不会更新这些字段，保留第一次的拼写。
        
        cursor = self._conn.cursor()
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO relations
                (hash, subject, predicate, object, vector_index, confidence, created_at, source_paragraph, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                hash_value,
                subject,  # 原始拼写
                predicate,
                obj,
                vector_index,
                confidence,
                now,
                source_paragraph, # 这里的 source_paragraph 仅作为 "首次发现地" 记录，也可留空
                pickle.dumps(metadata or {}),
            ))
            self._conn.commit()
            
            if cursor.rowcount > 0:
                logger.debug(f"添加关系: {subject} -{predicate}-> {obj}")
            else:
                logger.debug(f"关系已存在: {subject} -{predicate}-> {obj}")

            # 3. 建立来源关联 (幂等)
            # 无论关系是新创建的还是已存在的，只要提供了 source_paragraph，都要建立连接
            if source_paragraph:
                self.link_paragraph_relation(source_paragraph, hash_value)
                
            return hash_value
            
        except sqlite3.IntegrityError as e:
            logger.warning(f"添加关系异常: {e}")
            return hash_value

    def link_paragraph_relation(
        self,
        paragraph_hash: str,
        relation_hash: str,
    ) -> bool:
        """
        关联段落和关系 (幂等)
        """
        cursor = self._conn.cursor()
        try:
            # 使用 INSERT OR IGNORE 避免重复报错
            cursor.execute("""
                INSERT OR IGNORE INTO paragraph_relations
                (paragraph_hash, relation_hash)
                VALUES (?, ?)
            """, (paragraph_hash, relation_hash))
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def link_paragraph_entity(
        self,
        paragraph_hash: str,
        entity_hash: str,
        mention_count: int = 1,
    ) -> bool:
        """
        关联段落和实体 (幂等)
        """
        cursor = self._conn.cursor()
        try:
            # 首先尝试插入
            cursor.execute("""
                INSERT OR IGNORE INTO paragraph_entities
                (paragraph_hash, entity_hash, mention_count)
                VALUES (?, ?, ?)
            """, (paragraph_hash, entity_hash, mention_count))
            
            if cursor.rowcount == 0:
                # 如果已存在 (IGNORE生效)，则更新计数
                cursor.execute("""
                    UPDATE paragraph_entities
                    SET mention_count = mention_count + ?
                    WHERE paragraph_hash = ? AND entity_hash = ?
                """, (mention_count, paragraph_hash, entity_hash))
            
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_paragraph(self, hash_value: str) -> Optional[Dict[str, Any]]:
        """
        获取段落

        Args:
            hash_value: 段落哈希

        Returns:
            段落信息字典，不存在则返回None
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT * FROM paragraphs WHERE hash = ?
        """, (hash_value,))
        row = cursor.fetchone()

        if row:
            return self._row_to_dict(row, "paragraph")
        return None

    def get_entity(self, hash_value: str) -> Optional[Dict[str, Any]]:
        """
        获取实体

        Args:
            hash_value: 实体哈希

        Returns:
            实体信息字典，不存在则返回None
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT * FROM entities WHERE hash = ?
        """, (hash_value,))
        row = cursor.fetchone()

        if row:
            return self._row_to_dict(row, "entity")
        return None

    def get_relation(self, hash_value: str) -> Optional[Dict[str, Any]]:
        """
        获取关系

        Args:
            hash_value: 关系哈希

        Returns:
            关系信息字典，不存在则返回None
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT * FROM relations WHERE hash = ?
        """, (hash_value,))
        row = cursor.fetchone()

        if row:
            return self._row_to_dict(row, "relation")
        return None

    def get_paragraph_relations(self, paragraph_hash: str) -> List[Dict[str, Any]]:
        """
        获取段落的所有关系

        Args:
            paragraph_hash: 段落哈希

        Returns:
            关系列表
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT r.* FROM relations r
            JOIN paragraph_relations pr ON r.hash = pr.relation_hash
            WHERE pr.paragraph_hash = ?
        """, (paragraph_hash,))

        return [self._row_to_dict(row, "relation") for row in cursor.fetchall()]

    def get_paragraph_entities(self, paragraph_hash: str) -> List[Dict[str, Any]]:
        """
        获取段落的所有实体

        Args:
            paragraph_hash: 段落哈希

        Returns:
            实体列表
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT e.*, pe.mention_count
            FROM entities e
            JOIN paragraph_entities pe ON e.hash = pe.entity_hash
            WHERE pe.paragraph_hash = ?
        """, (paragraph_hash,))

        return [self._row_to_dict(row, "entity") for row in cursor.fetchall()]

    def get_paragraphs_by_entity(self, entity_name: str) -> List[Dict[str, Any]]:
        """
        获取包含指定实体的所有段落 (自动处理规范化)
        
        Args:
            entity_name: 实体名称 (支持任意大小写)
            
        Returns:
            段落列表
        """
        # 1. 计算规范化 Hash
        name_canon = self._canonicalize_name(entity_name)
        if not name_canon:
            return []
            
        entity_hash = compute_hash(name_canon)
        
        cursor = self._conn.cursor()
        # 2. 直接使用 Hash 查询中间表，完全避开 Name 匹配
        cursor.execute("""
            SELECT p.*
            FROM paragraphs p
            JOIN paragraph_entities pe ON p.hash = pe.paragraph_hash
            WHERE pe.entity_hash = ?
        """, (entity_hash,))

        return [self._row_to_dict(row, "paragraph") for row in cursor.fetchall()]

    def get_relations(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        查询关系（大小写不敏感）
        
        Args:
            subject: 主语（可选）
            predicate: 谓语（可选）
            object: 宾语（可选）
            
        Returns:
            关系列表
        """
        # 构建查询条件
        conditions = []
        params = []
        
        if subject:
            conditions.append("LOWER(subject) = ?")
            params.append(self._canonicalize_name(subject))
        if predicate:
            conditions.append("LOWER(predicate) = ?")
            params.append(self._canonicalize_name(predicate))
        if object:
            conditions.append("LOWER(object) = ?")
            params.append(self._canonicalize_name(object))
            
        sql = "SELECT * FROM relations"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
            
        cursor = self._conn.cursor()
        cursor.execute(sql, tuple(params))
        
        return [self._row_to_dict(row, "relation") for row in cursor.fetchall()]

    def get_all_triples(self) -> List[Tuple[str, str, str, str]]:
        """
        高效获取所有三元组 (subject, predicate, object, hash)
        直接返回元组，跳过字典转换和pickle反序列化，用于构建 V5 Map 缓存。
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT subject, predicate, object, hash FROM relations")
        return list(cursor.fetchall())

    def get_relation_status_batch(self, hashes: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        批量获取关系的状态信息 (V5 Memory)
        
        Args:
            hashes: 关系哈希列表
            
        Returns:
            Dict[hash, status_dict]
            status_dict 包含: is_inactive, is_pinned, protected_until, last_reinforced
        """
        if not hashes:
            return {}
            
        # SQLite 变量限制通常为 999 个，如有需要请分割。
        # 此处假设批大小合理或实施分割逻辑？
        # 简单起见，使用 500 个一组的切片
        
        result = {}
        chunk_size = 500
        
        cursor = self._conn.cursor()
        
        for i in range(0, len(hashes), chunk_size):
            chunk = hashes[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            sql = f"""
                SELECT hash, is_inactive, is_pinned, protected_until, last_reinforced
                FROM relations
                WHERE hash IN ({placeholders})
            """
            
            cursor.execute(sql, chunk)
            rows = cursor.fetchall()
            
            for row in rows:
                result[row['hash']] = {
                    "is_inactive": bool(row['is_inactive']),
                    "is_pinned": bool(row['is_pinned']),
                    "protected_until": row['protected_until'],
                    "last_reinforced": row['last_reinforced']
                }
                
        return result

    def get_paragraphs_by_relation(self, relation_hash: str) -> List[Dict[str, Any]]:
        """
        获取支持指定关系的所有段落

        Args:
            relation_hash: 关系哈希

        Returns:
            段落列表
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT p.*
            FROM paragraphs p
            JOIN paragraph_relations pr ON p.hash = pr.paragraph_hash
            WHERE pr.relation_hash = ?
        """, (relation_hash,))

        return [self._row_to_dict(row, "paragraph") for row in cursor.fetchall()]

    def get_paragraphs_by_source(self, source: str) -> List[Dict[str, Any]]:
        """
        按来源获取段落

        Args:
            source: 来源标识符

        Returns:
            段落列表
        """
        return self.query("SELECT * FROM paragraphs WHERE source = ?", (source,))

    def get_all_sources(self) -> List[Dict[str, Any]]:
        """
        获取所有来源文件统计信息
        
        Returns:
            来源列表 [{'source': 'name', 'count': int, 'last_updated': timestamp}]
        """
        cursor = self._conn.cursor()
        # 排除 source 为 NULL 或空的记录
        cursor.execute("""
            SELECT source, COUNT(*) as count, MAX(created_at) as last_updated 
            FROM paragraphs 
            WHERE source IS NOT NULL AND source != ''
            GROUP BY source
            ORDER BY last_updated DESC
        """)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "source": row[0],
                "count": row[1],
                "last_updated": row[2]
            })
        return results


    def search_paragraphs_by_content(self, content_query: str) -> List[Dict[str, Any]]:
        """按内容模糊搜索段落"""
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT * FROM paragraphs WHERE content LIKE ?
        """, (f"%{content_query}%",))
        return [self._row_to_dict(row, "paragraph") for row in cursor.fetchall()]

    def delete_paragraph(self, hash_value: str) -> bool:
        """
        删除段落（级联删除相关关联）

        Args:
            hash_value: 段落哈希

        Returns:
            是否成功删除
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            DELETE FROM paragraphs WHERE hash = ?
        """, (hash_value,))
        self._conn.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"删除段落: {hash_value[:16]}...")

        return deleted

    def delete_entity(self, hash_or_name: str) -> bool:
        """
        删除实体（级联删除相关关联）
        支持通过哈希值或名称删除
        
        注意：会同时删除所有引用该实体（作为主语或宾语）的关系
        """
        cursor = self._conn.cursor()
        
        # 1. 解析实体信息 (获取 Name 和 Hash)
        entity_name = None
        entity_hash = None
        
        # 尝试作为 Hash 查询
        cursor.execute("SELECT name, hash FROM entities WHERE hash = ?", (hash_or_name,))
        row = cursor.fetchone()
        if row:
            entity_name = row[0]
            entity_hash = row[1]
        else:
            # 尝试作为 Name 查询 (原始匹配)
            cursor.execute("SELECT name, hash FROM entities WHERE name = ?", (hash_or_name,))
            row = cursor.fetchone()
            if row:
                entity_name = row[0]
                entity_hash = row[1]
            else:
                # 最后的最后：尝试规范化名称 (Canonical) 查询，解决大小写或 WebUI 手动输入导致的不匹配
                name_canon = self._canonicalize_name(hash_or_name)
                canon_hash = compute_hash(name_canon)
                cursor.execute("SELECT name, hash FROM entities WHERE hash = ?", (canon_hash,))
                row = cursor.fetchone()
                if row:
                    entity_name = row[0]
                    entity_hash = row[1]
                
        if not entity_name or not entity_hash:
            logger.debug(f"删除实体请求跳过：未在元数据记录中找到 {hash_or_name}")
            return False

        logger.info(f"开始删除实体: {entity_name} (Hash: {entity_hash[:8]}...)")

        try:
            # 2. 查找相关关系 (Subject 或 Object 为该实体)
            cursor.execute("""
                SELECT hash FROM relations 
                WHERE subject = ? OR object = ?
            """, (entity_name, entity_name))
            
            relation_hashes = [r[0] for r in cursor.fetchall()]
            
            if relation_hashes:
                logger.info(f"发现 {len(relation_hashes)} 个相关关系，准备级联删除")
                
                # 3. 删除这些关系与段落的关联
                # SQLite 不支持直接 DELETE ... WHERE ... IN (...) 的列表参数，需要拼接占位符
                placeholders = ','.join(['?'] * len(relation_hashes))
                
                cursor.execute(f"""
                    DELETE FROM paragraph_relations 
                    WHERE relation_hash IN ({placeholders})
                """, relation_hashes)
                
                # 4. 删除关系本体
                cursor.execute(f"""
                    DELETE FROM relations 
                    WHERE hash IN ({placeholders})
                """, relation_hashes)
                
                logger.info("相关关系已级联删除")
            
            # 5. 删除实体与段落的关联
            cursor.execute("DELETE FROM paragraph_entities WHERE entity_hash = ?", (entity_hash,))
            
            # 6. 删除实体本体
            cursor.execute("DELETE FROM entities WHERE hash = ?", (entity_hash,))
            
            self._conn.commit()
            logger.info("实体删除完成")
            return True
            
        except Exception as e:
            logger.error(f"删除实体时发生错误: {e}")
            self._conn.rollback()
            return False

    def delete_relation(self, hash_value: str) -> bool:
        """
        删除关系（级联删除相关关联）

        Args:
            hash_value: 关系哈希

        Returns:
            是否成功删除
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            DELETE FROM relations WHERE hash = ?
        """, (hash_value,))
        self._conn.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"删除关系: {hash_value[:16]}...")

        return deleted

    def update_vector_index(
        self,
        item_type: str,
        hash_value: str,
        vector_index: int,
    ) -> bool:
        """
        更新向量索引

        Args:
            item_type: 类型（paragraph/entity/relation）
            hash_value: 哈希值
            vector_index: 向量索引

        Returns:
            是否成功更新
        """
        valid_types = ["paragraph", "entity", "relation"]
        if item_type not in valid_types:
            raise ValueError(f"无效的类型: {item_type}")

        table_map = {
            "paragraph": "paragraphs",
            "entity": "entities",
            "relation": "relations",
        }

        cursor = self._conn.cursor()
        cursor.execute(f"""
            UPDATE {table_map[item_type]}
            SET vector_index = ?
            WHERE hash = ?
        """, (vector_index, hash_value))
        self._conn.commit()

        return cursor.rowcount > 0

    def set_permanence(self, hash_value: str, item_type: str, is_permanent: bool) -> bool:
        """设置永久记忆标记"""
        table_map = {
            "paragraph": "paragraphs",
            "relation": "relations",
        }
        if item_type not in table_map:
            raise ValueError(f"类型 {item_type} 不支持设置永久性")
            
        cursor = self._conn.cursor()
        cursor.execute(f"""
            UPDATE {table_map[item_type]}
            SET is_permanent = ?
            WHERE hash = ?
        """, (1 if is_permanent else 0, hash_value))
        self._conn.commit()
        
        if cursor.rowcount > 0:
            logger.debug(f"设置永久记忆: {item_type}/{hash_value[:8]} -> {is_permanent}")
            return True
        return False

    def record_access(self, hash_value: str, item_type: str) -> bool:
        """记录访问（更新时间和次数）"""
        table_map = {
            "paragraph": "paragraphs",
            "relation": "relations",
        }
        if item_type not in table_map:
            return False
            
        now = datetime.now().timestamp()
        cursor = self._conn.cursor()
        cursor.execute(f"""
            UPDATE {table_map[item_type]}
            SET last_accessed = ?, access_count = access_count + 1
            WHERE hash = ?
        """, (now, hash_value))
        self._conn.commit()
        return cursor.rowcount > 0

    def query(
        self,
        sql: str,
        params: Optional[Tuple] = None,
    ) -> List[Dict[str, Any]]:
        """
        执行自定义查询

        Args:
            sql: SQL语句
            params: 参数

        Returns:
            查询结果列表
        """
        cursor = self._conn.cursor()
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)

        return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self) -> Dict[str, int]:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        cursor = self._conn.cursor()

        stats = {}

        # 段落数量
        cursor.execute("SELECT COUNT(*) FROM paragraphs")
        stats["paragraph_count"] = cursor.fetchone()[0]

        # 实体数量
        cursor.execute("SELECT COUNT(*) FROM entities")
        stats["entity_count"] = cursor.fetchone()[0]

        # 关系数量
        cursor.execute("SELECT COUNT(*) FROM relations")
        stats["relation_count"] = cursor.fetchone()[0]

        # 总词数
        cursor.execute("SELECT SUM(word_count) FROM paragraphs")
        result = cursor.fetchone()[0]
        stats["total_words"] = result if result else 0

        return stats

    def count_paragraphs(self, include_deleted: bool = False, only_deleted: bool = False) -> int:
        """
        获取段落数量
        """
        # 段落表目前由于级联删除是硬删除，此处仅为接口兼容
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM paragraphs")
        return cursor.fetchone()[0]

    def count_relations(self, include_deleted: bool = False, only_deleted: bool = False) -> int:
        """
        获取关系数量
        """
        # 关系表目前也是级联硬删除，此处仅为接口兼容
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM relations")
        return cursor.fetchone()[0]

    def count_entities(self) -> int:
        """
        获取实体数量

        Returns:
            实体数量
        """
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM entities")
        return cursor.fetchone()[0]

    def delete_paragraph_atomic(self, paragraph_hash: str) -> Dict[str, Any]:
        """
        两阶段删除段落：DB 事务内计算 + 提交后执行清理

        Args:
            paragraph_hash: 段落哈希

        Returns:
            cleanup_plan: 包含需要后续从 Vector/GraphStore 中移除的 ID 列表
        """
        cleanup_plan = {
            "paragraph_hash": paragraph_hash,
            "vector_id_to_remove": None,
            "edges_to_remove": [],  # (src, tgt) 元组列表 (fallback)
            "relation_prune_ops": []  # (subject, object, relation_hash) 精准裁剪
        }

        cursor = self._conn.cursor()
        try:
            # === Phase 1: DB Transaction (可回滚) ===
            # 使用 IMMEDIATE 模式，一旦开启事务立即锁定 DB (防止其他写操作插队导致幻读)
            cursor.execute("BEGIN IMMEDIATE")

            # 1. [快照] 获取候选关系
            cursor.execute("SELECT relation_hash FROM paragraph_relations WHERE paragraph_hash = ?", (paragraph_hash,))
            candidate_relations = [row[0] for row in cursor.fetchall()]

            # 2. [快照] 确认该段落存在并记录 ID 用于向量删除
            cursor.execute("SELECT hash FROM paragraphs WHERE hash = ?", (paragraph_hash,))
            if cursor.fetchone():
                cleanup_plan["vector_id_to_remove"] = paragraph_hash

            # 3. [主删除] 删除段落 (触发 CASCADE 删 paragraph_relations)
            cursor.execute("DELETE FROM paragraphs WHERE hash = ?", (paragraph_hash,))

            # 4. [计算孤儿]
            orphaned_hashes = []
            for rel_hash in candidate_relations:
                count = cursor.execute(
                    "SELECT count(*) FROM paragraph_relations WHERE relation_hash = ?",
                    (rel_hash,)
                ).fetchone()[0]

                if count == 0:
                    # 是孤儿：记录边信息以便后续删 Graph
                    cursor.execute("SELECT subject, object FROM relations WHERE hash = ?", (rel_hash,))
                    rel_info = cursor.fetchone()
                    if rel_info:
                        s_val, o_val = rel_info[0], rel_info[1]
                        cleanup_plan["relation_prune_ops"].append((s_val, o_val, rel_hash))

                        # 仅当 (subject, object) 不再有任何关系时，才计划删整条边（兼容旧实现）。
                        sibling_count = cursor.execute(
                            """
                            SELECT count(*) FROM relations
                            WHERE LOWER(TRIM(subject)) = LOWER(TRIM(?))
                              AND LOWER(TRIM(object)) = LOWER(TRIM(?))
                              AND hash != ?
                            """,
                            (s_val, o_val, rel_hash)
                        ).fetchone()[0]
                        if sibling_count == 0:
                            cleanup_plan["edges_to_remove"].append((s_val, o_val))

                    orphaned_hashes.append(rel_hash)

            # 5. [DB清理] 删除孤儿关系记录
            if orphaned_hashes:
                placeholders = ','.join(['?'] * len(orphaned_hashes))
                cursor.execute(f"DELETE FROM relations WHERE hash IN ({placeholders})", orphaned_hashes)

            self._conn.commit()
            if cleanup_plan["vector_id_to_remove"]:
                logger.debug(f"原子删除段落成功: {paragraph_hash}, 计划清理 {len(orphaned_hashes)} 个孤儿关系")
            return cleanup_plan

        except Exception as e:
            self._conn.rollback()
            logger.error(f"DB Transaction failed: {e}")
            raise e


    def clear_all(self) -> None:
        """清空所有表数据"""
        cursor = self._conn.cursor()
        tables = [
            "paragraphs", "entities", "relations", 
            "paragraph_relations", "paragraph_entities"
        ]
        for table in tables:
            cursor.execute(f"DELETE FROM {table}")
        self._conn.commit()
        logger.info("元数据存储所有表已清空")



    def update_relation_timestamp(self, hash_value: str, access_count_delta: int = 1) -> None:
        """更新关系的访问时间和计数"""
        now = datetime.now().timestamp()
        
        # 同时更新 last_accessed (旧) 和 last_reinforced (V5)
        
        cursor = self._conn.cursor()
        cursor.execute("""
            UPDATE relations
            SET last_accessed = ?,
                access_count = access_count + ?
            WHERE hash = ?
        """, (now, access_count_delta, hash_value))
        self._conn.commit()

    # =========================================================================
    # V5 Memory System Methods
    # =========================================================================

    def get_relation_status_batch(self, hashes: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        批量获取关系状态 (V5)
        
        Args:
            hashes: 关系哈希列表
            
        Returns:
            Dict[hash, status_dict]
            status_dict 包含: is_inactive, weight(confidence), is_pinned, protected_until, last_reinforced, inactive_since
        """
        if not hashes:
            return {}
            
        placeholders = ",".join(["?"] * len(hashes))
        cursor = self._conn.cursor()
        cursor.execute(f"""
            SELECT hash, is_inactive, confidence, is_pinned, protected_until, last_reinforced, inactive_since
            FROM relations
            WHERE hash IN ({placeholders})
        """, hashes)
        
        result = {}
        for row in cursor.fetchall():
            result[row["hash"]] = {
                "is_inactive": bool(row["is_inactive"]),
                "weight": row["confidence"],
                "is_pinned": bool(row["is_pinned"]),
                "protected_until": row["protected_until"],
                "last_reinforced": row["last_reinforced"],
                "inactive_since": row["inactive_since"]
            }
        return result

    def mark_relations_inactive(self, hashes: List[str], inactive_since: float) -> None:
        """
        批量标记关系为冷冻 (Inactive)
        """
        if not hashes:
            return
            
        placeholders = ",".join(["?"] * len(hashes))
        cursor = self._conn.cursor()
        cursor.execute(f"""
            UPDATE relations
            SET is_inactive = 1,
                inactive_since = ?
            WHERE hash IN ({placeholders})
        """, (inactive_since, *hashes))
        self._conn.commit()
        
    def mark_relations_active(self, hashes: List[str], boost_weight: Optional[float] = None) -> None:
        """
        批量标记关系为活跃 (Active/Revive)
        
        Args:
            hashes: 关系哈希列表
            boost_weight: 如果提供，将设置 confidence = max(confidence, boost_weight)
        """
        if not hashes:
            return
            
        placeholders = ",".join(["?"] * len(hashes))
        cursor = self._conn.cursor()
        
        if boost_weight is not None:
            cursor.execute(f"""
                UPDATE relations
                SET is_inactive = 0,
                    inactive_since = NULL,
                    confidence = MAX(confidence, ?)
                WHERE hash IN ({placeholders})
            """, (boost_weight, *hashes))
        else:
             cursor.execute(f"""
                UPDATE relations
                SET is_inactive = 0,
                    inactive_since = NULL
                WHERE hash IN ({placeholders})
            """, hashes)
            
        self._conn.commit()

    def update_relations_protection(
        self, 
        hashes: List[str], 
        protected_until: Optional[float] = None, 
        is_pinned: Optional[bool] = None,
        last_reinforced: Optional[float] = None
    ) -> None:
        """
        批量更新关系保护状态
        """
        if not hashes:
            return
            
        updates = []
        params = []
        
        if protected_until is not None:
            updates.append("protected_until = ?")
            params.append(protected_until)
        if is_pinned is not None:
            updates.append("is_pinned = ?")
            params.append(1 if is_pinned else 0)
        if last_reinforced is not None:
            updates.append("last_reinforced = ?")
            params.append(last_reinforced)
            
        if not updates:
            return

        sql_set = ", ".join(updates)
        placeholders = ",".join(["?"] * len(hashes))
        
        params.extend(hashes)
        
        cursor = self._conn.cursor()
        cursor.execute(f"""
            UPDATE relations
            SET {sql_set}
            WHERE hash IN ({placeholders})
        """, params)
        self._conn.commit()

    def get_prune_candidates(self, cutoff_time: float, limit: int = 1000) -> List[str]:
        """
        获取待修剪候选 (已过冷冻保留期)
        
        Args:
            cutoff_time: 截止时间 (now - 冷冻时长)
            limit: 限制数量
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT hash FROM relations
            WHERE is_inactive = 1 
            AND inactive_since < ?
            LIMIT ?
        """, (cutoff_time, limit))
        return [row[0] for row in cursor.fetchall()]

    def backup_and_delete_relations(self, hashes: List[str]) -> int:
        """
        备份并删除关系 (Prune)
        
        Returns:
            删除的数量
        """
        if not hashes:
            return 0
            
        placeholders = ",".join(["?"] * len(hashes))
        now = datetime.now().timestamp()
        
        cursor = self._conn.cursor()
        try:
            # 1. 备份
            cursor.execute(f"""
                INSERT OR REPLACE INTO deleted_relations 
                (hash, subject, predicate, object, vector_index, confidence, created_at, 
                 source_paragraph, metadata, is_permanent, last_accessed, access_count,
                 is_inactive, inactive_since, is_pinned, protected_until, last_reinforced, deleted_at)
                SELECT 
                 hash, subject, predicate, object, vector_index, confidence, created_at, 
                 source_paragraph, metadata, is_permanent, last_accessed, access_count,
                 is_inactive, inactive_since, is_pinned, protected_until, last_reinforced, ?
                FROM relations
                WHERE hash IN ({placeholders})
            """, (now, *hashes))
            
            # 2. 删除 (级联删除会自动处理 paragraph_relations 关联)
            cursor.execute(f"""
                DELETE FROM relations
                WHERE hash IN ({placeholders})
            """, hashes)
            
            deleted_count = cursor.rowcount
            self._conn.commit()
            return deleted_count
            
        except Exception as e:
            logger.error(f"备份删除失败: {e}")
            self._conn.rollback()
            return 0

    def restore_relation_metadata(self, hash_value: str) -> Optional[Dict[str, Any]]:
        """
        从回收站恢复关系元数据
        
        Returns:
            恢复后的关系数据 (字典)，失败返回 None
        """
        cursor = self._conn.cursor()
        try:
            # 1. 查询备份数据
            cursor.execute("SELECT * FROM deleted_relations WHERE hash = ?", (hash_value,))
            row = cursor.fetchone()
            if not row:
                return None
                
            data = dict(row)
            # 移除 deleted_at 字段
            if "deleted_at" in data:
                del data["deleted_at"]
                
            # 2. 插入回 relations 表
            # 动态构建 SQL 以适应字段变化
            columns = list(data.keys())
            placeholders = ",".join(["?"] * len(columns))
            cols_str = ",".join(columns)
            values = list(data.values())
            
            cursor.execute(f"""
                INSERT OR REPLACE INTO relations ({cols_str})
                VALUES ({placeholders})
            """, values)
            
            # 3. 从备份表删除
            cursor.execute("DELETE FROM deleted_relations WHERE hash = ?", (hash_value,))
            
            self._conn.commit()
            return self._row_to_dict(row, "relation") # 使用助手函数将原始行转换为字典
            
        except Exception as e:
            logger.error(f"恢复关系失败: {hash_value} - {e}")
            self._conn.rollback()
            return None
            
    def get_protected_relations_hashes(self) -> List[str]:
        """获取所有受保护关系的哈希 (Pinned 或 Protected Until > Now)"""
        now = datetime.now().timestamp()
        
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT hash FROM relations
            WHERE is_pinned = 1 OR protected_until > ?
        """, (now,))
        
        return [row[0] for row in cursor.fetchall()]


    
    def get_deleted_relations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取回收站中的关系记录"""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM deleted_relations ORDER BY deleted_at DESC LIMIT ?", (limit,))
        data = []
        for row in cursor.fetchall():
             d = dict(row)
             # 是否需要解码元数据？是的，与普通行相同
             if "metadata" in d and d["metadata"]:
                 try:
                     d["metadata"] = pickle.loads(d["metadata"])
                 except Exception:
                     d["metadata"] = {}
             data.append(d)
        return data

    def get_deleted_relation(self, hash_value: str) -> Optional[Dict[str, Any]]:
        """获取单条回收站记录"""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM deleted_relations WHERE hash = ?", (hash_value,))
        row = cursor.fetchone()
        if not row: return None
        
        d = dict(row)
        if "metadata" in d and d["metadata"]:
             try:
                 d["metadata"] = pickle.loads(d["metadata"])
             except Exception:
                 d["metadata"] = {}
        return d

    def reinforce_relations(self, hashes: List[str]) -> None:
        """强化关系 (更新 last_reinforced, is_inactive=0)"""
        if not hashes: return
        now = datetime.now().timestamp()
        
        cursor = self._conn.cursor()
        # Batch update? chunking
        chunk_size = 500
        for i in range(0, len(hashes), chunk_size):
            chunk = hashes[i:i+chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            sql = f"""
                UPDATE relations 
                SET last_reinforced = ?, is_inactive = 0, inactive_since = NULL
                WHERE hash IN ({placeholders})
            """
            cursor.execute(sql, [now] + chunk)
            
        self._conn.commit()

    def mark_relations_inactive(self, hashes: List[str]) -> None:
        """标记关系为非活跃 (Freeze)"""
        if not hashes: return
        now = datetime.now().timestamp()
        
        cursor = self._conn.cursor()
        chunk_size = 500
        for i in range(0, len(hashes), chunk_size):
            chunk = hashes[i:i+chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            sql = f"""
                UPDATE relations 
                SET is_inactive = 1, inactive_since = ?
                WHERE hash IN ({placeholders})
            """
            cursor.execute(sql, [now] + chunk)
            
        self._conn.commit()

    def protect_relations(
        self, 
        hashes: List[str], 
        is_pinned: bool = False, 
        ttl_seconds: float = 0
    ) -> None:
        """
        设置保护状态
        """
        if not hashes: return
        now = datetime.now().timestamp()
        protected_until = (now + ttl_seconds) if ttl_seconds > 0 else 0
        
        cursor = self._conn.cursor()
        chunk_size = 500
        for i in range(0, len(hashes), chunk_size):
            chunk = hashes[i:i+chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            
            # 由于 is_pinned 和 protected_until 是分开的，如果请求固定（pin），我们会同时更新这两项，
            # 但通常用户要么切换固定状态，要么设置 TTL。
            # 如果 is_pinned=True，TTL 通常就不重要了。
            # 但目前的逻辑是正交处理它们的。
            
            # 如果用户取消固定 (is_pinned=False)，我们是否应该尊重已设置的 TTL？
            # 当前的 API 会同时设置这两项。
            
            sql = f"""
                UPDATE relations 
                SET is_pinned = ?, protected_until = ?
                WHERE hash IN ({placeholders})
            """
            cursor.execute(sql, [is_pinned, protected_until] + chunk)
            
        self._conn.commit()

    def vacuum(self) -> None:
        """优化数据库"""
        cursor = self._conn.cursor()
        cursor.execute("VACUUM")
        self._conn.commit()
        logger.info("数据库优化完成")

    def _row_to_dict(self, row: sqlite3.Row, row_type: str) -> Dict[str, Any]:
        """
        将数据库行转换为字典

        Args:
            row: 数据库行
            row_type: 行类型

        Returns:
            字典
        """
        d = dict(row)

        # 解码pickle字段
        if "metadata" in d and d["metadata"]:
            try:
                d["metadata"] = pickle.loads(d["metadata"])
            except Exception:
                d["metadata"] = {}

        return d

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._conn is not None

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

    # =========================================================================
    # V5 Soft Delete & Garbage Collection
    # =========================================================================

    def get_entity_gc_candidates(self, isolated_hashes: List[str], retention_seconds: float) -> List[str]:
        """
        获取实体 GC 候选列表 (Soft Delete Candidates)
        条件:
        1. 在 isolated_hashes 列表中 (由 GraphStore 提供, 确保 active degree=0 & not frozen)
        2. is_deleted = 0 (未被标记)
        3. created_at < now - retention (过了新手保护期)
        4. 不被任何 active paragraph 引用 (paragraph_entities check)
        
        Args:
            isolated_hashes: 孤儿实体 Hash 列表
            retention_seconds: 保留时间 (秒)
        """
        if not isolated_hashes:
            return []
            
        now = datetime.now().timestamp()
        cutoff = now - retention_seconds
        
        candidates = []
        batch_size = 900
        
        # 分批处理 IN 查询
        for i in range(0, len(isolated_hashes), batch_size):
            batch = isolated_hashes[i:i+batch_size]
            placeholders = ",".join(["?"] * len(batch))
            
            # 使用 NOT EXISTS 子查询检查引用
            # 注意: paragraph_entities 中引用的 paragraph 如果被软删了，是否算引用？
            # 这里的语义: 只要有 rows 存在于 paragraph_entities 且该 row 对应的 paragraph 没被彻底物理删除，就算引用。
            # 更严格: ... OR (EXISTS ... AND entity_hash=... AND is_deleted=0)
            # 但 paragraph_entities 表没有 is_deleted 字段(它是关联表). 我们检查关联是否存在。
            # 如果 paragraph 本身 soft deleted, 它的引用应该失效吗？
            # 策略: 只有当 paragraph 也是 active 时，引用才有效。
            # JOIN paragraphs p ON pe.paragraph_hash = p.hash WHERE p.is_deleted = 0
            
            query = f"""
                SELECT e.hash FROM entities e
                WHERE e.hash IN ({placeholders})
                AND e.is_deleted = 0
                AND (e.created_at IS NULL OR e.created_at < ?)
                AND NOT EXISTS (
                    SELECT 1 FROM paragraph_entities pe
                    JOIN paragraphs p ON pe.paragraph_hash = p.hash
                    WHERE pe.entity_hash = e.hash
                    AND p.is_deleted = 0
                )
            """
            
            cursor = self._conn.cursor()
            cursor.execute(query, batch + [cutoff])
            candidates.extend([row[0] for row in cursor.fetchall()])
            
        return candidates

    def get_paragraph_gc_candidates(self, retention_seconds: float) -> List[str]:
        """
        获取段落 GC 候选列表
        条件:
        1. is_deleted = 0
        2. created_at < cutoff
        3. 没有 Relations (paragraph_relations empty)
        4. 没有 Entities 引用 (paragraph_entities empty) 
           OR 引用的 Entities 全是软删状态? (太复杂，简单点: 无引用)
           
        Refined Strategy: 
        段落孤儿判定 = 
          (Left Join paragraph_relations -> NULL) AND 
          (Left Join paragraph_entities -> NULL)
        """
        now = datetime.now().timestamp()
        cutoff = now - retention_seconds
        
        query = """
            SELECT p.hash FROM paragraphs p
            LEFT JOIN paragraph_relations pr ON p.hash = pr.paragraph_hash
            LEFT JOIN paragraph_entities pe ON p.hash = pe.paragraph_hash
            WHERE p.is_deleted = 0
            AND (p.created_at IS NULL OR p.created_at < ?)
            AND pr.relation_hash IS NULL
            AND pe.entity_hash IS NULL
        """
        
        cursor = self._conn.cursor()
        cursor.execute(query, (cutoff,))
        return [row[0] for row in cursor.fetchall()]

    def mark_as_deleted(self, hashes: List[str], type_: str) -> int:
        """
        标记为软删除 (Mark Phase)
        
        Args:
            hashes: Hash 列表
            type_: 'entity' | 'paragraph'
        """
        if not hashes:
            return 0
            
        table = "entities" if type_ == "entity" else "paragraphs"
        now = datetime.now().timestamp()
        
        count = 0
        batch_size = 900
        for i in range(0, len(hashes), batch_size):
            batch = hashes[i:i+batch_size]
            placeholders = ",".join(["?"] * len(batch))
            
            # 幂等更新: 只更那些 is_deleted=0 的
            cursor = self._conn.cursor()
            cursor.execute(f"""
                UPDATE {table}
                SET is_deleted = 1, deleted_at = ?
                WHERE is_deleted = 0 AND hash IN ({placeholders})
            """, [now] + batch)
            count += cursor.rowcount
            
        self._conn.commit()
        if count > 0:
            logger.info(f"软删除标记 ({table}): {count} 项")
        return count

    def sweep_deleted_items(self, type_: str, grace_period_seconds: float) -> List[Tuple[str, str]]:
        """
        扫描可物理清理的项目 (Sweep Phase - Selection)
        
        Args:
            type_: 'entity' | 'paragraph'
            grace_period_seconds: 宽限期
            
        Returns:
            List[(hash, name)]: 待删除项列表 (paragraph name为空)
        """
        table = "entities" if type_ == "entity" else "paragraphs"
        now = datetime.now().timestamp()
        cutoff = now - grace_period_seconds
        
        cols = "hash, name" if type_ == "entity" else "hash, '' as name"
        
        cursor = self._conn.cursor()
        cursor.execute(f"""
            SELECT {cols} FROM {table}
            WHERE is_deleted = 1
            AND deleted_at < ?
        """, (cutoff,))
        
        return [(row[0], row[1]) for row in cursor.fetchall()]

    def physically_delete_entities(self, hashes: List[str]) -> int:
        """物理删除实体 (批量)"""
        if not hashes: return 0
        
        count = 0
        batch_size = 900
        for i in range(0, len(hashes), batch_size):
            batch = hashes[i:i+batch_size]
            placeholders = ",".join(["?"] * len(batch))
            
            cursor = self._conn.cursor()
            cursor.execute(f"DELETE FROM entities WHERE hash IN ({placeholders})", batch)
            count += cursor.rowcount
            
        self._conn.commit()
        return count

    def physically_delete_paragraphs(self, hashes: List[str]) -> int:
        """物理删除段落 (批量)"""
        if not hashes: return 0
        
        count = 0
        batch_size = 900
        for i in range(0, len(hashes), batch_size):
            batch = hashes[i:i+batch_size]
            placeholders = ",".join(["?"] * len(batch))
            
            cursor = self._conn.cursor()
            cursor.execute(f"DELETE FROM paragraphs WHERE hash IN ({placeholders})", batch)
            count += cursor.rowcount
            
        self._conn.commit()
        return count

    def revive_if_deleted(self, entity_hashes: List[str] = None, paragraph_hashes: List[str] = None) -> int:
        """
        复活已软删的项目 (Auto Revival)
        当数据被再次访问、引用或导入时调用。
        """
        count = 0
        
        if entity_hashes:
            batch_size = 900
            for i in range(0, len(entity_hashes), batch_size):
                batch = entity_hashes[i:i+batch_size]
                placeholders = ",".join(["?"] * len(batch))
                
                cursor = self._conn.cursor()
                cursor.execute(f"""
                    UPDATE entities
                    SET is_deleted = 0, deleted_at = NULL
                    WHERE is_deleted = 1 AND hash IN ({placeholders})
                """, batch)
                count += cursor.rowcount
                
        if paragraph_hashes:
            batch_size = 900
            for i in range(0, len(paragraph_hashes), batch_size):
                batch = paragraph_hashes[i:i+batch_size]
                placeholders = ",".join(["?"] * len(batch))
                
                cursor = self._conn.cursor()
                cursor.execute(f"""
                    UPDATE paragraphs
                    SET is_deleted = 0, deleted_at = NULL
                    WHERE is_deleted = 1 AND hash IN ({placeholders})
                """, batch)
                count += cursor.rowcount
        
        if count > 0:
            self._conn.commit()
            logger.info(f"自动复活: {count} 项 (Soft Delete Revived)")
            
        return count

    def revive_entities_by_names(self, names: List[str]) -> int:
        """
        根据名称复活实体 (Convenience wrapper)
        """
        if not names: return 0
        
        # 使用内部方法计算哈希
        hashes = [compute_hash(self._canonicalize_name(n)) for n in names]
        return self.revive_if_deleted(entity_hashes=hashes)

    def get_entity_status_batch(self, hashes: List[str]) -> Dict[str, Dict[str, Any]]:
        """批量获取实体状态 (WebUI用)"""
        if not hashes: return {}
        
        result = {}
        batch_size = 900
        for i in range(0, len(hashes), batch_size):
            batch = hashes[i:i+batch_size]
            placeholders = ",".join(["?"] * len(batch))
            
            cursor = self._conn.cursor()
            cursor.execute(f"""
                SELECT hash, is_deleted, deleted_at 
                FROM entities 
                WHERE hash IN ({placeholders})
            """, batch)
            
            for row in cursor.fetchall():
                result[row[0]] = {
                    "is_deleted": bool(row[1]),
                    "deleted_at": row[2]
                }
        return result

    def get_deleted_entities(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取已软删除的实体 (回收站用)"""
        if not self.has_table("entities"): return []
        
        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT hash, name, deleted_at 
            FROM entities 
            WHERE is_deleted = 1 
            ORDER BY deleted_at DESC 
            LIMIT ?
        """, (limit,))
        
        items = []
        for row in cursor.fetchall():
            items.append({
                "hash": row[0],
                "name": row[1],
                "type": "entity", # 标记为实体
                "deleted_at": row[2]
            })
        return items

    def __repr__(self) -> str:
        stats = self.get_statistics() if self.is_connected else {}
        return (
            f"MetadataStore(paragraphs={stats.get('paragraph_count', 0)}, "
            f"entities={stats.get('entity_count', 0)}, "
            f"relations={stats.get('relation_count', 0)})"
        )

    def has_data(self) -> bool:
        """检查磁盘上是否存在现有数据"""
        if self.data_dir is None:
            return False
        return (self.data_dir / self.db_name).exists()
