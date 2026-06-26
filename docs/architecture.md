# 企业化重构架构说明

## 重构目标

- 保留现有导入流程与查询流程的可运行能力，并统一收口到 `app/process`
- 保留导入服务与查询服务独立运行、独立端口的原始语义
- 将接口入口统一收敛到 `app/api`，但仍保持两个独立服务
- 使用 `api / infra / shared / rag / process` 做收敛，而不过度增加中间层

## 当前分层

### 服务入口

- `app/api/http/import_server.py`
  - 导入服务独立入口
  - 默认端口 `8000`

- `app/api/http/query_server.py`
  - 查询服务独立入口
  - 默认端口 `8001`

### API 层

- `app/api/http/import_server.py`
  - 使用 `@app.get / @app.post` 直接定义导入接口
  - 在同文件内直接处理上传、后台任务与状态查询

- `app/api/http/query_server.py`
  - 使用 `@app.get / @app.post` 直接定义查询接口与 SSE
  - 在同文件内直接处理会话、查询执行与历史记录

- `app/api/schemas`
  - API 输入输出模型
  - 仅保留请求与响应数据结构

### Infra 层

- `app/infra/config/settings.py`
  - 读取导入服务和查询服务的端口、名称、环境等配置

- `app/infra/config`
  - 作为配置统一出口，对外优先通过包级导出访问 `settings`、`infra_config` 及各类配置别名

- `app/infra/llm/providers.py`
  - 聚合 ChatModel、Embedding、Reranker 的获取方式

- `app/infra/vectorstore/milvus_gateway.py`
  - 聚合 Milvus 客户端、混合检索请求、按 chunk_id 回查

- `app/infra/persistence/history_repository.py`
  - 聚合历史对话查询、写入、清空等能力

- `app/infra/object_storage/minio_gateway.py`
  - 聚合 MinIO 客户端与桶配置

- `app/infra/document_parse/mineru_gateway.py`
  - 聚合 MinerU 文档解析服务配置出口

- `app/infra/websearch/dashscope_gateway.py`
  - 聚合 DashScope WebSearch MCP 的连接与调用参数

- 该层职责是对外部系统做正式门面封装
  - 不是简单把旧代码再包一层
  - 上层 `api / process / rag` 只依赖这里暴露的稳定出口

### Resources 层

- `app/resources/prompts`
  - 存放查询改写、答案生成、图片总结等提示词模板
  - 属于应用运行资源，不放入 `shared` 或根目录

### Shared 层

- `app/shared/config`
  - 原始配置对象与环境变量读取

- `app/shared/runtime`
  - 日志、提示词加载等运行时公共能力

- `app/shared/model`
  - Embedding、LLM、Reranker 等模型基础工具

- `app/shared/clients`
  - Milvus、Mongo、MinIO 等底层客户端工具

- `app/shared/utils`
  - SSE、任务状态、路径处理等通用工具

- `app/shared/tool`
  - 下载脚本等辅助工具

### 流程编排层

- `app/process/import_/*`
  - 保留原导入图与节点

- `app/process/query/*`
  - 保留原查询图与节点

### RAG 能力层

- `app/rag/import_`
  - 对应导入域能力
  - 文件按导入步骤组织，例如 `entry_service.py`、`pdf_parse_service.py`、`split_service.py`

- `app/rag/query`
  - 对应查询域能力
  - 文件按查询节点职责组织，例如 `item_name_confirm_service.py`、`search_embedding_service.py`、`rrf_service.py`、`answer_output_service.py`

## 调用关系图

### 导入链

```text
HTTP Request
  -> app/api/http/import_server.py
  -> process/import_/agent/main_graph.py
  -> process/import_/agent/nodes/node_entry.py
  -> rag/import_/entry_service.py
  -> process/import_/agent/nodes/node_pdf_to_md.py
  -> rag/import_/pdf_parse_service.py
  -> process/import_/agent/nodes/node_md_img.py
  -> rag/import_/markdown_image_service.py
  -> process/import_/agent/nodes/node_document_split.py
  -> rag/import_/split_service.py
  -> process/import_/agent/nodes/node_item_name_recognition.py
  -> rag/import_/item_name_service.py
  -> process/import_/agent/nodes/node_bge_embedding.py
  -> rag/import_/embedding_service.py
  -> process/import_/agent/nodes/node_import_milvus.py
  -> rag/import_/index_service.py
  -> infra/vectorstore/milvus_gateway.py
```

- `import_server.py` 负责接收上传请求、保存文件、生成 `task_id`、启动后台任务与状态查询
- `process/import_` 负责导入流程编排与节点顺序控制
- `rag/import_` 负责每个导入步骤的具体实现
- `infra/*` 负责调用 Milvus、模型服务、对象存储等底层依赖

### 查询链

```text
HTTP Request
  -> app/api/http/query_server.py
  -> process/query/agent/main_graph.py
  -> process/query/agent/nodes/node_item_name_confirm.py
  -> rag/query/item_name_confirm_service.py
  -> process/query/agent/nodes/node_search_embedding.py
  -> rag/query/search_embedding_service.py
  -> process/query/agent/nodes/node_search_embedding_hyde.py
  -> rag/query/search_embedding_hyde_service.py
  -> process/query/agent/nodes/node_web_search_mcp.py
  -> rag/query/web_search_service.py
  -> process/query/agent/nodes/node_rrf.py
  -> rag/query/rrf_service.py
  -> process/query/agent/nodes/node_rerank.py
  -> rag/query/rerank_service.py
  -> process/query/agent/nodes/node_answer_output.py
  -> rag/query/answer_output_service.py
  -> infra/llm/providers.py
  -> infra/vectorstore/milvus_gateway.py
  -> infra/persistence/history_repository.py
```

- `query_server.py` 负责普通问答、流式问答、历史记录、会话管理、图调用与 SSE 收尾
- `process/query` 负责查询图编排与节点流转
- `rag/query` 负责按节点职责实现商品确认、检索、HyDE、联网检索、RRF、重排和答案生成
- `infra/*` 负责调用 LLM、Milvus、Mongo 等底层能力

## 当前结构说明

- `api`
  - 只负责接口定义、请求参数与响应返回
  - 不再额外包 `application` 转发层

- `process/import_ / process/query`
  - 负责流程编排
  - 保留原来 LangGraph 主链

- `rag/import_ / rag/query`
  - 负责节点背后的具体能力实现
  - 将复杂逻辑从节点中继续下沉

## 能力下沉情况

### 查询链

- 已下沉到 `app/rag/query` 的能力包括：
  - `item_name_confirm_service.py`
  - `search_embedding_service.py`
  - `search_embedding_hyde_service.py`
  - `web_search_service.py`
  - `rrf_service.py`
  - `rerank_service.py`
  - `answer_output_service.py`

- 已瘦身的查询节点包括：
  - `node_item_name_confirm`
  - `node_search_embedding`
  - `node_search_embedding_hyde`
  - `node_web_search_mcp`
  - `node_rrf`
  - `node_rerank`
  - `node_answer_output`

### 导入链

- 已下沉到 `app/rag/import_` 的能力包括：
  - `entry_service.py`
  - `split_service.py`
  - `embedding_service.py`
  - `index_service.py`
  - `item_name_service.py`
  - `pdf_parse_service.py`
  - `markdown_image_service.py`

- 已瘦身的导入节点包括：
  - `node_entry`
  - `node_document_split`
  - `node_bge_embedding`
  - `node_import_milvus`
  - `node_item_name_recognition`
  - `node_pdf_to_md`
  - `node_md_img`

## 启动建议

```bash
python -m app.api.http.import_server
python -m app.api.http.query_server
```

或

```bash
uvicorn app.api.http.import_server:app --host 0.0.0.0 --port 8000
uvicorn app.api.http.query_server:app --host 0.0.0.0 --port 8001
```
