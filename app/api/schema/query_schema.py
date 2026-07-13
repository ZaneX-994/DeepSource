from typing import Any

from pydantic import BaseModel, Field


class HealthResponseSchema(BaseModel):
    code:int = 200
    message:str = None

# 查询接口的请求参数
class QueryRequestSchema(BaseModel):
    query:str
    session_id:str = None
    is_stream:bool = False

# 查询模块流式响应结果
class QueryStreamResponseSchema(BaseModel):
    message:str
    session_id:str

class QueryNoneStreamResponseSchema(BaseModel):
    message:str
    session_id:str
    answer:str
    done_list:list
    image_urls:list

class HistoryClearResponseSchema(BaseModel):
    message:str
    deleted_count:int

class HistoryItemResponseSchema(BaseModel):
    id:str
    session_id:str
    role:str
    text:str
    rewritten_query:str = None
    item_names:list[str] = Field(description="关联的item_name", default_factory=list)
    image_urls:list[str] = Field(description="关联的图片地址", default_factory=list)
    ts:Any = None


class HistoryListResponseSchema(BaseModel):
    session_id:str
    items:list[HistoryItemResponseSchema] = Field(description="查询的数据记录列表", default_factory=list)