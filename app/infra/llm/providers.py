from app.infra.config.providers import infra_config
from app.shared.model import get_llm_client, get_bge_m3_ef, generate_embeddings, get_reranker_model


class LLMProvider:

    # 获取普通大语言模型
    def chat(self, model_name: str = None, json_mode: bool = False):
        return get_llm_client(model=model_name, json_mode=json_mode)


    # 获取视觉模型
    def chat_vision(self, vision_model_name: str = None):
        return get_llm_client(model=vision_model_name or infra_config.lm_config.lv_model)

    # 获取嵌入式模型
    def bge_m3_embedding(self):
        return get_bge_m3_ef()

    # 生成向量
    def generate_embeddings(self, text: list[str]):
        return generate_embeddings(text)

    # 重排序模型
    def reranker_model(self):
        return get_reranker_model()

llm_provider = LLMProvider()