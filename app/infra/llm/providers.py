from app import infra
from app.infra.config.providers import infra_config
from app.shared.model import get_llm_client


class LLMProvider:

    # 获取普通大语言模型
    def chat(self, model_name: str, json_mode: bool):
        return get_llm_client(model=model_name, json_mode=json_mode)



    # 获取视觉模型
    def chat_vision(self, vision_model_name: str = None):
        return get_llm_client(model=vision_model_name or infra_config.lm_config.lv_model)


llm_provider = LLMProvider()