import os
"""统一的 LLM 客户端封装。

提供商/模型命名规则：'provider/model'，provider 大小写不敏感，model 保留大小写与路径。
当前支持：deepseek、siliconflow、ollama。
"""
import requests
from typing import List, Dict, Tuple


class LLMClient:
    tokens = {
        'prompt': 0,
        'content': 0,
        'reasoning': 0,
        'total': 0,
    }

    def __init__(self, api_key: str, model: str, base_url: str):
        """
        初始化 LLM 客户端。

        :param api_key: API 密钥
        :param model: 模型名称
        :param base_url: API 的基础 URL
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.kwargs = {
            'max_tokens': 5000,
            'temperature': 0.7,
            'top_p': 0.7,
            'top_k': 50,
            'frequency_penalty': 0.5,
            'n': 1,
            'stream': False,
        }

    def chat(self, messages: List[Dict[str, str]]) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        request_url = f"{self.base_url.rstrip('/')}/chat/completions"

        model_name = self.model
        if 'qwen3' in model_name.lower():
            if '/think' in model_name:
                self.kwargs['enable_thinking'] = True
                model_name = model_name.replace('/think', '')
            else:
                self.kwargs['enable_thinking'] = False
                model_name = model_name.replace('/think', '')

        payload = {
            "model": model_name,
            "messages": messages,
        }
        payload.update(self.kwargs)

        # 计时（可按需启用）
        # start_time = time.time()
        try:
            response = requests.post(request_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            response_data = response.json()
            # end_time = time.time()

            message = response_data['choices'][0]['message']
            content = message.get('content', '')
            reasoning_content = message.get('reasoning_content', '')

            # token统计
            usage = response_data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', 0)
            reasoning_tokens = 0
            if 'completion_tokens_details' in usage:
                reasoning_tokens = usage['completion_tokens_details'].get('reasoning_tokens', 0)

            self.tokens['prompt'] += prompt_tokens
            self.tokens['content'] += completion_tokens - reasoning_tokens
            self.tokens['reasoning'] += reasoning_tokens
            self.tokens['total'] += total_tokens

            return {
                "content": content,
                "reasoning_content": reasoning_content,
                "tokens": {
                    "prompt": prompt_tokens,
                    "content": completion_tokens - reasoning_tokens,
                    "reasoning": reasoning_tokens,
                    "total": total_tokens
                }
            }

        except requests.exceptions.RequestException as e:
            print(f"通过 requests 调用 API 时出错: {e}")
            if e.response:
                try:
                    print("错误详情:", e.response.json())
                except ValueError:
                    print("无法解析错误响应。")
            raise

class DeepSeekClient(LLMClient):
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.deepseek.com"):
        super().__init__(api_key=api_key, model=model, base_url=base_url)

class SiliconflowClient(LLMClient):
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.siliconflow.cn/v1"):
        super().__init__(api_key=api_key, model=model, base_url=base_url)

# 兼容旧拼写，避免历史引用报错
SliconflowClient = SiliconflowClient

class OllamaClient(LLMClient):
    def __init__(self, api_key: str, model: str, base_url: str = "http://localhost:11111/v1"):
        super().__init__(api_key=api_key, model=model, base_url=base_url)


def parse_provider_model(model_str: str) -> Tuple[str, str]:
    """
    解析模型字符串为 (provider, model)。

    规则：第一个 '/' 之前为提供商（大小写不敏感），之后的全部为模型名（大小写敏感，允许包含 '/').
    示例：
    - "deepseek/deepseek-chat" -> ("deepseek", "deepseek-chat")
    - "SiliconFlow/Qwen/Qwen3-8B" -> ("siliconflow", "Qwen/Qwen3-8B")
    - "ollama/llama3.1:8b" -> ("ollama", "llama3.1:8b")
    """
    if not isinstance(model_str, str) or '/' not in model_str:
        raise ValueError("模型名格式无效，需为 'provider/model'，如 'deepseek/deepseek-chat'")
    provider, model = model_str.split('/', 1)
    return provider.lower(), model

class ClientFactory:
    @staticmethod
    def from_config(config: dict):
        """
        基于 'provider/model' 创建具体客户端。

        必填：config['model']（形如 'provider/model'）。
        选填：config['api_key']、config['base_url']。
        """
        if 'model' not in config:
            raise ValueError("缺少必要字段: model")

        provider, model = parse_provider_model(config['model'])
        api_key = config.get('api_key')
        base_url = config.get('base_url')

        # 设置默认 base_url
        if provider == 'deepseek':
            base_url = base_url or "https://api.deepseek.com"
            return DeepSeekClient(api_key=api_key or os.getenv('DEEPSEEK_API_KEY', ''), model=model, base_url=base_url)
        elif provider in ('siliconflow', 'silicon-flow', 'sflow'):
            base_url = base_url or "https://api.siliconflow.cn/v1"
            return SiliconflowClient(api_key=api_key or os.getenv('SILICONFLOW_API_KEY', ''), model=model, base_url=base_url)
        elif provider == 'ollama':
            base_url = base_url or "http://localhost:11111/v1"
            return OllamaClient(api_key=api_key or '', model=model, base_url=base_url)
        else:
            raise ValueError(f"不支持的提供商: {provider}，请使用 'deepseek'、'siliconflow' 或 'ollama'")
        


if __name__ == '__main__':
    # 确保你的 API 密钥已经设置为环境变量 SILICONFLOW_API_KEY
    # 或者直接在这里替换 "your-siliconflow-api-key"


    client = OllamaClient(api_key='', model='llama3.1:8b')
    messages = [
        {"role": "user", "content": "你好，请介绍一下你自己，并说明你的思考过程。"}
    ]
    response_content = client.chat(messages)
    print(response_content)

    # deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key")
    # if deepseek_api_key == "your-deepseek-api-key":
    #     print("请设置 DEEPSEEK_API_KEY 环境变量或直接在代码中提供您的 API 密钥。")
    # else:
    #     client = DeepSeekClient(api_key=deepseek_api_key, model='deepseek-reasoner')
    #     messages = [
    #         {"role": "user", "content": "你好，请介绍一下你自己，并说明你的思考过程。"}
    #     ]
    #     response_content = client.chat(messages)
    #     print(response_content)

    # print('=='*20)

    # api_key = os.getenv("SILICONFLOW_API_KEY", "your-siliconflow-api-key")
    # if api_key == "your-siliconflow-api-key":
    #     print("请设置 SILICONFLOW_API_KEY 环境变量或直接在代码中提供您的 API 密钥。")
    # else:
    #     model_lists = [
    #         'Qwen/Qwen3-8B/think',
    #         'Qwen/Qwen3-8B',
    #         'Qwen/QwQ-32B',
    #         'Qwen/Qwen3-32B',
    #         'Qwen/Qwen2.5-72B-Instruct',
    #         'Qwen/Qwen2.5-32B-Instruct',
    #     ]
    #     for model in model_lists:
    #         print('【this is model: 】', model)
    #         client = SliconflowClient(api_key=api_key, model=model)
    #         messages = [
    #             {"role": "user", "content": "你好，请介绍一下你自己，并说明你的思考过程。"}
    #         ]
            
    #         try:
    #             response_content = client.chat(messages)
    #             print(response_content)
    #         except Exception as e:
    #             print(f"调用模型时出错: {e}")

    #         print("\n" + "="*20 + "\n")
