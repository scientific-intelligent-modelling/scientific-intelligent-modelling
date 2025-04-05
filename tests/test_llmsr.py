from llmsr import config as config_lib
from llmsr import buffer
from llmsr import evaluator
from llmsr.sampler import Sampler, UnifiedLLM
import os

# 设置API密钥
os.environ["OPENAI_API_KEY"] = "您的OpenAI API密钥"

# 创建配置
config = config_lib.Config(
    use_api=True,
    api_model="gpt-4",
    api_params={"temperature": 0.8}
)

# 初始化必要的组件
database = buffer.ExperienceBuffer(...)  # 根据您的项目填充参数
evaluators = [evaluator.Evaluator(...)]  # 根据您的项目填充参数

# 创建Sampler实例
sampler = Sampler(
    database=database,
    evaluators=evaluators,
    samples_per_prompt=5,
    config=config,
    llm_class=UnifiedLLM  # 使用支持多种API的UnifiedLLM类
)

# 开始采样过程
sampler.sample()