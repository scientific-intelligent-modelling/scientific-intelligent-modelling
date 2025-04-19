# 如何集成LLM-SR环境

## 环境测试（与之前的环境融合，这里先忽略这一步）

测试llm-sr环境是否与之前的环境冲突，若冲突了，则需要新建一个新的环境，再集成进去

这里先不进行环境的测试，直接新建一个名为“llmsr”的环境

## 环境测试（测试能否在虚拟环境中跑通）

要把工具集成进去，第一步，就是得在自己的环境里面能够成功运行。

1、主仓库进行clone：

```bash
git clone https://github.com/ziwenhahaha/scientific-intelligent-modelling.git  --recursive
cd scientific-intelligent-modelling
```

2、把子仓库进行添加submodule

```bash
mkdir scientific_intelligent_modelling/algorithms/e2esr_wrapper
git submodule add https://github.com/scientific-intelligent-modelling/e2e-symbolic-regression.git scientific_intelligent_modelling/algorithms/e2esr_wrapper/e2esr
```

3、创建conda子环境：llmsr

```bash
conda create -n llmsr  python=3.11.7
conda activate llmsr
pip install -r scientific_intelligent_modelling/algorithms/llmsr_wrapper/llmsr/requirements.txt
```
