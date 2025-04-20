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

3、测试e2esr环境，先自己搭一遍，走通搭建的流程。
我这里已经走通了，所以我直接把搭建过程写下面：
把流程拆成三步，第一步是创建空环境，第二步是让后处理指令去更新环境
```
conda env create --name e2esr
conda activate e2esr
conda env update --file scientific_intelligent_modelling/algorithms/e2esr_wrapper/e2esr/environment.yml --prune
```

然后发现它自带的environment太冗余了，缩减为如下：
name: symbolicregression
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.7
  - pip=22.0.4
  - pip:
    - numpy==1.21.5
    - scipy==1.7.3
    - scikit-learn==1.0.2
    - pandas==1.3.5
    - sympy==1.10.1
    - torch==1.11.0
    - torchaudio==0.11.0
    - torchvision==0.12.0
    - tqdm==4.64.0
    - pyyaml==6.0
    - requests==2.27.1

然后将这个需求，填入