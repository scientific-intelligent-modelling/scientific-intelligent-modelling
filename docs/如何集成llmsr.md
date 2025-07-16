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
mkdir scientific_intelligent_modelling/algorithms/llmsr_wrapper
git submodule add https://github.com/deep-symbolic-mathematics/LLM-SR.git scientific_intelligent_modelling/algorithms/llmsr_wrapper/llmsr
```

3、创建conda子环境：llmsr

```bash
conda create -n llmsr  python=3.11.7
conda activate llmsr
pip install -r scientific_intelligent_modelling/algorithms/llmsr_wrapper/llmsr/requirements.txt
```

4、还需要给这个torch配置相应的cudatookit，通过查看它的requirement.txt得知，它所需要的cudatookit是11.8的，于是我们使用conda进行环境配置：

```bash
conda activate llmsr
conda install cudatoolkit=11.8
```

5、至此，环境就创建好了，先随便跑一个测试代码：
cd 到对应的项目目录，如

```bash
cd ./scientific_intelligent_modelling/algorithms/llmsr_wrapper/llmsr
```

执行测试脚本：

```bash
python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name stressstrain --spec_path ./specs/specification_stressstrain_numpy.txt --log_path ./logs/stressstrain_gpt3.5
```

可以看到它已经是在跑了，只不过就是api没有配好，导致它卡在了第一次迭代那里，但是总体而言，它是已经能运行了。

6、适配环境
于是为了让llmsr更好的融入到工具包环境中，我们需要进行一些自适应修改。
分析现存的问题：

Ⅰ、现在大模型的接口需要适配，但是各大厂商的api不一定统一，看看有没有什么包可以非常简单的适配这个。

- 比如说，LLM-SR的原来的仓库里面，并没有remote-api的选项，我希望科学家们使用SIM工具箱的时候，不需要自己去手动配一个大模型，即可使用。于是，我就需要修改，代码，使得其支持这些api。具体来说需要修改 `--use_api True --api_model "gpt-3.5-turbo"` 这两个参数，首先 `use_api`必须是True，其次 `gpt-3.5-turbo` 这个需要改成自己的，我还希望引进一个新变量，来表明它是什么平台的api，比如说是ollama的，比如说是硅基流动的。

Ⅱ、然后是适配工具包的框架.fit .predict

7、适应第一个问题，大模型的统一API
调研发现，github上有一个库litellm，专门实现了这个，然后使用copilot进行集成。具体的修改是，修改main.py这个入口文件

---

以上是对应于仓库本体的修改，还未涉及到迁移进入sim工具箱中。以下是迁移入工具箱：

8、修改config文件，需要修改scientific_intelligent_modelling/config/envs_config.json 以及 scientific_intelligent_modelling/config/toolbox_config.json

**envs_config.json**:

```json
    "llmsr": {
      "python_version": "3.11.7",
      "conda_packages": [
        "pip",
      ],
      "channels": [
        "conda-forge"
      ],
      "pip_packages": [
        "requests",
        "numpy",
        "litellm"
      ],
      "post_install_commands": [
        "pip install -e .",
        "pip install -r ./scientific_intelligent_modelling/algorithms/llmsr_wrapper/llmsr/requirements.txt"
      ]
    }
```

对应属性的含义：


### python_version

`"python_version": "3.11.7"`
指定conda环境使用的Python版本为3.11.7。创建环境时，会安装这个特定版本的Python解释器。

### conda_packages

`"conda_packages": ["pip"]`
列出需要通过conda包管理器安装的包。这里指定安装:

* `pip`: Python的包管理工具

### channels

`"channels": ["conda-forge"]`
指定conda获取包的渠道(channel)。conda-forge是一个社区维护的包仓库，提供了很多在默认conda渠道中不可用的包。

### pip_packages

`"pip_packages": ["requests", "numpy", "litellm"]`
列出通过pip安装的Python包:

* `requests`: HTTP库，用于发送网络请求
* `numpy`: 科学计算库，提供高性能的数组操作
* `litellm`: 用于统一不同大语言模型API的库

### post_install_commands

`"post_install_commands": [...] `
环境创建后需要执行的命令:

1. `"pip install -e ."`
   以可编辑模式(`-e`)安装当前目录下的Python包，这使得对源代码的修改立即生效，无需重新安装
2. `"pip install -r [requirements.txt](http://_vscodecontentref_/0)"`

这整个配置用于自动创建和设置运行LLMSR(Large Language Model Symbolic Regression)算法所需的全部环境依赖。

修改完这个json，去toolbox_config.json中，进行对应的修改。

9、适配完llmsr之后，就需要去修改wrapper了，建议使用copilot的agent模式，使用claude 3.7 sonnet模型
使用以下的prompt:

> 我现在描述一下我的项目所需要的需求：首先我这个是一个工具箱性质的，我有一个叫做sim的基础环境，我在里面调用各个regressor，每个regressor类都有.fit .predict 等等方法。帮我仿照别的wrapper，写一下专属于llmsr的wrapper.py，着重关注于子仓库里面的main.py来实现。

10、至此，一个及格水平的llmsr工具就集成成功了，达到及格远比达到优秀要重要。之后就需要各种打磨细节，比如参数的调用，参数的暴露等等，这些交给时间，用到才来打磨。
