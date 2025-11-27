#同步所有子模块
git pull --recurse-submodules  

# 测试是否能跑通
sim-cli   -a llmsr   -t scientific_intelligent_modelling/algorithms/llmsr_wrapper/llmsr/data/oscillator1/train.csv   --problem_name oscillator1   --llm_config_path ~/llm-4omini.config   --background '这是一个抽象的数学回归问题。'   --niterations 5  --use_wandb --wandb_entity ziwenhahaha --wandb_project 1.llm_background_ablation --wandb_group group1 --wandb_name llmsrbench_II.27.18_1_0_L1  --wandb_tags sim,llmsr