import subprocess
import sys

# 定义要执行的命令
# 这里我们使用 `source activate` 来激活环境并执行命令
command = """
source activate llmsr && \
pip install -r ./scientific_intelligent_modelling/algorithms/llmsr_wrapper/llmsr/requirements.txt --index-strategy unsafe-best-match
"""

# 使用 /bin/bash 来执行命令，因为我们需要使用 shell 的特性
process = subprocess.Popen(
    ['/bin/bash', '-c', command],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# 实时输出结果
while True:
    output = process.stdout.readline()
    if output == '' and process.poll() is not None:
        break
    if output:
        print(output.strip())

# 打印错误（如果有）
stderr_output = process.stderr.read()
if stderr_output:
    print(f"ERROR: {stderr_output.strip()}")

# 打印返回码
print(f"命令执行完成，返回码: {process.returncode}")
