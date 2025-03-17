import torch
print("torch版本:   ",torch.__version__)
print("cuda版本:    ",torch.version.cuda)
print("cudnn版本:   ",torch.backends.cudnn.version())
print("cuda能否使用: ",torch.cuda.is_available())
print("gpu数量:     ",torch.cuda.device_count())
print("当前设备索引: ",torch.cuda.current_device())
print("返回gpu名字： ",torch.cuda.get_device_name(0))
try:
    print("返回gpu名字： ",torch.cuda.get_device_name(1))
except:
    pass