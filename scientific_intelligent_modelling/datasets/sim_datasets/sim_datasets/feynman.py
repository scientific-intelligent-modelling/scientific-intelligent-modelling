"""
费曼数据集处理模块
"""
from typing import Dict, Any, Optional, Union, List, Tuple
import pandas as pd
import pathlib
import os
import random

from .utils import (
    get_dataset_dir, 
    parallel_download, 
    ensure_dir, 
    get_sample_data, 
    generate_mock_data
)

# 费曼数据集基本信息
FEYNMAN_DATASETS = [
    {
        "id": 1,
        "name": "feynman_I.6.2",
        "description": "位移公式",
        "equation": "x = x0 + v*t + 0.5*a*t^2",
        "features": ["x0", "v", "t", "a"],
        "target": "x"
    },
    {
        "id": 2,
        "name": "feynman_I.8.14", 
        "description": "动能公式",
        "equation": "KE = 0.5*m*v^2",
        "features": ["m", "v"],
        "target": "KE"
    },
    {
        "id": 3,
        "name": "feynman_I.9.18", 
        "description": "圆周运动向心加速度",
        "equation": "a = v^2/r",
        "features": ["v", "r"],
        "target": "a"
    },
    {
        "id": 4,
        "name": "feynman_I.12.1", 
        "description": "弹性势能",
        "equation": "V = 0.5*k*x^2",
        "features": ["k", "x"],
        "target": "V"
    },
    {
        "id": 5,
        "name": "feynman_I.12.11", 
        "description": "重力势能",
        "equation": "V = m*g*h",
        "features": ["m", "g", "h"],
        "target": "V"
    },
    # 更多费曼数据集...
]

# 创建ID和名称到数据集的映射
FEYNMAN_BY_ID = {str(ds["id"]): ds for ds in FEYNMAN_DATASETS}
FEYNMAN_BY_NAME = {ds["name"]: ds for ds in FEYNMAN_DATASETS}

def list_feynman_datasets() -> List[Dict[str, Any]]:
    """
    列出所有费曼数据集
    
    Returns:
        List[Dict[str, Any]]: 数据集列表
    """
    return FEYNMAN_DATASETS

def get_feynman_info(
    identifier: Optional[str] = None,
    n_samples: int = 5,
    data_root: Optional[Union[str, pathlib.Path]] = None
) -> Dict[str, Any]:
    """
    获取费曼数据集信息
    
    Args:
        identifier: 数据集标识符，可以是ID或名称
                   如果不提供，则返回所有费曼数据集信息
        n_samples: 返回的样本数量
        data_root: 数据根目录
        
    Returns:
        Dict: 数据集信息
        
    Raises:
        ValueError: 如果数据集不存在
    """
    if identifier is None:
        # 返回所有数据集的总体信息
        return {
            "dataset_group": "feynman",
            "description": "费曼方程数据集集合",
            "count": len(FEYNMAN_DATASETS),
            "datasets": FEYNMAN_DATASETS,
            "examples": random.sample(FEYNMAN_DATASETS, min(3, len(FEYNMAN_DATASETS)))
        }
    
    # 按ID查找
    if identifier in FEYNMAN_BY_ID:
        dataset_info = FEYNMAN_BY_ID[identifier]
    # 按名称查找
    elif identifier in FEYNMAN_BY_NAME:
        dataset_info = FEYNMAN_BY_NAME[identifier]
    else:
        raise ValueError(f"未找到费曼数据集: {identifier}")
    
    # 尝试加载样本数据
    try:
        df = load_feynman_dataset(identifier, data_root=data_root)
        sample_data = get_sample_data(df, n_samples)
    except Exception:
        # 数据不可用，生成模拟数据
        sample_data = generate_mock_data(
            dataset_info["features"], 
            dataset_info["target"], 
            n_samples
        )
    
    # 返回增强的数据集信息
    result = {**dataset_info}
    result["sample_data"] = sample_data
    return result

def load_feynman_dataset(
    dataset_name: str, 
    data_root: Optional[Union[str, pathlib.Path]] = None
) -> pd.DataFrame:
    """
    加载特定的费曼数据集
    
    Args:
        dataset_name: 数据集名称或ID
        data_root: 数据存储根目录
        
    Returns:
        pd.DataFrame: 加载的数据集
        
    Raises:
        ValueError: 如果数据集不存在
    """
    # 确定数据集名称
    if dataset_name in FEYNMAN_BY_ID:
        dataset_name = FEYNMAN_BY_ID[dataset_name]["name"]
    elif dataset_name not in FEYNMAN_BY_NAME:
        raise ValueError(f"未找到费曼数据集: {dataset_name}")
    
    # 确定数据目录
    if data_root is None:
        data_root = pathlib.Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "data"
    
    # 数据集文件路径
    dataset_dir = get_dataset_dir(data_root, "feynman") / dataset_name
    data_file = dataset_dir / "data.csv"
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"数据集文件不存在: {data_file}\n"
            f"请先下载数据集或确保数据文件放置在正确位置。"
        )
    
    # 加载数据
    try:
        df = pd.read_csv(data_file)
        return df
    except Exception as e:
        raise RuntimeError(f"加载数据集失败: {str(e)}")

def download_feynman_datasets(
    dataset_names: Optional[List[str]] = None,
    data_root: Optional[Union[str, pathlib.Path]] = None,
    base_url: str = "https://space.ia.ac.cn/feynman-datasets",
    max_workers: int = 4
) -> Dict[str, bool]:
    """
    下载费曼数据集
    
    Args:
        dataset_names: 要下载的数据集名称或ID列表，None表示下载所有
        data_root: 数据存储根目录
        base_url: 数据集基础URL
        max_workers: 并行下载的工作线程数
        
    Returns:
        Dict[str, bool]: 数据集名称到下载状态的映射
    """
    # 确定数据根目录
    if data_root is None:
        data_root = pathlib.Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "data"
    
    # 确定要下载的数据集
    if dataset_names is None:
        dataset_names = [ds["name"] for ds in FEYNMAN_DATASETS]
    else:
        # 将数字ID转换为名称
        processed_names = []
        for name in dataset_names:
            if name in FEYNMAN_BY_ID:
                processed_names.append(FEYNMAN_BY_ID[name]["name"])
            elif name in FEYNMAN_BY_NAME:
                processed_names.append(name)
            else:
                raise ValueError(f"未找到费曼数据集: {name}")
        dataset_names = processed_names
    
    # 准备下载任务
    downloads = []
    for name in dataset_names:
        dataset_dir = get_dataset_dir(data_root, "feynman") / name
        ensure_dir(dataset_dir)
        
        # 数据和元信息文件
        data_url = f"{base_url}/{name}/data.csv"
        data_path = dataset_dir / "data.csv"
        downloads.append((data_url, data_path))
        
        # 可能还有其他文件需要下载
        info_url = f"{base_url}/{name}/info.json" 
        info_path = dataset_dir / "info.json"
        downloads.append((info_url, info_path))
    
    # 执行并行下载
    results = parallel_download(downloads, max_workers=max_workers)
    
    # 整理结果
    dataset_results = {}
    for name in dataset_names:
        data_path = str(get_dataset_dir(data_root, "feynman") / name / "data.csv")
        info_path = str(get_dataset_dir(data_root, "feynman") / name / "info.json")
        
        # 只有当所有必需文件都下载成功时，才视为成功
        dataset_results[name] = results.get(data_path, False) and results.get(info_path, False)
    
    return dataset_results