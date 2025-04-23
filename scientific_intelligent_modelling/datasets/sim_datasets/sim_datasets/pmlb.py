"""
PMLB (Penn Machine Learning Benchmark) 数据集处理模块
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

# PMLB数据集信息
PMLB_DATASETS = [
    {
        "id": "1",
        "name": "adult",
        "description": "成人收入预测数据集",
        "task": "classification",
        "features": ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
                    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
                    "hours-per-week", "native-country"],
        "target": "income",
        "n_samples": 48842,
        "n_features": 14,
        "binary": True
    },
    {
        "id": "2",
        "name": "iris",
        "description": "鸢尾花分类数据集",
        "task": "classification",
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "target": "class",
        "n_samples": 150,
        "n_features": 4,
        "binary": False
    },
    {
        "id": "3",
        "name": "boston",
        "description": "波士顿房价预测数据集",
        "task": "regression",
        "features": ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", 
                    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"],
        "target": "MEDV",
        "n_samples": 506,
        "n_features": 13,
        "binary": False
    }
    # 可以添加更多PMLB数据集
]

# 创建ID和名称到数据集的映射
PMLB_BY_ID = {ds["id"]: ds for ds in PMLB_DATASETS}
PMLB_BY_NAME = {ds["name"]: ds for ds in PMLB_DATASETS}

def list_pmlb_datasets() -> List[Dict[str, Any]]:
    """
    列出所有PMLB数据集
    
    Returns:
        List[Dict[str, Any]]: 数据集列表
    """
    return PMLB_DATASETS

def get_pmlb_info(
    identifier: Optional[str] = None,
    n_samples: int = 5,
    data_root: Optional[Union[str, pathlib.Path]] = None
) -> Dict[str, Any]:
    """
    获取PMLB数据集信息
    
    Args:
        identifier: 数据集标识符，可以是ID或名称
                   如果不提供，则返回所有PMLB数据集信息
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
            "dataset_group": "pmlb",
            "description": "Penn Machine Learning Benchmark数据集集合",
            "count": len(PMLB_DATASETS),
            "datasets": PMLB_DATASETS,
            "examples": random.sample(PMLB_DATASETS, min(3, len(PMLB_DATASETS)))
        }
    
    # 按ID查找
    if identifier in PMLB_BY_ID:
        dataset_info = PMLB_BY_ID[identifier]
    # 按名称查找
    elif identifier in PMLB_BY_NAME:
        dataset_info = PMLB_BY_NAME[identifier]
    else:
        raise ValueError(f"未找到PMLB数据集: {identifier}")
    
    # 尝试加载样本数据
    try:
        df = load_pmlb_dataset(identifier, data_root=data_root)
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

def load_pmlb_dataset(
    dataset_name: str, 
    data_root: Optional[Union[str, pathlib.Path]] = None
) -> pd.DataFrame:
    """
    加载特定的PMLB数据集
    
    Args:
        dataset_name: 数据集名称或ID
        data_root: 数据存储根目录
        
    Returns:
        pd.DataFrame: 加载的数据集
        
    Raises:
        ValueError: 如果数据集不存在
    """
    # 确定数据集名称
    if dataset_name in PMLB_BY_ID:
        dataset_name = PMLB_BY_ID[dataset_name]["name"]
    elif dataset_name not in PMLB_BY_NAME:
        raise ValueError(f"未找到PMLB数据集: {dataset_name}")
    
    # 确定数据目录
    if data_root is None:
        data_root = pathlib.Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "data"
    
    # 数据集文件路径
    dataset_dir = get_dataset_dir(data_root, "pmlb") / dataset_name
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

def download_pmlb_datasets(
    dataset_names: Optional[List[str]] = None,
    data_root: Optional[Union[str, pathlib.Path]] = None,
    base_url: str = "https://github.com/EpistasisLab/pmlb/raw/master/datasets",
    max_workers: int = 4
) -> Dict[str, bool]:
    """
    下载PMLB数据集
    
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
        dataset_names = [ds["name"] for ds in PMLB_DATASETS]
    else:
        # 将数字ID转换为名称
        processed_names = []
        for name in dataset_names:
            if name in PMLB_BY_ID:
                processed_names.append(PMLB_BY_ID[name]["name"])
            elif name in PMLB_BY_NAME:
                processed_names.append(name)
            else:
                raise ValueError(f"未找到PMLB数据集: {name}")
        dataset_names = processed_names
    
    # 准备下载任务
    downloads = []
    for name in dataset_names:
        dataset_dir = get_dataset_dir(data_root, "pmlb") / name
        ensure_dir(dataset_dir)
        
        # 数据文件
        data_url = f"{base_url}/{name}/{name}.tsv.gz"
        data_path = dataset_dir / "data.csv"
        downloads.append((data_url, data_path))
        
        # 元信息文件
        meta_url = f"{base_url}/{name}/{name}_metadata.json"
        meta_path = dataset_dir / "info.json"
        downloads.append((meta_url, meta_path))
    
    # 执行并行下载
    results = parallel_download(downloads, max_workers=max_workers)
    
    # 整理结果
    dataset_results = {}
    for name in dataset_names:
        data_path = str(get_dataset_dir(data_root, "pmlb") / name / "data.csv")
        meta_path = str(get_dataset_dir(data_root, "pmlb") / name / "info.json")
        
        # 只有当所有必需文件都下载成功时，才视为成功
        dataset_results[name] = results.get(data_path, False) and results.get(meta_path, False)
    
    return dataset_results