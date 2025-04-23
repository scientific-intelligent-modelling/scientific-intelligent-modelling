"""
SIM Datasets - 科学智能建模数据集集合
"""
from typing import Dict, List, Any, Union, Optional
import pandas as pd
import os
import pathlib
import random

# 导入数据集处理模块 
from . import feynman
from . import pmlb

# 导入工具函数
from .utils import parse_dataset_identifier

# 数据集根目录
DATA_ROOT = pathlib.Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "data"

# 数据集组映射表 - 更加直观的组织方式
DATASET_GROUPS = {
    "feynman": {
        "list": feynman.list_feynman_datasets,
        "info": feynman.get_feynman_info,
        "load": feynman.load_feynman_dataset,
        "download": feynman.download_feynman_datasets
    },
    "pmlb": {
        "list": pmlb.list_pmlb_datasets,
        "info": pmlb.get_pmlb_info,
        "load": pmlb.load_pmlb_dataset,
        "download": pmlb.download_pmlb_datasets
    }
    # 可以在此添加更多数据集组
}

def list_datasets(group: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    列出所有可用的数据集
    
    Args:
        group: 数据集组名称，如果提供，则只列出该组的数据集
    
    Returns:
        List[Dict[str, Any]]: 数据集信息列表
    """
    if group is not None:
        if group not in DATASET_GROUPS:
            raise ValueError(f"未知的数据集组: {group}")
        return DATASET_GROUPS[group]["list"]()
    
    # 合并所有数据集组的数据集
    all_datasets = []
    for group_name, functions in DATASET_GROUPS.items():
        datasets = functions["list"]()
        for dataset in datasets:
            # 添加组信息以便识别
            enriched_dataset = {**dataset, "group": group_name}
            all_datasets.append(enriched_dataset)
    
    return all_datasets

def get_dataset_info(
    identifier: Optional[str] = None,
    n_samples: int = 5,
    data_root: Optional[Union[str, pathlib.Path]] = None
) -> Dict[str, Any]:
    """
    获取数据集信息
    
    Args:
        identifier: 数据集标识符，格式为"组/ID"或"组/名称"，如"feynman/1"或"feynman/feynman_I.6.2"
                  如果只传入"组"，则返回该组的概要信息
                  如果不传值，则返回所有数据集的概要信息
        n_samples: 示例数据的样本数量
        data_root: 数据根目录
        
    Returns:
        Dict: 数据集信息
    """
    # 如果未指定，则返回所有数据集组的概要
    if identifier is None:
        return {
            "dataset_groups": list(DATASET_GROUPS.keys()),
            "total_count": len(list_datasets()),
            "groups_info": {
                group: functions["info"](None) 
                for group, functions in DATASET_GROUPS.items()
            }
        }
    
    # 解析标识符
    group, dataset_id = parse_dataset_identifier(identifier)
    
    # 检查数据集组是否存在
    if group not in DATASET_GROUPS:
        raise ValueError(f"未知的数据集组: {group}")
    
    # 获取该组的处理函数
    info_func = DATASET_GROUPS[group]["info"]
    
    # 调用对应的info函数
    return info_func(dataset_id, n_samples=n_samples, data_root=data_root)

def load_dataset(
    identifier: str, 
    data_root: Optional[Union[str, pathlib.Path]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    加载指定的数据集
    
    Args:
        identifier: 数据集标识符，格式为"组/ID"或"组/名称"，如"feynman/1"或"feynman/feynman_I.6.2"
        data_root: 数据根目录
        **kwargs: 额外参数
    
    Returns:
        pd.DataFrame: 加载的数据集
        
    Raises:
        ValueError: 如果数据集不存在或格式不正确
    """
    # 解析标识符
    group, dataset_id = parse_dataset_identifier(identifier)
    
    # 如果没有指定数据集ID，抛出错误
    if dataset_id is None:
        raise ValueError(f"数据集标识符必须包含组和ID/名称: {identifier}")
    
    # 检查数据集组是否存在
    if group not in DATASET_GROUPS:
        raise ValueError(f"未知的数据集组: {group}")
    
    # 获取该组的处理函数
    load_func = DATASET_GROUPS[group]["load"]
    
    # 调用对应的load函数
    return load_func(dataset_id, data_root=data_root, **kwargs)

def download_datasets(
    identifiers: Optional[List[str]] = None,
    data_root: Optional[Union[str, pathlib.Path]] = None,
    max_workers: int = 4,
    **kwargs
) -> Dict[str, bool]:
    """
    下载指定的数据集
    
    Args:
        identifiers: 要下载的数据集标识符列表，格式为"组/ID"或"组/名称"
                   如果只指定了组，如"feynman"，则下载该组的所有数据集
                   如果为None，则下载所有数据集
        data_root: 数据根目录
        max_workers: 并行下载的最大线程数
        **kwargs: 额外参数
        
    Returns:
        Dict[str, bool]: 数据集标识符到下载状态的映射
    """
    results = {}
    
    # 如果未指定标识符，下载所有数据集
    if identifiers is None:
        for group, functions in DATASET_GROUPS.items():
            group_results = functions["download"](None, data_root=data_root, max_workers=max_workers, **kwargs)
            
            # 格式化结果键以包含组名
            for name, success in group_results.items():
                results[f"{group}/{name}"] = success
                
        return results
    
    # 按组分类数据集
    grouped_datasets = {}
    for identifier in identifiers:
        group, dataset_id = parse_dataset_identifier(identifier)
        
        # 检查数据集组是否存在
        if group not in DATASET_GROUPS:
            results[identifier] = False
            print(f"警告: 未知的数据集组 '{group}'")
            continue
            
        if group not in grouped_datasets:
            grouped_datasets[group] = []
            
        # 如果只指定了组，则下载该组的所有数据集
        if dataset_id is None:
            # 在结果中标记为组下载
            results[group] = True
        else:
            grouped_datasets[group].append(dataset_id)
    
    # 按组下载数据集
    for group, dataset_ids in grouped_datasets.items():
        download_func = DATASET_GROUPS[group]["download"]
        
        # 如果有任何只指定组的标识符，下载该组的所有数据集
        if group in results and results[group]:
            dataset_ids = None
            
        # 下载数据集
        group_results = download_func(dataset_ids, data_root=data_root, 
                                    max_workers=max_workers, **kwargs)
        
        # 更新结果
        for name, success in group_results.items():
            results[f"{group}/{name}"] = success
    
    return results

__all__ = ["list_datasets", "get_dataset_info", "load_dataset", "download_datasets"]