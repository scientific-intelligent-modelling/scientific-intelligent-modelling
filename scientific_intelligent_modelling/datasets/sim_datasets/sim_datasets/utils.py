"""
通用工具函数
"""
from typing import List, Dict, Union, Optional, Tuple, Any
import os
import pathlib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
import time
import random
import pandas as pd
import json

def ensure_dir(directory: Union[str, pathlib.Path]) -> None:
    """确保目录存在，不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def get_dataset_dir(data_root: Union[str, pathlib.Path], dataset_group: str) -> pathlib.Path:
    """
    获取数据集组的存储目录
    
    Args:
        data_root: 数据根目录
        dataset_group: 数据集组名称(feynman, pmlb等)
    
    Returns:
        pathlib.Path: 数据集组的目录路径
    """
    dataset_dir = pathlib.Path(data_root) / dataset_group
    ensure_dir(dataset_dir)
    return dataset_dir

def download_file(url: str, target_path: Union[str, pathlib.Path], 
                  chunk_size: int = 8192, timeout: int = 30) -> None:
    """
    从URL下载文件到指定路径
    
    Args:
        url: 源文件URL
        target_path: 目标保存路径
        chunk_size: 下载块大小
        timeout: 连接超时时间(秒)
    """
    # 确保目标目录存在
    target_dir = os.path.dirname(str(target_path))
    ensure_dir(target_dir)
    
    # 下载文件
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        file_size = int(response.headers.get('content-length', 0))
        
        with open(target_path, 'wb') as f:
            with tqdm.tqdm(
                total=file_size, 
                unit='B', 
                unit_scale=True,
                desc=f"下载 {os.path.basename(str(target_path))}"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

def parallel_download(
    urls_and_paths: List[Tuple[str, Union[str, pathlib.Path]]],
    max_workers: int = 4, 
    chunk_size: int = 8192,
    timeout: int = 30,
    retry_count: int = 3,
    retry_delay: int = 5
) -> Dict[str, bool]:
    """
    并行下载多个文件
    
    Args:
        urls_and_paths: 包含(URL, 目标路径)的元组列表
        max_workers: 最大工作线程数
        chunk_size: 下载块大小
        timeout: 下载超时时间(秒)
        retry_count: 下载失败重试次数
        retry_delay: 重试之间的延迟(秒)
    
    Returns:
        Dict[str, bool]: 文件路径到下载状态的映射
    """
    result = {}
    
    def download_with_retry(url: str, path: Union[str, pathlib.Path]) -> Tuple[str, bool]:
        """带重试的下载单个文件"""
        path_str = str(path)
        for attempt in range(retry_count):
            try:
                download_file(url, path, chunk_size, timeout)
                return path_str, True
            except Exception as e:
                if attempt < retry_count - 1:
                    tqdm.tqdm.write(f"下载 {path_str} 失败 (尝试 {attempt+1}/{retry_count}): {str(e)}")
                    time.sleep(retry_delay)
                else:
                    tqdm.tqdm.write(f"下载 {path_str} 最终失败: {str(e)}")
                    return path_str, False
    
    # 并行执行下载任务
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(download_with_retry, url, path): path 
            for url, path in urls_and_paths
        }
        
        for future in as_completed(future_to_path):
            path_str, success = future.result()
            result[path_str] = success
    
    return result

def get_sample_data(df: pd.DataFrame, n_samples: int = 5) -> pd.DataFrame:
    """
    从DataFrame中获取样本数据
    
    Args:
        df: 数据框
        n_samples: 返回的样本数量
        
    Returns:
        pd.DataFrame: 样本数据
    """
    if len(df) <= n_samples:
        return df
    return df.sample(n_samples, random_state=42)

def generate_mock_data(features: List[str], target: str, n_samples: int = 10) -> pd.DataFrame:
    """
    生成模拟数据用于展示
    
    Args:
        features: 特征列名
        target: 目标列名
        n_samples: 样本数量
        
    Returns:
        pd.DataFrame: 生成的模拟数据
    """
    # 为每个特征生成随机值
    data = {}
    for feature in features:
        data[feature] = [round(random.uniform(-10, 10), 2) for _ in range(n_samples)]
    
    # 生成目标值（仅用于示例）
    data[target] = [round(random.uniform(-100, 100), 2) for _ in range(n_samples)]
    
    return pd.DataFrame(data)

def parse_dataset_identifier(identifier: str) -> Tuple[str, Optional[str]]:
    """
    解析数据集标识符
    
    Args:
        identifier: 数据集标识符，格式为"组/ID"或"组"
        
    Returns:
        Tuple[str, Optional[str]]: (数据集组, 数据集标识符) 元组
    """
    parts = identifier.split("/", 1)
    if len(parts) == 1:
        return parts[0], None
    else:
        return parts[0], parts[1]

def save_json(data: Dict, file_path: Union[str, pathlib.Path]) -> None:
    """
    将数据保存为JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(file_path: Union[str, pathlib.Path]) -> Dict:
    """
    加载JSON文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        Dict: 加载的数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)