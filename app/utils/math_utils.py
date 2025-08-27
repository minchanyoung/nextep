# app/utils/math_utils.py
"""
수학 및 벡터 연산 유틸리티
"""

import numpy as np
from typing import List, Union


def cosine_similarity(vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]) -> float:
    """
    두 벡터 간의 코사인 유사도를 계산합니다.
    
    Args:
        vec1: 첫 번째 벡터
        vec2: 두 번째 벡터
    
    Returns:
        float: 코사인 유사도 (-1 ~ 1)
    """
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    
    if len(vec1) != len(vec2):
        raise ValueError("벡터의 차원이 일치하지 않습니다.")
    
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


def euclidean_distance(vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]) -> float:
    """
    두 벡터 간의 유클리드 거리를 계산합니다.
    
    Args:
        vec1: 첫 번째 벡터
        vec2: 두 번째 벡터
    
    Returns:
        float: 유클리드 거리
    """
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    
    if len(vec1) != len(vec2):
        raise ValueError("벡터의 차원이 일치하지 않습니다.")
    
    return float(np.linalg.norm(vec1 - vec2))


def normalize_vector(vec: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    벡터를 정규화합니다 (단위 벡터로 변환).
    
    Args:
        vec: 입력 벡터
    
    Returns:
        np.ndarray: 정규화된 벡터
    """
    vec = np.array(vec, dtype=float)
    norm = np.linalg.norm(vec)
    
    if norm == 0:
        return vec
    
    return vec / norm


def manhattan_distance(vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]) -> float:
    """
    두 벡터 간의 맨해튼 거리를 계산합니다.
    
    Args:
        vec1: 첫 번째 벡터
        vec2: 두 번째 벡터
    
    Returns:
        float: 맨해튼 거리
    """
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    
    if len(vec1) != len(vec2):
        raise ValueError("벡터의 차원이 일치하지 않습니다.")
    
    return float(np.sum(np.abs(vec1 - vec2)))