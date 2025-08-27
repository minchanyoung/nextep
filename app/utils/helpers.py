# app/utils/helpers.py
"""
공통 헬퍼 함수들
"""

import hashlib
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from flask import current_app


def generate_hash(text: str, algorithm: str = 'md5', length: Optional[int] = None) -> str:
    """문자열 해시 생성"""
    if algorithm == 'md5':
        hash_obj = hashlib.md5(text.encode('utf-8'))
    elif algorithm == 'sha256':
        hash_obj = hashlib.sha256(text.encode('utf-8'))
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    hash_str = hash_obj.hexdigest()
    return hash_str[:length] if length else hash_str


def format_currency(amount: Union[int, float], currency: str = 'KRW') -> str:
    """통화 형식으로 포매팅"""
    if currency == 'KRW':
        return f"{amount:,}원"
    elif currency == 'USD':
        return f"${amount:,.2f}"
    else:
        return f"{amount:,} {currency}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """퍼센트 형식으로 포매팅"""
    return f"{value:.{decimal_places}%}"


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """텍스트 자르기"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_dict_get(data: Dict, keys: List[str], default: Any = None) -> Any:
    """중첩된 딕셔너리에서 안전하게 값 가져오기"""
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def merge_dicts(*dicts: Dict) -> Dict:
    """딕셔너리 병합"""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """리스트를 청크로 분할"""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def time_ago(timestamp: datetime) -> str:
    """상대 시간 표시"""
    now = datetime.utcnow()
    diff = now - timestamp
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "방금 전"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        return f"{minutes}분 전"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours}시간 전"
    elif seconds < 2592000:  # 30 days
        days = int(seconds // 86400)
        return f"{days}일 전"
    else:
        return timestamp.strftime("%Y-%m-%d")


def sanitize_filename(filename: str) -> str:
    """파일명 안전하게 처리"""
    import re
    # 위험한 문자 제거
    filename = re.sub(r'[^\w\s.-]', '', filename)
    # 연속된 공백을 단일 공백으로
    filename = re.sub(r'\s+', ' ', filename)
    # 앞뒤 공백 제거
    return filename.strip()


class PerformanceTimer:
    """성능 측정 타이머"""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        current_app.logger.debug(f"{self.name}: {duration:.4f}초 소요")
    
    @property
    def duration(self) -> float:
        """실행 시간 반환"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


class RateLimiter:
    """간단한 레이트 리미터"""
    
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """요청 허용 여부 확인"""
        now = time.time()
        
        # 오래된 요청 기록 정리
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if now - req_time < self.time_window
            ]
        
        # 현재 요청 수 확인
        current_requests = len(self.requests.get(identifier, []))
        
        if current_requests >= self.max_requests:
            return False
        
        # 요청 기록
        if identifier not in self.requests:
            self.requests[identifier] = []
        self.requests[identifier].append(now)
        
        return True


def get_client_ip() -> str:
    """클라이언트 IP 주소 가져오기"""
    from flask import request
    
    # Proxy 헤더 확인
    if request.environ.get('HTTP_X_FORWARDED_FOR'):
        return request.environ['HTTP_X_FORWARDED_FOR'].split(',')[0].strip()
    elif request.environ.get('HTTP_X_REAL_IP'):
        return request.environ['HTTP_X_REAL_IP']
    else:
        return request.environ.get('REMOTE_ADDR', 'unknown')


def is_valid_json(text: str) -> bool:
    """유효한 JSON 문자열인지 확인"""
    try:
        import json
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def deep_merge_dict(dict1: Dict, dict2: Dict) -> Dict:
    """딕셔너리 깊은 병합"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    
    return result