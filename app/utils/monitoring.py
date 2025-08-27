# app/utils/monitoring.py
"""
모니터링 및 메트릭 수집
"""

import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricData:
    """메트릭 데이터"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ServiceHealth:
    """서비스 건강 상태"""
    service_name: str
    status: str  # healthy, warning, critical
    last_check: datetime
    response_time: float
    error_rate: float
    message: str = ""


class MetricsCollector:
    """메트릭 수집기"""
    
    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        self.metrics = defaultdict(lambda: deque(maxlen=max_points))
        self.counters = defaultdict(int)
        self.timers = {}
        self.lock = threading.RLock()
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """메트릭 기록"""
        with self.lock:
            metric = MetricData(name=name, value=value, tags=tags or {})
            self.metrics[name].append(metric)
    
    def increment_counter(self, name: str, value: int = 1):
        """카운터 증가"""
        with self.lock:
            self.counters[name] += value
    
    def start_timer(self, name: str) -> str:
        """타이머 시작"""
        timer_id = f"{name}_{time.time()}"
        self.timers[timer_id] = time.time()
        return timer_id
    
    def end_timer(self, timer_id: str) -> float:
        """타이머 종료 및 소요시간 반환"""
        if timer_id in self.timers:
            duration = time.time() - self.timers[timer_id]
            del self.timers[timer_id]
            
            # 타이머 이름 추출
            timer_name = timer_id.split('_')[0]
            self.record_metric(f"{timer_name}_duration", duration)
            
            return duration
        return 0.0
    
    def get_metrics(self, name: str, minutes: int = 60) -> List[MetricData]:
        """특정 기간의 메트릭 조회"""
        with self.lock:
            if name not in self.metrics:
                return []
            
            cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
            return [
                metric for metric in self.metrics[name]
                if metric.timestamp >= cutoff_time
            ]
    
    def get_counter_value(self, name: str) -> int:
        """카운터 값 조회"""
        with self.lock:
            return self.counters[name]
    
    def get_average(self, name: str, minutes: int = 60) -> float:
        """평균값 계산"""
        metrics = self.get_metrics(name, minutes)
        if not metrics:
            return 0.0
        return sum(m.value for m in metrics) / len(metrics)
    
    def get_summary(self) -> Dict[str, Any]:
        """메트릭 요약"""
        with self.lock:
            summary = {
                'counters': dict(self.counters),
                'metrics_summary': {},
                'active_timers': len(self.timers)
            }
            
            for name, metric_deque in self.metrics.items():
                if metric_deque:
                    values = [m.value for m in metric_deque]
                    summary['metrics_summary'][name] = {
                        'count': len(values),
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'latest': values[-1]
                    }
            
            return summary


class HealthChecker:
    """서비스 건강 상태 검사"""
    
    def __init__(self):
        self.health_status = {}
        self.lock = threading.RLock()
    
    def check_llm_service(self) -> ServiceHealth:
        """LLM 서비스 건강 상태 확인"""
        start_time = time.time()
        
        try:
            from app.llm_service import LLMService
            from flask import current_app
            
            llm_service = current_app.extensions.get("llm_service")
            if not llm_service:
                return ServiceHealth(
                    service_name="LLM",
                    status="critical",
                    last_check=datetime.utcnow(),
                    response_time=0,
                    error_rate=1.0,
                    message="LLM 서비스가 초기화되지 않았습니다."
                )
            
            # 간단한 테스트 요청
            test_messages = [{"role": "user", "content": "테스트"}]
            response = llm_service.chat_sync(test_messages)
            
            response_time = time.time() - start_time
            
            if response and "오류" not in response:
                status = "healthy" if response_time < 5.0 else "warning"
                return ServiceHealth(
                    service_name="LLM",
                    status=status,
                    last_check=datetime.utcnow(),
                    response_time=response_time,
                    error_rate=0.0
                )
            else:
                return ServiceHealth(
                    service_name="LLM",
                    status="warning",
                    last_check=datetime.utcnow(),
                    response_time=response_time,
                    error_rate=0.5,
                    message="LLM 응답에 오류 메시지가 포함되어 있습니다."
                )
                
        except Exception as e:
            return ServiceHealth(
                service_name="LLM",
                status="critical",
                last_check=datetime.utcnow(),
                response_time=time.time() - start_time,
                error_rate=1.0,
                message=f"LLM 서비스 오류: {str(e)}"
            )
    
    def check_database(self) -> ServiceHealth:
        """데이터베이스 건강 상태 확인"""
        start_time = time.time()
        
        try:
            from app import db
            
            # 간단한 쿼리 실행
            db.session.execute(db.text("SELECT 1 FROM DUAL"))
            
            response_time = time.time() - start_time
            status = "healthy" if response_time < 1.0 else "warning"
            
            return ServiceHealth(
                service_name="Database",
                status=status,
                last_check=datetime.utcnow(),
                response_time=response_time,
                error_rate=0.0
            )
            
        except Exception as e:
            return ServiceHealth(
                service_name="Database",
                status="critical",
                last_check=datetime.utcnow(),
                response_time=time.time() - start_time,
                error_rate=1.0,
                message=f"데이터베이스 오류: {str(e)}"
            )
    
    def check_all_services(self) -> Dict[str, ServiceHealth]:
        """모든 서비스 건강 상태 확인"""
        services = {}
        
        try:
            services['llm'] = self.check_llm_service()
        except Exception as e:
            logger.error(f"LLM 서비스 건강 확인 실패: {e}")
        
        try:
            services['database'] = self.check_database()
        except Exception as e:
            logger.error(f"데이터베이스 건강 확인 실패: {e}")
        
        with self.lock:
            self.health_status.update(services)
        
        return services
    
    def get_overall_status(self) -> str:
        """전체 시스템 상태 반환"""
        if not self.health_status:
            return "unknown"
        
        statuses = [health.status for health in self.health_status.values()]
        
        if "critical" in statuses:
            return "critical"
        elif "warning" in statuses:
            return "warning"
        elif all(status == "healthy" for status in statuses):
            return "healthy"
        else:
            return "unknown"


# 글로벌 인스턴스
_metrics_collector = MetricsCollector()
_health_checker = HealthChecker()


def get_metrics_collector() -> MetricsCollector:
    """메트릭 수집기 인스턴스 반환"""
    return _metrics_collector


def get_health_checker() -> HealthChecker:
    """건강 검사기 인스턴스 반환"""
    return _health_checker


def record_request_metric(endpoint: str, method: str, status_code: int, duration: float):
    """HTTP 요청 메트릭 기록"""
    collector = get_metrics_collector()
    collector.record_metric("http_request_duration", duration, {
        'endpoint': endpoint,
        'method': method,
        'status_code': str(status_code)
    })
    collector.increment_counter(f"http_requests_total_{status_code}")


def record_ml_prediction_metric(model_name: str, duration: float, success: bool):
    """ML 예측 메트릭 기록"""
    collector = get_metrics_collector()
    collector.record_metric("ml_prediction_duration", duration, {
        'model': model_name,
        'success': str(success)
    })
    
    if success:
        collector.increment_counter("ml_predictions_success")
    else:
        collector.increment_counter("ml_predictions_error")