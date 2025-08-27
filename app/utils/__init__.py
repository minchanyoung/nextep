# app/utils/__init__.py
"""
Utility modules for common functionality
"""

from .cache_manager import get_cache_manager, RAGCacheManager
from .error_handler import get_error_handler, handle_exceptions, handle_service_exceptions
from .validators import UserProfileValidator, MLInputValidator
from .helpers import PerformanceTimer, generate_hash, format_currency, truncate_text
from .monitoring import get_metrics_collector, get_health_checker, record_request_metric
from .math_utils import cosine_similarity, euclidean_distance, normalize_vector
from .flask_utils import get_llm_service, get_rag_manager, is_app_context_available

__all__ = [
    'get_cache_manager', 'RAGCacheManager',
    'get_error_handler', 'handle_exceptions', 'handle_service_exceptions',
    'UserProfileValidator', 'MLInputValidator',
    'PerformanceTimer', 'generate_hash', 'format_currency', 'truncate_text',
    'get_metrics_collector', 'get_health_checker', 'record_request_metric',
    'cosine_similarity', 'euclidean_distance', 'normalize_vector',
    'get_llm_service', 'get_rag_manager', 'is_app_context_available'
]