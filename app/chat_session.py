# app/chat_session.py
"""
챗봇 멀티턴 대화 세션 관리 시스템
사용자별 대화 컨텍스트를 유지하고 관리
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from flask import session, current_app

logger = logging.getLogger(__name__)

class ChatSession:
    """개별 채팅 세션 관리 클래스"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages = []
        self.context = {}
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.user_profile = {}
        self.prediction_context = {}
        
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """메시지 추가"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.messages.append(message)
        self.last_activity = datetime.now()
        
    def get_messages(self, limit: Optional[int] = None) -> List[Dict]:
        """최근 메시지 반환"""
        if limit:
            return self.messages[-limit:]
        return self.messages
    
    def get_context_summary(self) -> str:
        """대화 컨텍스트 요약 반환"""
        if not self.prediction_context:
            return "사용자 정보가 없습니다."
        
        summary_parts = []

        # 사용자 프로필 정보
        if self.user_profile:
            profile_items = []
            # 원하는 순서대로 프로필 정보 추가
            order = ['age', 'gender', 'current_job', 'income', 'focus_factor']
            for key in order:
                if key in self.user_profile:
                    # 한글 키 이름으로 변경
                    key_map = {
                        'age': '나이', 'gender': '성별', 'current_job': '현재 직업',
                        'income': '월소득', 'focus_factor': '중요 가치'
                    }
                    profile_items.append(f"{key_map.get(key, key)}: {self.user_profile[key]}")
            if profile_items:
                summary_parts.append(f"사용자 프로필: {', '.join(profile_items)}")

        # 예측 결과 요약
        if 'prediction_summary' in self.prediction_context:
            # prediction_summary가 이미 충분한 정보를 담고 있으므로 그대로 사용
            # 필요하다면 여기서 추가 가공
            summary_parts.append(f"AI 예측 요약: {self.prediction_context['prediction_summary']}")

        return "\n".join(summary_parts)
    
    def set_user_context(self, user_input: Dict, prediction_results: List[Dict], 
                        job_category_map: Dict, satis_factor_map: Dict):
        """사용자 컨텍스트 설정"""
        from app.services import summarize_context_hf
        
        self.prediction_context = {
            'user_input': user_input,
            'prediction_results': prediction_results,
            'job_category_map': job_category_map,
            'satis_factor_map': satis_factor_map,
            'prediction_summary': summarize_context_hf(user_input, prediction_results, job_category_map, satis_factor_map)
        }
        
        # 사용자 프로필 정보 저장
        self.user_profile = {
            'age': user_input.get('age', 'N/A'),
            'gender': '여성' if str(user_input.get('gender')) == '1' else '남성',
            'current_job': job_category_map.get(user_input.get('current_job_category', ''), '알 수 없음'),
            'income': user_input.get('monthly_income', 'N/A'),
            'focus_factor': satis_factor_map.get(user_input.get('satis_focus_key'), '지정되지 않음')
        }
    
    def is_expired(self, hours: int = 24) -> bool:
        """세션 만료 확인"""
        return datetime.now() - self.last_activity > timedelta(hours=hours)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (세션 저장용)"""
        return {
            'session_id': self.session_id,
            'messages': self.messages,
            'context': self.context,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'user_profile': self.user_profile,
            'prediction_context': self.prediction_context
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """딕셔너리에서 복원"""
        session_obj = cls(data['session_id'])
        session_obj.messages = data.get('messages', [])
        session_obj.context = data.get('context', {})
        session_obj.created_at = datetime.fromisoformat(data['created_at'])
        session_obj.last_activity = datetime.fromisoformat(data['last_activity'])
        session_obj.user_profile = data.get('user_profile', {})
        session_obj.prediction_context = data.get('prediction_context', {})
        return session_obj

class ChatSessionManager:
    """채팅 세션 매니저"""
    
    def __init__(self):
        self.sessions = {}  # 메모리 기반 세션 저장
        
    def get_or_create_session(self, session_id: str) -> ChatSession:
        """세션 가져오기 또는 생성"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(session_id)
        elif self.sessions[session_id].is_expired():
            logger.info(f"만료된 세션 제거: {session_id}")
            del self.sessions[session_id]
            self.sessions[session_id] = ChatSession(session_id)
        
        return self.sessions[session_id]
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """기존 세션 가져오기"""
        if session_id in self.sessions and not self.sessions[session_id].is_expired():
            return self.sessions[session_id]
        return None
    
    def remove_session(self, session_id: str):
        """세션 제거"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        expired_sessions = []
        for session_id, chat_session in self.sessions.items():
            if chat_session.is_expired():
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            logger.info(f"만료된 세션 {len(expired_sessions)}개 정리 완료")
    
    def get_session_stats(self) -> Dict:
        """세션 통계"""
        total_sessions = len(self.sessions)
        active_sessions = sum(1 for s in self.sessions.values() if not s.is_expired())
        total_messages = sum(len(s.messages) for s in self.sessions.values())
        
        return {
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'total_messages': total_messages
        }

# 전역 세션 매니저 인스턴스
chat_session_manager = ChatSessionManager()

def get_current_chat_session() -> ChatSession:
    """현재 Flask 세션의 채팅 세션 반환"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            # 새로운 세션 ID 생성
            import uuid
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
        
        return chat_session_manager.get_or_create_session(session_id)
    except Exception as e:
        logger.error(f"채팅 세션 가져오기 실패: {e}")
        # 폴백: 임시 세션 생성
        import uuid
        session_id = str(uuid.uuid4())
        return chat_session_manager.get_or_create_session(session_id)

def clear_chat_session():
    """현재 채팅 세션 초기화"""
    session_id = session.get('session_id')
    if session_id:
        chat_session_manager.remove_session(session_id)
        # 새로운 세션 ID 생성
        import uuid
        new_session_id = str(uuid.uuid4())
        session['session_id'] = new_session_id

def get_conversation_context(include_prediction: bool = True) -> List[Dict]:
    """대화 컨텍스트를 LLM 메시지 형식으로 반환"""
    chat_session = get_current_chat_session()
    
    messages = []
    
    # 예측 컨텍스트가 있으면 시스템 메시지에 포함
    if include_prediction and chat_session.prediction_context:
        context_summary = chat_session.get_context_summary()
        if context_summary:
            messages.append({
                "role": "system", 
                "content": f"사용자 컨텍스트: {context_summary}"
            })
    
    # 최근 대화 내역 추가 (최대 20개 메시지)
    recent_messages = chat_session.get_messages(limit=20)
    for msg in recent_messages:
        if msg['role'] in ['user', 'assistant']:
            messages.append({
                "role": msg['role'],
                "content": msg['content']
            })
    
    return messages