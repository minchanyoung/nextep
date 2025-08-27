# app/prompt_templates.py
"""
프롬프트 엔지니어링 템플릿 시스템
체계화된 프롬프트 템플릿으로 AI 응답 품질 향상
"""

from typing import Dict, List, Optional
import json

class PromptTemplateManager:
    """프롬프트 템플릿 관리 클래스"""
    
    def __init__(self):
        self.templates = {
            "career_coach_system": {
                "role": "전문 커리어 코치",
                "expertise": "한국 노동시장 전문가, ML 기반 예측 분석가, 개인 맞춤형 상담사",
                "personality": "친절하고 전문적이며 실용적인 조언 제공",
                "guidelines": [
                    "예측 데이터와 노동시장 트렌드를 종합하여 객관적 분석 제공",
                    "5가지 이상의 구체적이고 실행 가능한 조언 제시",
                    "수치와 근거를 포함한 신뢰할 수 있는 정보 전달",
                    "개인의 상황과 선호도를 고려한 맞춤형 솔루션 제공",
                    "동어반복 지양, 간결하고 명확한 표현 사용"
                ]
            },
            
            "follow_up_coach_system": {
                "role": "지속적 멘토링 코치",
                "context_awareness": "이전 대화 맥락을 충분히 이해하고 연속성 있는 조언 제공",
                "response_style": "간결하면서도 심층적인 후속 질문에 대한 답변",
                "focus": "실행 계획, 구체적 방법론, 추가 리소스 제공"
            }
        }
    
    def get_career_coach_system_prompt(self, user_context: Dict = None) -> str:
        """커리어 코치 시스템 프롬프트 생성"""
        template = self.templates["career_coach_system"]
        
        base_prompt = f"""당신은 {template['role']}이자 {template['expertise']}입니다.

【역할 및 전문성】
- 한국 노동시장 분석 전문가
- 머신러닝 기반 커리어 예측 분석가  
- 개인 맞춤형 커리어 상담 전문가
- 2024-2025 노동시장 동향 분석 전문가

【응답 가이드라인】
"""
        
        for i, guideline in enumerate(template["guidelines"], 1):
            base_prompt += f"{i}. {guideline}\n"
        
        base_prompt += """
【응답 구조】
1. 상황 분석: 예측 결과와 개인 상황 종합 분석
2. 핵심 인사이트: 데이터 기반 주요 발견사항 (3-5개)
3. 실행 전략: 구체적이고 실행 가능한 액션 플랜
4. 위험 요소: 고려해야 할 잠재적 리스크
5. 성공 지표: 진행 상황을 측정할 수 있는 지표

【금지 사항】
- 모호하거나 일반적인 조언 금지
- "추가 정보가 필요합니다" 등의 메타 발화 금지
- 동일한 내용 반복 금지
- 근거 없는 추측 금지
- #, *, - 등 마크다운 문법 금지"""

        if user_context:
            base_prompt += f"\n\n【사용자 맥락】\n{self._format_user_context(user_context)}"
        
        return base_prompt
    
    def get_follow_up_system_prompt(self) -> str:
        """후속 질문용 시스템 프롬프트 생성"""
        template = self.templates["follow_up_coach_system"]
        
        return f"""당신은 {template['role']}입니다.

【핵심 원칙】
- 이전 대화의 맥락을 완전히 이해하고 연속성 있는 조언 제공
- 사용자의 구체적인 질문에 직접적이고 실용적으로 답변
- 실행 가능한 구체적인 방법론과 리소스 제공
- 3-4개의 핵심 포인트로 간결하게 구성

【응답 스타일】
- 직접적이고 실행 중심적
- 구체적인 수치, 기간, 방법 포함
- 단계별 실행 계획 제시
- 관련 리소스나 도구 추천

【금지 사항】
- 이전 조언의 단순 반복
- 일반론적 답변
- 장황한 설명"""
    
    def get_career_advice_prompt(self, user_input: Dict, prediction_results: List[Dict], 
                                job_category_map: Dict, satis_factor_map: Dict,
                                labor_market_context: str = "", learning_context: str = "") -> List[Dict]:
        """커리어 조언 생성용 메시지 구성"""
        
        # 사용자 정보 포맷팅
        current_job_name = job_category_map.get(user_input.get('current_job_category', ''), '알 수 없음')
        job_a_name = job_category_map.get(user_input.get('job_A_category', ''), '알 수 없음')
        job_b_name = job_category_map.get(user_input.get('job_B_category', ''), '알 수 없음')
        focus_key_name = satis_factor_map.get(user_input.get('satis_focus_key'), '지정되지 않음')
        gender_text = '여성' if str(user_input.get('gender')) == '1' else '남성'
        
        # 예측 결과 포맷팅
        p0, p1, p2 = prediction_results
        prediction_details = f"""
【예측 결과 분석】
▪ 현직 유지({current_job_name}): 
  - 소득 변화율: {p0['income_change_rate']:.2%}
  - 만족도 변화: {p0['satisfaction_change_score']:.2f}점

▪ 이직 옵션 A({job_a_name}):
  - 소득 변화율: {p1['income_change_rate']:.2%}  
  - 만족도 변화: {p1['satisfaction_change_score']:.2f}점

▪ 이직 옵션 B({job_b_name}):
  - 소득 변화율: {p2['income_change_rate']:.2%}
  - 만족도 변화: {p2['satisfaction_change_score']:.2f}점"""

        # 사용자 프로필
        user_profile = f"""
【사용자 프로필】
▪ 나이: {user_input.get('age', 'N/A')}세
▪ 성별: {gender_text}  
▪ 현재 직업: {current_job_name}
▪ 현재 월소득: {user_input.get('monthly_income', 'N/A')}만원
▪ 중요 만족 요인: {focus_key_name}"""

        # RAG 컨텍스트 추가
        rag_context = ""
        if labor_market_context:
            rag_context += f"\n【노동시장 동향 분석】\n{labor_market_context}"
        if learning_context:
            rag_context += f"\n【추천 역량 개발 방향】\n{learning_context}"
        
        # 시스템 프롬프트
        system_prompt = self.get_career_coach_system_prompt()
        
        # 사용자 프롬프트
        user_prompt = f"""{user_profile}

{prediction_details}
{rag_context}

【분석 요청】
위 예측 결과와 노동시장 동향을 종합하여 다음 사항에 대해 전문적인 분석과 조언을 제공해주세요:

1. 각 시나리오별 상세 분석 (장단점, 위험요소)
2. 현재 노동시장 트렌드를 고려한 최적 선택 제안
3. 선택한 경로별 구체적 실행 전략 (6개월/1년 계획)
4. {focus_key_name} 만족도 향상을 위한 특별 전략  
5. 성공 확률을 높이기 위한 핵심 역량 개발 방향"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def get_rag_enhanced_context(self, query: str, rag_results: List[Dict]) -> str:
        """RAG 검색 결과를 컨텍스트로 포맷팅"""
        if not rag_results:
            return ""
        
        context_parts = []
        context_parts.append(f"【사용자 질문 관련 심층 분석: '{query}'】")

        for i, result in enumerate(rag_results, 1): # 모든 결과 사용
            source = result['metadata'].get('source', 'unknown')
            content = result['content'] # 전체 내용 사용
            score = result.get('score', 0)
            chunk_type = result['metadata'].get('chunk_type', '일반')
            keywords = ", ".join(result['metadata'].get('keywords', []))

            source_info = ""
            if source.endswith('.pdf'):
                page = result['metadata'].get('page', 0)
                source_info = f"출처: {source} (페이지: {page})"
            elif source.startswith('legacy_'):
                source_info = f"출처: 내부 전문 데이터 ({source.replace('legacy_', '')})"
            else:
                source_info = f"출처: {source}"

            context_parts.append(f"\n--- 관련 정보 {i} (유사도: {1-score:.1%}) ---")
            context_parts.append(f"유형: {chunk_type}")
            if keywords:
                context_parts.append(f"핵심 키워드: {keywords}")
            context_parts.append(f"{source_info}")
            context_parts.append(f"내용:\n{content}")
            context_parts.append("------------------------------------")
        
        return "\n".join(context_parts)
    
    def _format_user_context(self, context: Dict) -> str:
        """사용자 컨텍스트를 포맷팅"""
        formatted = ""
        for key, value in context.items():
            if isinstance(value, dict):
                formatted += f"- {key}: {json.dumps(value, ensure_ascii=False)}\n"
            else:
                formatted += f"- {key}: {value}\n"
        return formatted

# 싱글톤 인스턴스
prompt_manager = PromptTemplateManager()