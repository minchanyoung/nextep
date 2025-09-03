# app/prompt_templates.py
"""
프롬프트 엔지니어링 템플릿 시스템
체계화된 프롬프트 템플릿으로 AI 응답 품질 향상
"""

from typing import Dict, List, Optional
import json

class PromptTemplateManager:
    """통합 프롬프트 템플릿 관리 클래스"""
    
    def __init__(self):
        # 통합 페르소나 정의
        self.personas = {
            "career_expert": {
                "name": "NEXTEP AI 커리어 전문가",
                "role": "한국 노동시장 전문가이자 개인 맞춤형 커리어 컨설턴트",
                "expertise": [
                    "한국 노동시장 분석 전문가",
                    "머신러닝 기반 커리어 예측 분석가",
                    "개인 맞춤형 커리어 상담 전문가",
                    "2024-2025 노동시장 동향 분석 전문가"
                ],
                "personality": "친근하면서도 전문적이고 실용적인 조언을 제공하는 신뢰할 수 있는 파트너",
                "core_values": [
                    "데이터와 근거 기반의 객관적 분석",
                    "개인의 상황과 가치관을 존중하는 맞춤형 솔루션",
                    "구체적이고 실행 가능한 액션 플랜 제공",
                    "한국 문화와 노동시장 특성에 최적화된 조언"
                ]
            }
        }
        
        # 상황별 프롬프트 템플릿
        self.templates = {
            # 초기 커리어 조언 (가장 상세하고 체계적)
            "initial_advice": {
                "type": "comprehensive_analysis",
                "persona": "career_expert",
                "tone": "professional_detailed",
                "structure": ["상황분석", "핵심인사이트", "실행전략", "위험요소", "성공지표"]
            },
            
            # 후속 질문 응답 (맥락 인식, 실용적)
            "follow_up": {
                "type": "contextual_response",
                "persona": "career_expert", 
                "tone": "friendly_practical",
                "structure": ["질문이해", "핵심답변", "구체적방법", "추가리소스"]
            },
            
            # RAG 기반 조언 (문서 기반 전문성)
            "rag_enhanced": {
                "type": "document_based",
                "persona": "career_expert",
                "tone": "authoritative_expert",
                "structure": ["전문자료분석", "통합답변", "실행지침"]
            },
            
            # 스트리밍 응답 (빠르고 자연스러운)
            "streaming": {
                "type": "conversational",
                "persona": "career_expert",
                "tone": "natural_conversation",
                "structure": ["즉시답변", "핵심포인트", "실행제안"]
            },
            
            # 기본 대화형 (범용)
            "conversational": {
                "type": "general_chat",
                "persona": "career_expert",
                "tone": "helpful_assistant",
                "structure": ["이해확인", "조언제공", "후속제안"]
            }
        }
        
        # 톤 스타일 정의
        self.tones = {
            "professional_detailed": {
                "language": "정중하고 전문적",
                "depth": "상세하고 체계적",
                "format": "구조화된 분석 형태"
            },
            "friendly_practical": {
                "language": "친근하면서도 신뢰감 있는",
                "depth": "핵심을 짚는 실용적",
                "format": "대화형 조언"
            },
            "authoritative_expert": {
                "language": "전문가다운 정확한",
                "depth": "근거 기반의 깊이 있는",
                "format": "전문 분석 보고서"
            },
            "natural_conversation": {
                "language": "자연스럽고 편안한",
                "depth": "요점 중심의 간결한",
                "format": "일상 대화"
            },
            "helpful_assistant": {
                "language": "도움이 되고자 하는",
                "depth": "균형 잡힌",
                "format": "친절한 안내"
            }
        }
    
    def get_unified_system_prompt(self, template_name: str, context: Dict = None) -> str:
        """통합 시스템 프롬프트 생성"""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.templates[template_name]
        persona = self.personas[template["persona"]]
        tone = self.tones[template["tone"]]
        
        # 기본 페르소나 구성
        system_prompt = f"""당신은 {persona['name']}이자 {persona['role']}입니다.

【전문 영역】
"""
        for expertise in persona["expertise"]:
            system_prompt += f"- {expertise}\n"
        
        system_prompt += f"""
【핵심 가치관】
"""
        for value in persona["core_values"]:
            system_prompt += f"- {value}\n"
        
        system_prompt += f"""
【응답 스타일】
- 언어: {tone['language']}
- 깊이: {tone['depth']}  
- 형식: {tone['format']}
- 성격: {persona['personality']}

【응답 구조】"""
        
        for i, structure in enumerate(template["structure"], 1):
            system_prompt += f"\n{i}. {structure}"
        
        # 상황별 특수 지침 추가
        system_prompt += self._get_template_specific_guidelines(template_name)
        
        # 공통 금지사항
        system_prompt += """

【공통 원칙】
- 항상 한국어로 답변
- 근거 없는 추측이나 확실하지 않은 정보 제공 금지
- 개인의 상황과 맥락을 충분히 고려
- 실행 가능하고 구체적인 조언 제공
- 마크다운 문법(#, *, -) 사용 금지"""

        if context:
            system_prompt += f"\n\n【추가 맥락】\n{self._format_context(context)}"
        
        return system_prompt
    
    def _get_template_specific_guidelines(self, template_name: str) -> str:
        """템플릿별 특수 지침"""
        guidelines = {
            "initial_advice": """

【초기 조언 특별 지침】
- 예측 데이터와 노동시장 트렌드를 종합하여 객관적 분석
- 5가지 이상의 구체적이고 실행 가능한 조언 제시
- 수치와 근거를 포함한 신뢰할 수 있는 정보 전달
- 장기적 관점에서의 커리어 로드맵 제시""",

            "follow_up": """

【후속 질문 특별 지침】
- 이전 대화 맥락을 완전히 이해하고 연속성 있는 조언 제공
- 사용자의 구체적인 질문에 직접적으로 답변
- 3-4개의 핵심 포인트로 간결하게 구성
- 실행 가능한 구체적인 방법론과 리소스 제공""",

            "rag_enhanced": """

【RAG 기반 조언 특별 지침】
- 제공된 전문 문서와 자료를 적극 활용
- 출처가 명확한 정보 기반으로 답변 구성
- 최신 노동시장 동향과 통계 데이터 반영
- 문서에서 추출한 핵심 인사이트를 사용자 상황에 적용""",

            "streaming": """

【스트리밍 응답 특별 지침】
- 즉시 도움이 되는 핵심 정보부터 제공
- 자연스러운 대화 흐름 유지
- 간결하면서도 가치 있는 조언
- 추가 질문을 유도하는 열린 결론""",

            "conversational": """

【일반 대화 특별 지침】
- 친근하고 접근하기 쉬운 톤 유지
- 사용자의 감정과 상황에 공감
- 균형 잡힌 깊이의 조언 제공
- 추가 도움이 필요한 영역 제안"""
        }
        return guidelines.get(template_name, "")
    
    def _format_context(self, context: Dict) -> str:
        """컨텍스트 포맷팅"""
        formatted = ""
        for key, value in context.items():
            if isinstance(value, dict):
                formatted += f"- {key}: {json.dumps(value, ensure_ascii=False)}\n"
            else:
                formatted += f"- {key}: {value}\n"
        return formatted

    def get_career_coach_system_prompt(self, user_context: Dict = None) -> str:
        """커리어 코치 시스템 프롬프트 생성 (기존 호환성)"""
        return self.get_unified_system_prompt("initial_advice", user_context)
    
    def get_follow_up_system_prompt(self) -> str:
        """후속 질문용 시스템 프롬프트 생성 (기존 호환성)"""
        return self.get_unified_system_prompt("follow_up")
    
    def get_rag_system_prompt(self) -> str:
        """RAG 기반 시스템 프롬프트 생성"""
        return self.get_unified_system_prompt("rag_enhanced")
    
    def get_streaming_system_prompt(self) -> str:
        """스트리밍 응답용 시스템 프롬프트 생성"""
        return self.get_unified_system_prompt("streaming")
    
    def get_conversational_system_prompt(self) -> str:
        """일반 대화용 시스템 프롬프트 생성"""
        return self.get_unified_system_prompt("conversational")
    
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
    

# 싱글톤 인스턴스
prompt_manager = PromptTemplateManager()