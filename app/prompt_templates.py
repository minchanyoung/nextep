# app/prompt_templates.py
"""
통합 프롬프트 및 RAG 데이터 관리 시스템
프롬프트 템플릿과 RAG 데이터를 한곳에서 통합 관리
"""

from typing import Dict, List, Optional
import json

# === RAG 데이터 통합 ===
# === RAG 데이터 완전 통합 (rag_data.py 내용 통합) ===
# 노동 시장 및 산업 트렌드 데이터 (2024년 노동시장 평가와 2025년 전망 기반)
LABOR_MARKET_TRENDS = [
    {
        "id": "LMT001",
        "title": "2024년 노동시장 개관 - 고용지표 전반적 둔화",
        "content": "2024년 노동시장은 전반적으로 고용지표가 둔화되는 모습을 보였습니다. 취업자 증가폭은 2023년 33.6만 명에서 2024년 18.4만 명으로 크게 줄었으며, 고용률 증가폭이 둔화되고 실업률은 소폭 증가했습니다. 비경제활동인구 감소폭도 줄어들고 있어 2023년까지 보였던 긍정적 변화들이 전반적으로 약화되고 있는 상황입니다.",
        "keywords": ["2024년 노동시장", "고용지표 둔화", "취업자 감소", "고용률", "실업률", "비경제활동인구"]
    },
    {
        "id": "LMT002", 
        "title": "청년층 고용 부진 심화 - 20대 고용 위기",
        "content": "20대 취업자는 11.3만 명 감소했으며, 대부분 인구효과에 의한 것이지만 고용률이 5월 이후 감소하는 등 청년층 고용 상황이 전년대비 악화되었습니다. 특히 신규 취업자(근속 1년 미만) 감소가 두드러져 10만 명 가량 줄었고, 신규 학졸자 고용률도 1.5%p 감소했습니다. 노동시장에 새로이 진입하는 청년들이 어려움을 겪고 있는 상황입니다.",
        "keywords": ["청년 고용", "20대", "신규 취업자", "학졸자", "고용률 감소", "노동시장 진입"]
    },
    {
        "id": "LMT003",
        "title": "제조업 고용 둔화 지속 - 반도체와 제조업의 괴리",
        "content": "제조업 취업자는 2024년 1∼10월 평균 1.2만 명 증가했으나, 계절조정 기준으로는 4월을 정점으로 감소세를 보였습니다. 제조업 경기가 회복세를 나타났음에도 고용은 지난해부터의 부진한 흐름이 이어졌는데, 이는 반도체 중심의 경기 개선과 반도체를 제외한 제조업의 상대적 부진 때문입니다. 반도체 제조업의 고용유발계수가 낮아 전체 제조업 고용 증가 효과는 미미했습니다.",
        "keywords": ["제조업", "반도체", "고용 둔화", "계절조정", "고용유발계수", "경기 회복"]
    },
    {
        "id": "LMT004",
        "title": "건설업 고용 부진 심화 - 건설경기 악화 영향",
        "content": "건설업 취업자는 2024년 1∼10월 평균 3.3만 명 감소했으며, 2분기부터 감소하기 시작해 3분기에 감소폭이 더욱 확대되었습니다. 건설투자가 2분기에 감소로 전환되고 3분기에는 감소폭이 확대되면서 건설업 고용 감소로 이어졌습니다. 상용직과 일용직 모두에서 감소했으며, 특히 일용직 감소가 두드러졌습니다.",
        "keywords": ["건설업", "건설투자", "건설경기", "일용직", "상용직", "고용 감소"]
    },
    {
        "id": "LMT005",
        "title": "서비스업 취업자 증가폭 둔화 - 업종별 명암",
        "content": "서비스업 취업자는 2024년 1∼10월 평균 23.9만 명 증가했으나 전년보다 약 18만 명 줄어들었습니다. 도소매업이 가장 큰 감소(-5.5만 명)를 보였고, 교육서비스업(-1.2만 명)도 감소했습니다. 반면 정보통신업(7.4만 명), 전문과학 및 기술서비스업(6.2만 명), 보건업 및 사회복지서비스업(8.8만 명) 등은 양호한 증가세를 유지했습니다.",
        "keywords": ["서비스업", "도소매업", "정보통신업", "전문과학기술", "보건복지", "교육서비스"]
    },
    {
        "id": "LMT006",
        "title": "여성 고용 증가 vs 남성 고용 감소 - 성별 고용 격차",
        "content": "2024년 고용은 성별에 따라 큰 차이를 보였습니다. 지난해와 마찬가지로 남성 고용률은 감소한 반면, 여성 고용률은 증가했습니다. 특히 30대 여성 고용률이 크게 늘어났는데, 지난 4년간 30대 초반에서 약 9.3%p, 30대 후반에서 약 13.9%p 증가했습니다. 남성 취업자가 다수인 제조업과 건설업의 고용 부진과 여성 고용 비중이 큰 일부 서비스업 고용 증가의 영향입니다.",
        "keywords": ["여성 고용", "남성 고용", "성별 격차", "30대 여성", "고용률 증가", "서비스업"]
    },
    {
        "id": "LMT007",
        "title": "고령층 고용 증가세 지속 - 노인일자리사업 영향",
        "content": "60세 이상 고령층 고용은 인구구조 변화와 노인일자리사업 영향으로 지속적으로 증가했습니다. 노인일자리사업 참여자로 간주할 수 있는 취업자(65세 이상·임시직·공공행정 및 보건사회복지서비스업·단순노무직)가 7.7만 명 증가했습니다. 고령층에서도 여성 고용이 호조를 보였으며, 안정적인 고용률 증가세와 실업률 증가가 동시에 나타나 고령층의 경제활동 진출이 여전히 활발함을 보여줍니다.",
        "keywords": ["고령층 고용", "60세 이상", "노인일자리사업", "여성 고령층", "경제활동", "인구구조"]
    },
    {
        "id": "LMT008",
        "title": "상용직 증가폭 둔화 vs 임시직 증가 - 고용의 질 변화",
        "content": "2024년 상용직 증가폭은 전년 49.2만 명에서 18.4만 명으로 크게 줄었습니다. 보건업 및 사회복지서비스업, 제조업, 건설업에서 상용직 증가폭이 둔화되거나 감소했습니다. 반면 임시직은 다양한 산업에서 증가했으며, 특히 노인일자리사업과 간호·간병 인력이 포함된 보건업 및 사회복지서비스업에서 가장 많이 늘었습니다.",
        "keywords": ["상용직", "임시직", "고용의 질", "보건복지", "노인일자리", "간호간병"]
    },
    {
        "id": "LMT009",
        "title": "구직기간 장기화 - 중장기 실업자 증가",
        "content": "2024년 실업자는 구직기간이 3개월 이상인 중·장기적 실업자 위주로 늘었습니다. 실업자의 평균 구직기간이 하반기로 갈수록 증가폭이 확대되었으며, 단기간에 취업으로 이행하지 못한 실업자가 늘어났습니다. 전직 실업자는 주로 비자발적 사유로 직장을 그만두었으며, 특히 건설업(1.6만 명 증가)과 연관 업종인 부동산업에서 증가했습니다.",
        "keywords": ["실업자", "구직기간", "중장기 실업", "비자발적 실직", "건설업", "부동산업"]
    },
    {
        "id": "LMT010",
        "title": "2025년 노동시장 전망 - 완만한 둔화 예상",
        "content": "2025년 노동시장은 2024년보다 완만한 둔화가 전망됩니다. 취업자는 전년대비 약 12만 명 증가할 것으로 예상되며, 이는 2024년 증가폭(18.2만 명)보다 6.2만 명 감소한 수치입니다. 생산가능인구 감소폭 확대(-38만 명), 정부 직접일자리사업 증가세 둔화, 내수 부문의 큰 반등 없는 서비스업 고용 확대 제한, 제조업과 건설업 부진 지속 등이 주요 원인입니다.",
        "keywords": ["2025년 전망", "노동시장 둔화", "취업자 증가", "생산가능인구", "정부일자리", "내수 부진"]
    }
]

# 학습 및 역량 강화 추천 데이터 (2024년 노동시장 동향 기반)
LEARNING_RECOMMENDATIONS = [
    {
        "id": "LR001",
        "skill_name": "정보통신기술 (ICT) 전문 역량",
        "description": "정보통신업은 2024년 7.4만 명이 증가한 고성장 분야입니다. 소프트웨어 개발, 데이터 분석, AI/ML, 클라우드 컴퓨팅 등 ICT 전문 역량은 안정적인 고용과 높은 성장 잠재력을 제공합니다.",
        "category": "IT/디지털 전환",
        "related_job_categories": ["2", "7", "8"],
        "learning_resources": [
            {"type": "자격증", "name": "정보처리기사", "link": "https://www.q-net.or.kr/crf005.do?id=crf00505&jmCd=1320"},
            {"type": "온라인 강의", "name": "K-Digital Training - AI/빅데이터", "link": "https://www.hrd.go.kr/hrdp/ma/pmmao/indexNew.do"}
        ],
        "keywords": ["ICT", "정보통신", "소프트웨어", "데이터분석", "AI", "클라우드", "고성장"]
    },
    {
        "id": "LR002",
        "skill_name": "보건의료 및 사회복지 전문성",
        "description": "보건업 및 사회복지서비스업은 2024년 8.8만 명이 증가한 안정적 성장 분야입니다. 고령화 사회 진입으로 간호, 간병, 사회복지 전문가 수요가 지속 증가하고 있습니다.",
        "category": "보건복지",
        "related_job_categories": ["2", "4"],
        "learning_resources": [
            {"type": "자격증", "name": "사회복지사", "link": "https://www.q-net.or.kr/crf005.do?id=crf00505&jmCd=7930"},
            {"type": "자격증", "name": "간병사 자격증", "link": "https://www.hrd.go.kr"}
        ],
        "keywords": ["보건의료", "사회복지", "간호", "간병", "고령화", "안정성장", "전문가"]
    },
    {
        "id": "LR003",
        "skill_name": "전문과학기술 서비스 역량",
        "description": "전문과학 및 기술서비스업은 2024년 6.2만 명이 증가했습니다. 연구개발, 기술컨설팅, 엔지니어링 등 고부가가치 전문 서비스 분야의 성장이 지속되고 있습니다.",
        "category": "전문서비스",
        "related_job_categories": ["2"],
        "learning_resources": [
            {"type": "자격증", "name": "기술사", "link": "https://www.q-net.or.kr/crf005.do?id=crf00505&jmCd=1590"},
            {"type": "자격증", "name": "기사/산업기사", "link": "https://www.q-net.or.kr"}
        ],
        "keywords": ["전문과학기술", "연구개발", "기술컨설팅", "엔지니어링", "고부가가치", "전문서비스"]
    },
    {
        "id": "LR004",
        "skill_name": "재취업 및 직업전환 지원",
        "description": "건설업(-3.3만 명)과 도소매업(-5.5만 명)의 고용 감소로 인한 이직자를 위한 재교육이 필요합니다. 성장 분야로의 직업 전환을 위한 디지털 역량 강화와 서비스업 진출 준비가 중요합니다.",
        "category": "직업전환",
        "related_job_categories": ["3", "4", "5", "6", "7"],
        "learning_resources": [
            {"type": "정부지원", "name": "국민내일배움카드", "link": "https://www.hrd.go.kr/hrdp/ma/pmmao/indexNew.do"},
            {"type": "직업훈련", "name": "폴리텍대학 재직자과정", "link": "https://www.kopo.ac.kr"}
        ],
        "keywords": ["재취업", "직업전환", "재교육", "디지털역량", "서비스업", "건설업", "도소매업"]
    },
    {
        "id": "LR005",
        "skill_name": "청년층 취업 역량 강화",
        "description": "20대 취업자가 11.3만 명 감소하고 신규 학졸자 고용률이 1.5%p 감소한 상황입니다. 청년층의 노동시장 진입을 위한 실무 역량과 직업 경험 축적이 필요합니다.",
        "category": "청년취업",
        "related_job_categories": ["2", "3", "4", "5"],
        "learning_resources": [
            {"type": "정부지원", "name": "청년취업사관학교", "link": "https://www.work.go.kr/youthjob"},
            {"type": "인턴십", "name": "청년인턴십 프로그램", "link": "https://www.work.go.kr"}
        ],
        "keywords": ["청년취업", "20대", "신규학졸자", "인턴십", "실무역량", "직업경험", "노동시장진입"]
    },
    {
        "id": "LR006",
        "skill_name": "여성 경력개발 및 경력복귀",
        "description": "30대 여성 고용률이 지난 4년간 크게 증가(30대 초반 9.3%p, 30대 후반 13.9%p)했습니다. 여성의 경력 단절 예방과 경력 복귀를 위한 유연근무제 활용과 전문성 강화가 중요합니다.",
        "category": "여성경력개발",
        "related_job_categories": ["2", "3", "4"],
        "learning_resources": [
            {"type": "정부지원", "name": "새일센터 경력복귀 프로그램", "link": "https://saeil.mogef.go.kr"},
            {"type": "온라인 교육", "name": "여성인재개발원", "link": "https://www.kwdi.re.kr"}
        ],
        "keywords": ["여성경력개발", "30대여성", "경력복귀", "유연근무", "전문성강화", "경력단절예방"]
    },
    {
        "id": "LR007",
        "skill_name": "고령층 활동적 노후 준비",
        "description": "60세 이상 고령층 고용이 지속 증가하고 있으며, 노인일자리사업 참여자가 7.7만 명 증가했습니다. 고령층의 경험과 전문성을 활용한 사회공헌 및 소득창출 활동 준비가 필요합니다.",
        "category": "고령층 활용",
        "related_job_categories": ["4", "9"],
        "learning_resources": [
            {"type": "정부지원", "name": "노인일자리 및 사회활동 지원사업", "link": "https://www.seniorro.or.kr"},
            {"type": "교육과정", "name": "시니어 창업교육", "link": "https://www.sba.seoul.kr"}
        ],
        "keywords": ["고령층고용", "60세이상", "노인일자리", "사회공헌", "소득창출", "활동적노후", "경험활용"]
    },
    {
        "id": "LR008",
        "skill_name": "반도체 및 첨단제조업 전문기술",
        "description": "반도체 중심의 제조업 경기 개선이 나타나고 있습니다. 반도체, 전자부품, 정밀기계 등 첨단제조업 분야의 전문 기술 역량이 미래 제조업의 핵심이 될 것입니다.",
        "category": "첨단제조",
        "related_job_categories": ["2", "7", "8"],
        "learning_resources": [
            {"type": "자격증", "name": "반도체설계기사", "link": "https://www.q-net.or.kr"},
            {"type": "직업훈련", "name": "첨단제조 기술교육", "link": "https://www.kopo.ac.kr"}
        ],
        "keywords": ["반도체", "첨단제조", "전자부품", "정밀기계", "전문기술", "제조업경기", "기술혁신"]
    }
]

# 통합된 RAG 데이터 접근 함수들 (기존 호환성 유지)
def get_labor_market_trends():
    """노동시장 트렌드 데이터 반환 (기존 API 호환)"""
    return LABOR_MARKET_TRENDS

def get_learning_recommendations():
    """학습 추천 데이터 반환 (기존 API 호환)"""
    return LEARNING_RECOMMENDATIONS

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