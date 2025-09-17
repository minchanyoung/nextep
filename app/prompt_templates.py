from typing import Dict, List
import json

LABOR_MARKET_TRENDS = [
    {"id": "LMT001", "title": "2024년 노동시장 개관", "content": "2024년 노동시장은 전반적으로 고용지표가 둔화되었습니다.", "keywords": ["노동시장", "고용지표", "취업자 감소"]},
    {"id": "LMT002", "title": "청년 고용 부진", "content": "20대 고용률이 감소하며 신규 취업자 감소가 두드러졌습니다.", "keywords": ["청년 고용", "20대", "신규 취업자"]},
]

LEARNING_RECOMMENDATIONS = [
    {"id": "LR001", "skill_name": "ICT 역량", "description": "소프트웨어, 데이터 분석, AI/ML, 클라우드 컴퓨팅 등 ICT 전문 역량은 성장 잠재력이 큽니다.", "keywords": ["ICT", "데이터분석", "AI"]},
    {"id": "LR002", "skill_name": "보건복지 전문성", "description": "고령화 사회로 간호, 간병, 사회복지 전문가 수요가 증가하고 있습니다.", "keywords": ["보건의료", "사회복지", "간호"]},
]

def get_labor_market_trends():
    return LABOR_MARKET_TRENDS

def get_learning_recommendations():
    return LEARNING_RECOMMENDATIONS

class PromptTemplateManager:
    def __init__(self):
        self.persona = {"name": "NEXTEP 커리어 코치", "role": "한국 노동시장 전문가이자 커리어 컨설턴트"}
        self.templates = {
            "initial_advice": ["핵심결론", "근거", "실행계획", "위험요소", "추가제안"],
            "follow_up": ["질문이해", "핵심답변", "구체적방법", "추가자료"],
            "rag_enhanced": ["자료분석", "통합답변", "실행지침"],
            "conversational": ["이해확인", "조언제공", "후속제안"],
        }

    def get_system_prompt(self, template_name: str, context: Dict = None) -> str:
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        system_prompt = f"""당신은 전문 커리어 컨설턴트, {self.persona['name']}입니다. 사용자의 상황에 공감하며 현실적인 조언을 제공하세요.

[대화 원칙]
- 모든 답변은 머리글, 목록, 굵은 글씨 등 마크다운 서식을 전혀 사용하지 않고, 오직 줄 바꿈으로만 단락을 구분하는 서술형 문장으로만 작성하세요.
- 핵심적인 조언을 먼저 제시하고, 그에 대한 구체적인 근거와 실행 계획을 이어서 설명해주세요.
- 사용자가 바로 실행에 옮길 수 있도록, 현실적이고 구체적인 행동 지침을 포함하여 조언해주세요.
- 전문가의 신뢰감 있는 어조를 유지하되, 긍정적이고 지지하는 태도를 보여주세요.
"""
        if context:
            system_prompt += "\n\n추가적인 고려사항은 다음과 같습니다.\n"
            for k, v in context.items():
                system_prompt += f"{k}: {json.dumps(v, ensure_ascii=False)}\n"
        return system_prompt

    def get_career_advice_prompt(
        self,
        user_input: Dict,
        prediction_results: List[Dict],
        job_category_map: Dict,
        satis_factor_map: Dict,
        labor_market_context: str = "",
        learning_context: str = "",
        ui_recommendation: str = None
    ) -> List[Dict]:
        current_job = job_category_map.get(user_input.get('current_job_category', ''), '알 수 없음')
        job_a = job_category_map.get(user_input.get('job_A_category', ''), '알 수 없음')
        job_b = job_category_map.get(user_input.get('job_B_category', ''), '알 수 없음')
        gender_text = '여성' if str(user_input.get('gender')) == '1' else '남성'
        
        # 사용자가 선택한 직업군에 맞는 예측 결과 사용
        job_A_code = user_input.get('job_A_category')
        job_B_code = user_input.get('job_B_category')

        # 디버깅 로그
        print(f"DEBUG prompt - job_A_code: {job_A_code}, job_B_code: {job_B_code}")
        print(f"DEBUG prompt - prediction_results keys: {list(prediction_results.keys()) if isinstance(prediction_results, dict) else 'Not dict'}")

        p0 = prediction_results.get('current', {})
        p1 = prediction_results.get(job_A_code, {})
        p2 = prediction_results.get(job_B_code, {})

        prediction_text = f"""
[예측 결과]
- 현직 유지({current_job}): 소득 {p0.get('income_change_rate', 0):.2%}, 만족도 {p0.get('satisfaction_change_score', 0):.2f}
- 옵션 A({job_a}): 소득 {p1.get('income_change_rate', 0):.2%}, 만족도 {p1.get('satisfaction_change_score', 0):.2f}
- 옵션 B({job_b}): 소득 {p2.get('income_change_rate', 0):.2%}, 만족도 {p2.get('satisfaction_change_score', 0):.2f}"""
        user_profile = f"""
[사용자 프로필]
- 나이: {user_input.get('age', 'N/A')}세
- 성별: {gender_text}
- 현재 직업: {current_job}
- 월소득: {user_input.get('monthly_income', 'N/A')}만원"""
        rag_context = ""
        if labor_market_context:
            rag_context += f"\n[노동시장] {labor_market_context}"
        if learning_context:
            rag_context += f"\n[학습추천] {learning_context}"
        if ui_recommendation:
            rag_context += f"\n[UI 추천] {ui_recommendation}"
        system_prompt = self.get_system_prompt("initial_advice")
        user_prompt = f"""{user_profile}
{prediction_text}
{rag_context}

[요청]
위 사용자 프로필과 AI 예측 결과를 바탕으로, 세 가지 커리어 경로(현직 유지, 옵션 A({job_a}), 옵션 B({job_b}))에 대해 각각의 긍정적인 점과 부정적인 점, 그리고 현실적인 위험 요소를 자연스러운 문장으로 서술하여 비교 분석해주세요. 분석을 마친 후, 노동 시장 트렌드와 사용자의 상황을 종합적으로 고려하여 가장 추천하는 커리어 경로를 하나 선택하고 그 이유를 설명해주세요. 마지막으로, 해당 경로를 성공적으로 걷기 위한 구체적인 첫 단계부터 시작하여, 필요한 역량을 쌓기 위한 학습 계획까지 상세하게 이야기해주세요.

참고사항:
- 옵션 A는 "{job_a}"이고, 옵션 B는 "{job_b}"입니다. 반드시 정확한 직업명을 사용하여 조언하세요.
- 사용자가 예측 결과 페이지에서 설정한 우선순위와 AI 추천 결과를 고려하여 일관성 있는 조언을 제공해주세요."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

prompt_manager = PromptTemplateManager()
