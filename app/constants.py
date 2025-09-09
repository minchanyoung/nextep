# app/constants.py
"""
애플리케이션 상수 정의
"""

from collections import OrderedDict

# 직업 카테고리 매핑
JOB_CATEGORY_MAP = OrderedDict([
    ("1", "관리자"), 
    ("2", "전문가 및 관련 종사자"), 
    ("3", "사무 종사자"),
    ("4", "서비스 종사자"), 
    ("5", "판매 종사자"), 
    ("6", "농림 어업 숙련 종사자"),
    ("7", "기능원 및 관련 기능 종사자"), 
    ("8", "장치·기계조작 및 조립 종사자"),
    ("9", "단순 노무 종사자")
])

# 만족도 요인 매핑
SATIS_FACTOR_MAP = {
    "satis_wage": "임금 또는 보수", 
    "satis_stability": "고용 안정성",
    "satis_task_content": "일의 내용", 
    "satis_work_env": "근무 환경",
    "satis_work_time": "근무 시간과 여가", 
    "satis_growth": "발전 가능성",
    "satis_communication": "인간 관계", 
    "satis_fair_eval": "공정한 평가와 보상",
    "satis_welfare": "복리 후생"
}

# 교육 수준 매핑
EDUCATION_MAP = {
    "1": "무학 또는 초중퇴", 
    "2": "고졸 미만", 
    "3": "고졸",
    "4": "대재 또는 중퇴", 
    "5": "전문대 졸업", 
    "6": "대학교 졸업 이상"
}

# 성별 매핑
GENDER_MAP = {
    0: "남성",
    1: "여성"
}

# 프로필 필수 필드
REQUIRED_PROFILE_FIELDS = [
    'age', 'gender', 'education', 'monthly_income', 
    'job_category', 'job_satisfaction'
]

# 기본 예측 결과 (오류 시 사용)
DEFAULT_PREDICTION_RESULTS = [
    {"income_change_rate": 0, "satisfaction_change_score": 0, "distribution": None},
    {"income_change_rate": 0, "satisfaction_change_score": 0, "distribution": None},
    {"income_change_rate": 0, "satisfaction_change_score": 0, "distribution": None}
]

# HTTP 상태 코드
HTTP_STATUS = {
    'OK': 200,
    'BAD_REQUEST': 400,
    'UNAUTHORIZED': 401,
    'NOT_FOUND': 404,
    'INTERNAL_SERVER_ERROR': 500
}

# 메시지 템플릿
MESSAGES = {
    'LOGIN_REQUIRED': '로그인이 필요합니다.',
    'PROFILE_INCOMPLETE': '예측을 위해 프로필 정보를 모두 입력해주세요.',
    'PREDICTION_ERROR': '예측 중 오류가 발생했습니다.',
    'NO_PREDICTION_DATA': '예측 데이터가 없습니다. 다시 예측을 시도해주세요.',
    'PROFILE_UPDATE_SUCCESS': '프로필이 성공적으로 업데이트되었습니다.',
    'PROFILE_UPDATE_FAILED': '프로필 업데이트에 실패했습니다.',
    'LOGIN_FAILED': '아이디 또는 비밀번호가 올바르지 않습니다.',
    'SIGNUP_SUCCESS': '회원가입이 완료되었습니다. 로그인해주세요.',
}