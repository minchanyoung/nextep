# app/utils/validators.py
"""
데이터 검증 유틸리티
"""

import re
from typing import Any, Dict, List, Optional, Union
from app.core.exceptions import ValidationError


class BaseValidator:
    """기본 검증자 클래스"""
    
    def validate(self, value: Any, field_name: str = "field") -> Any:
        """검증 실행"""
        raise NotImplementedError


class RequiredValidator(BaseValidator):
    """필수 값 검증"""
    
    def validate(self, value: Any, field_name: str = "field") -> Any:
        if value is None or value == "":
            raise ValidationError(f"{field_name}은(는) 필수 입력값입니다.")
        return value


class IntegerValidator(BaseValidator):
    """정수 검증"""
    
    def __init__(self, min_val: Optional[int] = None, max_val: Optional[int] = None):
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, value: Any, field_name: str = "field") -> int:
        if value is None:
            return None
        
        try:
            int_val = int(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name}은(는) 정수여야 합니다.")
        
        if self.min_val is not None and int_val < self.min_val:
            raise ValidationError(f"{field_name}은(는) {self.min_val} 이상이어야 합니다.")
        
        if self.max_val is not None and int_val > self.max_val:
            raise ValidationError(f"{field_name}은(는) {self.max_val} 이하여야 합니다.")
        
        return int_val


class StringValidator(BaseValidator):
    """문자열 검증"""
    
    def __init__(self, min_length: Optional[int] = None, max_length: Optional[int] = None, 
                 pattern: Optional[str] = None):
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = re.compile(pattern) if pattern else None
    
    def validate(self, value: Any, field_name: str = "field") -> str:
        if value is None:
            return None
        
        if not isinstance(value, str):
            raise ValidationError(f"{field_name}은(는) 문자열이어야 합니다.")
        
        if self.min_length is not None and len(value) < self.min_length:
            raise ValidationError(f"{field_name}은(는) 최소 {self.min_length}자 이상이어야 합니다.")
        
        if self.max_length is not None and len(value) > self.max_length:
            raise ValidationError(f"{field_name}은(는) 최대 {self.max_length}자 이하여야 합니다.")
        
        if self.pattern and not self.pattern.match(value):
            raise ValidationError(f"{field_name}의 형식이 올바르지 않습니다.")
        
        return value


class EmailValidator(BaseValidator):
    """이메일 검증"""
    
    EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    def validate(self, value: Any, field_name: str = "email") -> str:
        if value is None:
            return None
        
        if not isinstance(value, str):
            raise ValidationError(f"{field_name}은(는) 문자열이어야 합니다.")
        
        if not re.match(self.EMAIL_PATTERN, value):
            raise ValidationError(f"{field_name}의 형식이 올바르지 않습니다.")
        
        return value


class ChoiceValidator(BaseValidator):
    """선택지 검증"""
    
    def __init__(self, choices: List[Any]):
        self.choices = choices
    
    def validate(self, value: Any, field_name: str = "field") -> Any:
        if value is None:
            return None
        
        if value not in self.choices:
            raise ValidationError(f"{field_name}은(는) {self.choices} 중 하나여야 합니다.")
        
        return value


class UserProfileValidator:
    """사용자 프로필 검증"""
    
    VALIDATORS = {
        'username': [RequiredValidator(), StringValidator(min_length=3, max_length=50)],
        'email': [RequiredValidator(), EmailValidator()],
        'age': [IntegerValidator(min_val=15, max_val=100)],
        'gender': [ChoiceValidator([0, 1])],  # 0: 남성, 1: 여성
        'education': [IntegerValidator(min_val=1, max_val=10)],
        'monthly_income': [IntegerValidator(min_val=0, max_val=999999)],
        'job_category': [IntegerValidator(min_val=1, max_val=50)],
        'job_satisfaction': [IntegerValidator(min_val=1, max_val=5)],
        'satis_focus_key': [StringValidator(max_length=50)]
    }
    
    @classmethod
    def validate_profile_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """프로필 데이터 검증"""
        validated_data = {}
        errors = {}
        
        for field_name, value in data.items():
            if field_name in cls.VALIDATORS:
                try:
                    validated_value = value
                    for validator in cls.VALIDATORS[field_name]:
                        validated_value = validator.validate(validated_value, field_name)
                    validated_data[field_name] = validated_value
                except ValidationError as e:
                    errors[field_name] = str(e)
            else:
                validated_data[field_name] = value
        
        if errors:
            raise ValidationError(f"검증 오류: {errors}")
        
        return validated_data


class MLInputValidator:
    """머신러닝 입력 데이터 검증"""
    
    REQUIRED_FIELDS = [
        'age', 'gender', 'education', 'monthly_income', 
        'current_job_category', 'job_A_category', 'job_B_category'
    ]
    
    @classmethod
    def validate_prediction_input(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """예측 입력 데이터 검증"""
        # 필수 필드 확인
        for field in cls.REQUIRED_FIELDS:
            if field not in data or data[field] is None:
                raise ValidationError(f"필수 입력값 '{field}'이(가) 누락되었습니다.")
        
        # 데이터 타입 및 범위 검증
        validators = {
            'age': IntegerValidator(min_val=15, max_val=100),
            'gender': ChoiceValidator([0, 1]),
            'education': IntegerValidator(min_val=1, max_val=10),
            'monthly_income': IntegerValidator(min_val=0, max_val=999999),
            'current_job_category': IntegerValidator(min_val=1, max_val=50),
            'job_A_category': IntegerValidator(min_val=1, max_val=50),
            'job_B_category': IntegerValidator(min_val=1, max_val=50)
        }
        
        validated_data = {}
        for field, validator in validators.items():
            if field in data:
                validated_data[field] = validator.validate(data[field], field)
        
        # 추가 비즈니스 로직 검증
        if validated_data['current_job_category'] == validated_data['job_A_category']:
            raise ValidationError("현재 직업과 이직 옵션 A는 서로 달라야 합니다.")
        
        if validated_data['current_job_category'] == validated_data['job_B_category']:
            raise ValidationError("현재 직업과 이직 옵션 B는 서로 달라야 합니다.")
        
        return validated_data