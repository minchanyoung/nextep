from app import db

class User(db.Model):
    __tablename__ = 'members'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.Integer, nullable=True) # 0: 남성, 1: 여성
    education = db.Column(db.Integer, nullable=True)
    monthly_income = db.Column(db.Integer, nullable=True)
    job_category = db.Column(db.Integer, nullable=True)
    job_satisfaction = db.Column(db.Integer, nullable=True)
    
    # 각 만족도 요인별 점수 (1-5점)
    satis_wage = db.Column(db.Integer, nullable=True, default=3)  # 임금 또는 보수
    satis_stability = db.Column(db.Integer, nullable=True, default=3)  # 고용 안정성
    satis_growth = db.Column(db.Integer, nullable=True, default=3)  # 발전 가능성
    satis_task_content = db.Column(db.Integer, nullable=True, default=3)  # 일의 내용
    satis_work_env = db.Column(db.Integer, nullable=True, default=3)  # 근무 환경
    satis_work_time = db.Column(db.Integer, nullable=True, default=3)  # 근무 시간과 여가
    satis_communication = db.Column(db.Integer, nullable=True, default=3)  # 인간 관계
    satis_fair_eval = db.Column(db.Integer, nullable=True, default=3)  # 공정한 평가와 보상
    satis_welfare = db.Column(db.Integer, nullable=True, default=3)  # 복리 후생
    
    # 추가 분석용 컬럼
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

    def __repr__(self):
        return f'<User {self.username}>'
    
    def get_satisfaction_vector(self):
        """만족도 요인들을 벡터로 반환"""
        return [
            self.satis_wage or 3,
            self.satis_stability or 3,
            self.satis_growth or 3,
            self.satis_task_content or 3,
            self.satis_work_env or 3,
            self.satis_work_time or 3,
            self.satis_communication or 3,
            self.satis_fair_eval or 3,
            self.satis_welfare or 3
        ]
    
    def calculate_satisfaction_stats(self):
        """만족도 통계 계산"""
        import numpy as np
        satis_vector = self.get_satisfaction_vector()
        return {
            'mean': np.mean(satis_vector),
            'std': np.std(satis_vector),
            'min': np.min(satis_vector),
            'max': np.max(satis_vector)
        }