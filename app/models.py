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
    satis_focus_key = db.Column(db.String(50), nullable=True)

    def __repr__(self):
        return f'<User {self.username}>'