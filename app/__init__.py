from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_compress import Compress
from app.llm_service import LLMService
from app.rag_manager import RAGManager
import os

db = SQLAlchemy()
migrate = Migrate()
compress = Compress()
# LangChain 기반 통합 서비스들
llm_service = LLMService()
rag_manager = RAGManager()

def create_app(config_class):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)
    compress.init_app(app)
    
    # LangChain 기반 통합 서비스들 초기화
    try:
        llm_service.init_app(app)
        rag_manager.init_app(app)
        app.logger.info("LangChain 기반 서비스들이 성공적으로 초기화되었습니다.")
    except Exception as e:
        app.logger.error(f"LangChain 서비스 초기화 실패: {e}")
        raise e

    # 로깅 설정
    import logging
    from logging.handlers import RotatingFileHandler

    if not app.debug and not app.testing:
        if app.config['LOG_TO_STDOUT']:
            handler = logging.StreamHandler()
        else:
            if not os.path.exists('logs'):
                os.mkdir('logs')
            file_handler = RotatingFileHandler('logs/' + app.config['LOG_FILE'], maxBytes=10240, backupCount=10)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)

        app.logger.setLevel(getattr(logging, app.config['LOG_LEVEL']))
        app.logger.info('Nextep startup')

    # Register blueprints here
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)

    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    from app.ml import bp as ml_bp
    app.register_blueprint(ml_bp, url_prefix='/api/ml')

    # ML 모델 사전 로딩
    from app.ml import routes as ml_routes
    ml_routes.init_app(app)
    
    # RAG 시스템 초기화 (지연 로딩)
    app.logger.info("RAG 시스템이 초기화되었습니다. 첫 번째 요청 시 데이터를 로드합니다.")

    @app.context_processor
    def utility_processor():
        def format_income_change(value):
            value = float(value)
            icon = "▲" if value > 0.001 else ("▼" if value < -0.001 else "―")
            return f"{icon} {value * 100:.2f}%"

        def format_satisfaction_change(value):
            value = float(value)
            icon = "▲" if value > 0.001 else ("▼" if value < -0.001 else "―")
            return f"{icon} {value:.2f}점"

        def get_change_class(value):
            value = float(value)
            if value > 0.001:
                return "positive-change"
            if value < -0.001:
                return "negative-change"
            return "no-change"
            
        return dict(
            format_income_change=format_income_change,
            format_satisfaction_change=format_satisfaction_change,
            get_change_class=get_change_class
        )

    @app.route('/test/')
    def test_page():
        return '<h1>Testing the Flask Application Factory Pattern</h1>'

    return app
