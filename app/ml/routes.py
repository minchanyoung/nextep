from flask import current_app, Blueprint, request, jsonify
import logging

# Blueprint 생성
bp = Blueprint('ml', __name__)
logger = logging.getLogger(__name__)

def init_app(app):
    """
    Flask 앱 초기화 과정에서 MLPredictor 인스턴스를 생성하고 등록합니다.
    이 함수는 더 이상 routes.py에 직접적인 ML 로직을 포함하지 않습니다.
    실제 초기화는 create_app 팩토리에서 처리됩니다.
    """
    pass

@bp.route('/predict', methods=['POST'])
def run_prediction_route():
    """
    웹 요청을 받아 ML 예측을 수행하는 라우트 함수.
    실제 로직은 MLPredictor 클래스에 위임합니다.
    """
    user_input = request.json
    if not user_input:
        return jsonify({"error": "Invalid input"}), 400

    scenarios_to_run = user_input.pop('scenarios_to_run', None)

    try:
        # 앱 컨텍스트에서 predictor 인스턴스를 가져옵니다.
        predictor = current_app.extensions.get('ml_predictor')
        if not predictor:
            logger.error("MLPredictor가 초기화되지 않았습니다.")
            return jsonify({"error": "ML service not available"}), 503

        results = predictor.predict(user_input, scenarios_to_run)
        return jsonify(results)

    except Exception as e:
        logger.error(f"API 예측 실패: {e}", exc_info=True)
        return jsonify({"error": "Prediction failed due to an internal error"}), 500

