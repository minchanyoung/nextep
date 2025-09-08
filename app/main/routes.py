import json
import time
import traceback
from flask import render_template, request, redirect, url_for, flash, session, current_app, jsonify, Response
from app import services
from app.constants import (
    JOB_CATEGORY_MAP, SATIS_FACTOR_MAP, EDUCATION_MAP,
    REQUIRED_PROFILE_FIELDS, DEFAULT_PREDICTION_RESULTS, MESSAGES
)
from app.utils.web_helpers import (
    get_prediction_data, set_prediction_data, login_required,
    get_current_user, handle_api_exception, success_response, error_response
)
from . import bp

def _get_alternative_jobs(current_job_code: str) -> tuple[str, str]:
    job_map = {
        '1': ('2', '3'),
        '2': ('3', '1'),
        '3': ('2', '4'),
    }
    return job_map.get(current_job_code, ('2', '3'))

def _create_user_input(source) -> dict:
    is_form = hasattr(source, 'get')
    def get_val(key, default=None):
        return source.get(key, default) if is_form else getattr(source, key, default)
    current_job = str(get_val('job_category') or get_val('current_job_category', '3'))
    job_A = str(get_val('job_A_category', '2'))
    job_B = str(get_val('job_B_category', '4'))
    user_input = {
        'age': str(get_val('age')),
        'gender': str(get_val('gender')),
        'education': str(get_val('education')),
        'monthly_income': str(get_val('monthly_income')),
        'current_job_category': current_job,
        'job_satisfaction': str(get_val('job_satisfaction')),
        'job_A_category': job_A,
        'job_B_category': job_B,
    }
    for key in SATIS_FACTOR_MAP.keys():
        user_input[key] = str(get_val(key, 3))
    return user_input

@bp.route('/')
def index():
    return render_template('main/index.html')

@bp.route('/about')
def about():
    return render_template('main/about.html')

@bp.route('/faq')
def faq():
    return render_template('main/faq.html')

@bp.route('/predict', methods=['GET'])
def predict():
    user = get_current_user()
    if user and getattr(user, 'age', None):
        try:
            user_input = _create_user_input(user)
            prediction_results = services.run_prediction(user_input)
            set_prediction_data(user_input, prediction_results)
            return redirect(url_for('main.predict_result'))
        except Exception as e:
            current_app.logger.error(f"자동 예측 중 오류: {e}")
            flash("프로필 기반 자동 예측 중 오류가 발생했습니다. 수동으로 입력해주세요.")
    return render_template(
        'main/predict.html',
        job_category_map=JOB_CATEGORY_MAP,
        satis_factor_map=SATIS_FACTOR_MAP,
        education=EDUCATION_MAP
    )

@bp.route('/predict-result', methods=['GET', 'POST'])
def predict_result():
    user = get_current_user()
    is_guest = not user
    error_message = None
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    if request.method == 'POST':
        user_input = _create_user_input(request.form)
        try:
            prediction_results = services.run_prediction(user_input)
            set_prediction_data(user_input, prediction_results)
        except Exception as e:
            current_app.logger.error(f"수동 예측 오류: {e}", exc_info=True)
            if is_ajax:
                return jsonify({'status': 'error', 'message': '예측 중 오류가 발생했습니다.'}), 500
            flash("예측 중 오류가 발생했습니다. 다시 시도해주세요.")
            return redirect(url_for('main.predict'))
    else:
        if is_guest:
            return redirect(url_for('main.predict'))
        prediction_data = get_prediction_data()
        if not prediction_data:
            flash("세션에 예측 데이터가 없습니다. 다시 예측해주세요.")
            return redirect(url_for('main.predict'))
        user_input = prediction_data['user_input']
        prediction_results = prediction_data['prediction_results']

    if is_ajax:
        return jsonify({
            'status': 'success',
            'prediction_results': prediction_results,
            'user_input': user_input,
            'job_category_map': JOB_CATEGORY_MAP
        })

    return render_template(
        'main/predict_result.html',
        user_input=user_input,
        prediction_results=prediction_results,
        job_category_map=JOB_CATEGORY_MAP,
        education=EDUCATION_MAP,
        satis_factor_map=SATIS_FACTOR_MAP,
        error_message=error_message,
        is_guest=is_guest
    )

@bp.route('/advice', methods=['GET', 'POST'])
def advice():
    user = get_current_user()
    if request.method == 'POST' and not user:
        user_input = _create_user_input(request.form)
        try:
            prediction_results = services.run_prediction(user_input)
        except Exception as e:
            current_app.logger.error(f"비회원 예측 중 오류: {e}")
            flash("예측 중 오류가 발생했습니다. 다시 시도해주세요.")
            return redirect(url_for('main.predict'))
    elif user:
        prediction_data = get_prediction_data()
        if not prediction_data:
            flash("AI 조언을 위한 예측 데이터가 없습니다. 먼저 예측을 실행해주세요.")
            return redirect(url_for('main.predict'))
        user_input = prediction_data['user_input']
        prediction_results = prediction_data['prediction_results']
    else:
        flash("AI 조언을 받으려면 로그인하거나 예측을 먼저 실행해주세요.")
        return redirect(url_for('main.predict'))

    try:
        from app.chat_session import get_current_chat_session, clear_chat_session
        clear_chat_session()
        chat_session = get_current_chat_session()
        ai_advice = services.generate_career_advice(user_input, prediction_results)
        context_summary = services.summarize_context(user_input, prediction_results)
        chat_session.set_user_context(user_input, prediction_results, JOB_CATEGORY_MAP, SATIS_FACTOR_MAP)
        chat_session.add_message("assistant", context_summary, {"type": "context_summary"})
        chat_session.add_message("assistant", ai_advice, {"type": "initial_advice"})
    except Exception as e:
        current_app.logger.error(f"AI 조언 생성 오류: {e}", exc_info=True)
        ai_advice = "AI 조언 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

    return render_template('main/advice.html', ai_advice=ai_advice)

@bp.route('/ask-ai-stream', methods=['POST'])
def ask_ai_stream():
    data = request.get_json(silent=True) or {}
    user_message = (data.get('message') or '').strip()
    if not user_message:
        return Response(json.dumps({'error': '메시지가 없습니다.'}), status=400, mimetype='application/json')

    def generate_stream():
        try:
            from app.chat_session import get_current_chat_session
            chat_session = get_current_chat_session()
            chat_session.add_message("user", user_message)
            full_response = ""
            for chunk in services.generate_follow_up_advice_stream(chat_session):
                if chunk and chunk.strip():
                    full_response += chunk
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                    time.sleep(0.005)
            chat_session.add_message("assistant", full_response, {"streaming": True})
            yield f"data: {json.dumps({'done': True, 'full_response': full_response})}\n\n"
        except Exception as e:
            current_app.logger.error(f"스트리밍 AI 응답 중 오류: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': 'AI 응답 생성 중 오류가 발생했습니다.'})}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    }
    return Response(generate_stream(), mimetype='text/event-stream', headers=headers)

@bp.route('/ask-ai', methods=['POST'])
def ask_ai():
    data = request.get_json(silent=True) or {}
    user_message = (data.get('message') or '').strip()
    if not user_message:
        return jsonify({'error': '메시지가 없습니다.'}), 400

    try:
        from app.chat_session import get_current_chat_session
        chat_session = get_current_chat_session()
        chat_session.add_message("user", user_message)
        
        response = services.generate_follow_up_advice(
            user_message=user_message,
            chat_history=chat_session.get_messages(),
            context_summary=chat_session.get_context_summary()
        )
        
        chat_session.add_message("assistant", response)
        return jsonify({'reply': response})
    except Exception as e:
        current_app.logger.error(f"AI 응답 중 오류: {e}", exc_info=True)
        return jsonify({'error': 'AI 응답 생성 중 오류가 발생했습니다.'}), 500

@bp.route('/db-test')
def db_test():
    try:
        status = services.example_db_query()
        return f"<h1>DB 테스트 결과: {status}</h1>"
    except Exception as e:
        return f"<h1>DB 테스트 실패: {e}</h1>", 500

@bp.route('/ml-status')
def ml_status():
    ml_resources = current_app.extensions.get('ml_resources', {})
    status_html = "<h1>ML 모델 상태 확인</h1>"
    for key in ['lgb_income', 'cat_satis', 'klips_df', 'job_category_stats', 'income_features', 'satis_features']:
        status_html += f"<p>{'✅' if key in ml_resources and ml_resources[key] is not None else '❌'} {key}</p>"
    return status_html

@bp.route('/profile', methods=['GET', 'POST'], endpoint='profile')
@login_required
def profile():
    user = get_current_user()
    if request.method == 'POST':
        success = services.update_user_profile(user.id, request.form.to_dict())
        if success:
            flash('프로필이 성공적으로 업데이트되었습니다.')
        else:
            flash('프로필 업데이트에 실패했습니다.')
        return redirect(url_for('main.profile'))

    return render_template('main/profile.html', user=user,
                           job_category_map=JOB_CATEGORY_MAP,
                           satis_factor_map=SATIS_FACTOR_MAP,
                           education=EDUCATION_MAP)

@bp.route('/chat/new', methods=['POST'])
def chat_new_redirect():
    return redirect(url_for('main.advice'))


