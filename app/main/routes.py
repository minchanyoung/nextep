import json
import sys
import time
import traceback
from flask import render_template, request, redirect, url_for, flash, session, current_app, jsonify, Response
from app import services, db
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
    if user:
        # 로그인한 경우, 프로필 정보 완전성 확인 (값이 0인 경우도 유효하도록 수정)
        missing_fields = [field for field in REQUIRED_PROFILE_FIELDS if getattr(user, field, None) is None]
        if not missing_fields:
            # 프로필이 완전하면, 프로필 정보로 바로 예측 실행
            try:
                user_input = _create_user_input(user)
                all_job_codes = list(JOB_CATEGORY_MAP.keys())
                all_prediction_results = services.run_prediction(user_input, scenarios_to_run=all_job_codes)
                set_prediction_data(user_input, all_prediction_results)
                return redirect(url_for('main.predict_result'))
            except Exception as e:
                current_app.logger.error(f"프로필 기반 자동 예측 중 오류: {e}", exc_info=True)
                flash("자동 예측 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
                # 오류 발생 시 수동 입력 페이지로 이동
                return render_template(
                    'main/predict.html',
                    job_category_map=JOB_CATEGORY_MAP,
                    satis_factor_map=SATIS_FACTOR_MAP,
                    education=EDUCATION_MAP
                )
        else:
            # 프로필이 불완전하면, 프로필 페이지로 리디렉션
            flash(f'정확한 예측을 위해 프로필을 먼저 완성해주세요. (부족한 정보: {", ".join(missing_fields)})')
            return redirect(url_for('main.profile'))
    
    # 비로그인 사용자는 수동 입력 페이지 표시
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

    if request.method == 'POST':
        user_input = _create_user_input(request.form)
        try:
            # 모든 직업군 코드를 가져와서 예측 실행
            all_job_codes = list(JOB_CATEGORY_MAP.keys())
            all_prediction_results = services.run_prediction(user_input, scenarios_to_run=all_job_codes)
            
            # 세션에 모든 결과 저장
            set_prediction_data(user_input, all_prediction_results)
            return redirect(url_for('main.predict_result'))

        except Exception as e:
            current_app.logger.error(f"전체 예측 중 오류: {e}", exc_info=True)
            flash("예측 중 오류가 발생했습니다. 다시 시도해주세요.")
            return redirect(url_for('main.predict'))
    
    # GET 요청 처리
    prediction_data = get_prediction_data()
    if not prediction_data:
        flash("세션에 예측 데이터가 없습니다. 다시 예측해주세요.")
        return redirect(url_for('main.predict'))

    user_input = prediction_data['user_input']
    all_prediction_results = prediction_data['prediction_results']

    # 초기 화면에 표시할 예측 결과 3개를 구성
    # all_prediction_results는 이제 딕셔너리 형태
    prediction_results_display = [
        all_prediction_results.get('current'),
        all_prediction_results.get(user_input['job_A_category']),
        all_prediction_results.get(user_input['job_B_category'])
    ]

    return render_template(
        'main/predict_result.html',
        user_input=user_input,
        prediction_results=prediction_results_display,
        all_prediction_results=all_prediction_results,  # 모든 예측 결과를 템플릿에 전달
        job_category_map=JOB_CATEGORY_MAP,
        education=EDUCATION_MAP,
        satis_factor_map=SATIS_FACTOR_MAP,
        is_guest=is_guest
    )

@bp.route('/advice', methods=['GET', 'POST'])
def advice():
    user = get_current_user()
    if request.method == 'POST' and not user:
        user_input = _create_user_input(request.form)
        try:
            # 모든 직업군에 대한 예측 실행 (predict_result와 동일하게)
            all_job_codes = list(JOB_CATEGORY_MAP.keys())
            prediction_results = services.run_prediction(user_input, scenarios_to_run=all_job_codes)
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
        # 디버깅: 전달되는 데이터 확인
        current_app.logger.info(f"DEBUG - user_input: job_A_category={user_input.get('job_A_category')}, job_B_category={user_input.get('job_B_category')}")
        current_app.logger.info(f"DEBUG - prediction_results keys: {list(prediction_results.keys()) if isinstance(prediction_results, dict) else 'Not a dict'}")

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
        import subprocess
        import tempfile
        import os
        try:
            current_app.logger.info("별도 프로세스에서 챗봇 실행 시작")

            # 채팅 세션 정보 가져오기
            from app.chat_session import get_current_chat_session
            from app.utils.web_helpers import get_current_user
            chat_session = get_current_chat_session()
            chat_session.add_message("user", user_message)

            # 회원 정보 및 예측 데이터 수집 (Oracle 접근 필요 시)
            user_context = {}
            try:
                current_user = get_current_user()
                if current_user:
                    # 회원 기본 정보
                    user_context['user_info'] = {
                        'age': getattr(current_user, 'age', None),
                        'gender': getattr(current_user, 'gender', None),
                        'education': getattr(current_user, 'education', None),
                        'monthly_income': getattr(current_user, 'monthly_income', None),
                        'job_category': getattr(current_user, 'job_category', None),
                        'job_satisfaction': getattr(current_user, 'job_satisfaction', None)
                    }

                    # 예측 결과 (세션에 저장된 것)
                    from app.utils.web_helpers import get_prediction_data
                    prediction_data = get_prediction_data()
                    if prediction_data:
                        user_context['prediction_results'] = prediction_data.get('prediction_results', {})
                        user_context['user_input'] = prediction_data.get('user_input', {})
            except Exception as e:
                current_app.logger.warning(f"회원 컨텍스트 수집 실패: {e}")

            # 요청 데이터 준비
            request_data = {
                'message': user_message,
                'chat_history': chat_session.get_messages(),
                'context_summary': chat_session.get_context_summary(),
                'user_context': user_context,  # 회원 정보 추가
                'streaming': True
            }

            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as input_file:
                json.dump(request_data, input_file, ensure_ascii=False, indent=2)
                input_path = input_file.name

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as output_file:
                output_path = output_file.name

            try:
                # 별도 프로세스에서 챗봇 워커 실행
                worker_script = os.path.join(current_app.root_path, '..', 'chatbot_worker.py')
                process = subprocess.run([
                    sys.executable, worker_script, input_path, output_path
                ], timeout=300, capture_output=True, text=False, errors='ignore')

                if process.returncode != 0:
                    stderr_text = process.stderr.decode('utf-8', errors='ignore') if process.stderr else 'Unknown error'
                    current_app.logger.error(f"챗봇 워커 프로세스 실패: {stderr_text}")
                    yield f"data: {json.dumps({'error': '챗봇 처리 중 오류가 발생했습니다.'})}\n\n"
                    return

                # 결과 읽기
                with open(output_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)

                if 'error' in result:
                    yield f"data: {json.dumps({'error': result['error']})}\n\n"
                    return

                # 스트리밍 시뮬레이션
                response = result.get('response', '')
                chat_session.add_message("assistant", response, {"streaming": True})

                # 청크 단위로 전송
                chunk_size = 20
                for i in range(0, len(response), chunk_size):
                    chunk = response[i:i + chunk_size]
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                    time.sleep(0.05)

                yield f"data: {json.dumps({'done': True, 'full_response': response})}\n\n"

            finally:
                # 임시 파일 정리
                try:
                    os.unlink(input_path)
                    os.unlink(output_path)
                except:
                    pass

        except subprocess.TimeoutExpired:
            current_app.logger.error("챗봇 워커 프로세스 타임아웃")
            yield f"data: {json.dumps({'error': '요청 처리 시간이 초과되었습니다.'})}\n\n"
        except Exception as e:
            current_app.logger.error(f"별도 프로세스 스트리밍 중 오류: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': 'AI 응답 생성 중 오류가 발생했습니다.'})}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    }
    return Response(generate_stream(), mimetype='text/event-stream', headers=headers)

@bp.route('/ask-ai', methods=['POST'])
def ask_ai():
    import subprocess
    import tempfile
    import os
    data = request.get_json(silent=True) or {}
    user_message = (data.get('message') or '').strip()
    if not user_message:
        return jsonify({'error': '메시지가 없습니다.'}), 400

    try:
        current_app.logger.info("별도 프로세스에서 챗봇 실행 시작")

        # 채팅 세션 정보 가져오기
        from app.chat_session import get_current_chat_session
        from app.utils.web_helpers import get_current_user
        chat_session = get_current_chat_session()
        chat_session.add_message("user", user_message)

        # 회원 정보 및 예측 데이터 수집 (Oracle 접근 필요 시)
        user_context = {}
        try:
            current_user = get_current_user()
            if current_user:
                # 회원 기본 정보
                user_context['user_info'] = {
                    'age': getattr(current_user, 'age', None),
                    'gender': getattr(current_user, 'gender', None),
                    'education': getattr(current_user, 'education', None),
                    'monthly_income': getattr(current_user, 'monthly_income', None),
                    'job_category': getattr(current_user, 'job_category', None),
                    'job_satisfaction': getattr(current_user, 'job_satisfaction', None)
                }

                # 예측 결과 (세션에 저장된 것)
                from app.utils.web_helpers import get_prediction_data
                prediction_data = get_prediction_data()
                if prediction_data:
                    user_context['prediction_results'] = prediction_data.get('prediction_results', {})
                    user_context['user_input'] = prediction_data.get('user_input', {})
        except Exception as e:
            current_app.logger.warning(f"회원 컨텍스트 수집 실패: {e}")

        # 요청 데이터 준비
        request_data = {
            'message': user_message,
            'chat_history': chat_session.get_messages(),
            'context_summary': chat_session.get_context_summary(),
            'user_context': user_context,  # 회원 정보 추가
            'streaming': False
        }

        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as input_file:
            json.dump(request_data, input_file, ensure_ascii=False, indent=2)
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as output_file:
            output_path = output_file.name

        try:
            # 별도 프로세스에서 챗봇 워커 실행
            worker_script = os.path.join(current_app.root_path, '..', 'chatbot_worker.py')
            process = subprocess.run([
                sys.executable, worker_script, input_path, output_path
            ], timeout=300, capture_output=True, text=False, errors='ignore')

            if process.returncode != 0:
                stderr_text = process.stderr.decode('utf-8', errors='ignore') if process.stderr else 'Unknown error'
                current_app.logger.error(f"챗봇 워커 프로세스 실패: {stderr_text}")
                return jsonify({'error': '챗봇 처리 중 오류가 발생했습니다.'}), 500

            # 결과 읽기
            with open(output_path, 'r', encoding='utf-8') as f:
                result = json.load(f)

            if 'error' in result:
                return jsonify({'error': result['error']}), 500

            response = result.get('response', '')
            chat_session.add_message("assistant", response)
            return jsonify({'reply': response})

        finally:
            # 임시 파일 정리
            try:
                os.unlink(input_path)
                os.unlink(output_path)
            except:
                pass

    except subprocess.TimeoutExpired:
        current_app.logger.error("챗봇 워커 프로세스 타임아웃")
        return jsonify({'error': '요청 처리 시간이 초과되었습니다.'}), 500
    except Exception as e:
        current_app.logger.error(f"별도 프로세스 AI 응답 중 오류: {e}", exc_info=True)
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


@bp.route('/api/similar-cases-distribution', methods=['POST'])
def get_similar_cases_distribution():
    """AI 추천 시나리오의 유사 사례 분포 데이터 API"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '요청 데이터가 없습니다.'}), 400

        user_input = data.get('user_input')
        recommended_scenario = data.get('recommended_scenario')

        if not user_input or not recommended_scenario:
            return jsonify({'error': '필수 데이터가 누락되었습니다.'}), 400

        # ML Predictor에서 유사 사례 분포 데이터 생성
        predictor = current_app.extensions.get('ml_predictor')
        if not predictor:
            return jsonify({'error': 'ML 서비스를 사용할 수 없습니다.'}), 500

        distribution_data = predictor.get_similar_cases_distribution(user_input, recommended_scenario)

        return jsonify({
            'success': True,
            'distribution': distribution_data
        })

    except Exception as e:
        current_app.logger.error(f"유사 사례 분포 API 오류: {e}", exc_info=True)
        return jsonify({'error': '서버 오류가 발생했습니다.'}), 500

@bp.route('/rag', methods=['GET', 'POST'])
def rag():
    """RAG 시스템 테스트를 위한 공개 페이지"""
    answer = None
    sources = []
    query = ""
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query:
            try:
                from app.rag_manager import get_rag_manager
                rag_manager = get_rag_manager()
                if rag_manager:
                    # 소스와 답변을 함께 가져오는 새로운 함수 호출
                    results = rag_manager.get_advice_with_sources(query)
                    answer = results.get('answer')
                    sources = results.get('sources', [])
                else:
                    flash("RAG Manager가 초기화되지 않았습니다.", "error")
            except Exception as e:
                current_app.logger.error(f"RAG 테스트 중 오류: {e}", exc_info=True)
                flash(f"오류가 발생했습니다: {e}", "error")
        else:
            flash("질문을 입력해주세요.", "warning")

    return render_template('main/rag.html', query=query, answer=answer, sources=sources)


