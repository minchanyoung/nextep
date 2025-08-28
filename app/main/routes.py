import traceback
from flask import render_template, request, redirect, url_for, flash, session, current_app
from app import services
from app.constants import (
    JOB_CATEGORY_MAP, SATIS_FACTOR_MAP, EDUCATION_MAP,
    REQUIRED_PROFILE_FIELDS, DEFAULT_PREDICTION_RESULTS, MESSAGES
)
from app.utils.session_utils import (
    get_current_username, is_user_logged_in, get_prediction_data,
    set_prediction_data, get_chat_messages, set_chat_messages, add_chat_message
)
from app.utils.auth_utils import login_required, get_current_user, require_profile_complete
from app.utils.response_utils import json_response, error_response, success_response, handle_api_exception
from . import bp
import json

# 상수들은 app.constants에서 임포트

# --- 라우트 함수 ---
@bp.route('/')
def index():
    return render_template('main/index.html')

@bp.route('/about')
def about():
    return render_template('main/about.html')

@bp.route('/faq')
def faq():
    return render_template('main/faq.html')

@bp.route('/industry-form')
def industry_form():
    return render_template('main/industry_form.html')

@bp.route('/industry-result')
def industry_result():
    range_limit = request.args.get('rangeLimit', 5, type=int)
    main_type = request.args.get('detail', '0')
    dummy_data_map = {
        "2025_manufacturing": {"data": [1, 2, 3, 4, range_limit]},
        "2025_it": {"data": [5, 4, 3, 2, 1]},
    }
    return render_template(
        'main/industry_result.html',
        data_map=dummy_data_map, range_limit=range_limit, main_type=main_type
    )

@bp.route('/predict', methods=['GET'])
def predict():
    """비회원/회원 모두 접근 가능한 예측 페이지"""
    user = get_current_user()

    # 회원이면서 프로필이 완료된 경우 자동으로 예측 실행
    if user and hasattr(user, 'age') and user.age:
            # 사용자 프로필 데이터를 user_input 형식으로 변환
            user_input = {
                'age': str(user.age),
                'gender': str(user.gender),
                'education': str(user.education),
                'monthly_income': str(user.monthly_income),
                'current_job_category': str(user.job_category),
                'job_satisfaction': str(user.job_satisfaction),
                'job_A_category': '2', # 기본값
                'job_B_category': '3',  # 기본값
                # 9개 만족도 요인 추가
                'satis_wage': str(user.satis_wage or 3),
                'satis_stability': str(user.satis_stability or 3),
                'satis_growth': str(user.satis_growth or 3),
                'satis_task_content': str(user.satis_task_content or 3),
                'satis_work_env': str(user.satis_work_env or 3),
                'satis_work_time': str(user.satis_work_time or 3),
                'satis_communication': str(user.satis_communication or 3),
                'satis_fair_eval': str(user.satis_fair_eval or 3),
                'satis_welfare': str(user.satis_welfare or 3)
            }

            try:
                prediction_results = services.run_prediction(user_input)
                set_prediction_data(user_input, prediction_results)
                return redirect(url_for('main.predict_result'))
            except Exception as e:
                current_app.logger.error(f"자동 예측 중 오류 발생: {e}")
                flash("프로필 기반 예측 중 오류가 발생했습니다. 수동으로 입력해주세요.")

    # 비회원이거나 프로필이 완료되지 않은 회원의 경우 입력 폼 표시
    return render_template('main/predict.html')


@bp.route('/predict-result', methods=['GET', 'POST'])
def predict_result():
    user_input = {}
    prediction_results = []
    error_message = None
    user = get_current_user()
    is_guest = not user  # 비회원 여부 확인

    if request.method == 'POST':
        user_input = request.form.to_dict()
        if 'job_A_category' not in user_input: user_input['job_A_category'] = '2'
        if 'job_B_category' not in user_input: user_input['job_B_category'] = '3'

        current_app.logger.info(f"Prediction started with user_input: {user_input}")

        try:
            prediction_results = services.run_prediction(user_input)
            current_app.logger.info("Prediction successful.")
            # 회원인 경우에만 세션에 저장
            if not is_guest:
                set_prediction_data(user_input, prediction_results)
        except Exception as e:
            tb_str = traceback.format_exc()
            current_app.logger.error(f"An error occurred during prediction: {e}\n{tb_str}")
            prediction_results = DEFAULT_PREDICTION_RESULTS
            error_message = "예측 중 오류가 발생했습니다. 입력값을 확인하거나 잠시 후 다시 시도해주세요."
    elif request.method == 'GET':
        # GET 요청은 회원만 가능 (세션 데이터 사용)
        if is_guest:
            flash("예측 데이터가 없습니다. 다시 예측을 시도해주세요.")
            return redirect(url_for('main.predict'))

        prediction_data = get_prediction_data()
        if prediction_data:
            user_input = prediction_data['user_input']
            prediction_results = prediction_data['prediction_results']
        else:
            flash("예측 데이터가 없습니다. 다시 예측을 시도해주세요.")
            return redirect(url_for('main.predict'))

    return render_template(
        'main/predict_result.html',
        user_input=user_input,
        prediction_results=prediction_results,
        job_category_map=JOB_CATEGORY_MAP,
        focus_key_name=SATIS_FACTOR_MAP.get(user_input.get('satis_focus_key')),
        error_message=error_message,
        education=EDUCATION_MAP,
        satis_focus_key=SATIS_FACTOR_MAP,
        is_guest=is_guest  # 템플릿에서 사용할 수 있도록 전달
    )


@bp.route('/advice', methods=['GET', 'POST'])
def advice():
    user = get_current_user()
    is_guest = not user
    
    # POST 요청인 경우 (비회원이 predict_result에서 넘어온 경우)
    if request.method == 'POST':
        # 폼 데이터에서 예측 정보 가져오기
        user_input = {
            'age': request.form.get('age'),
            'gender': request.form.get('gender'),
            'education': request.form.get('education'),
            'monthly_income': request.form.get('monthly_income'),
            'current_job_category': request.form.get('current_job_category'),
            'job_satisfaction': request.form.get('job_satisfaction'),
            'satis_focus_key': request.form.get('satis_focus_key'),
            'job_A_category': request.form.get('job_A_category', '2'),
            'job_B_category': request.form.get('job_B_category', '3')
        }
        
        # 예측 결과 다시 생성 (비회원의 경우)
        try:
            prediction_results = services.run_prediction(user_input)
        except Exception as e:
            current_app.logger.error(f"비회원 AI 조언용 예측 중 오류 발생: {e}")
            flash("예측 데이터 생성 중 오류가 발생했습니다. 다시 시도해주세요.")
            return redirect(url_for('main.predict'))
            
    else:  # GET 요청인 경우 (회원이 세션 데이터 사용)
        if is_guest:
            flash("예측 데이터가 없습니다. 다시 예측을 시도해주세요.")
            return redirect(url_for('main.predict'))
            
        prediction_data = get_prediction_data()
        if not prediction_data:
            flash("예측 데이터가 없습니다. 다시 예측을 시도해주세요.")
            return redirect(url_for('main.predict'))
        
        user_input = prediction_data['user_input']
        prediction_results = prediction_data['prediction_results']

    try:
        from app.chat_session import get_current_chat_session, clear_chat_session
        
        # 회원의 경우에만 세션 관리 (비회원은 일회성)
        if not is_guest:
            # 새로운 상담을 위해 기존 세션 초기화
            clear_chat_session()
        
        # AI 조언 생성
        ai_advice = services.generate_career_advice_hf(user_input, prediction_results, JOB_CATEGORY_MAP, SATIS_FACTOR_MAP)
        context_summary = services.summarize_context(user_input, prediction_results, JOB_CATEGORY_MAP, SATIS_FACTOR_MAP)
        
        # 회원의 경우에만 채팅 세션 설정
        if not is_guest:
            # 새 채팅 세션에 컨텍스트 설정
            chat_session = get_current_chat_session()
            chat_session.set_user_context(user_input, prediction_results, JOB_CATEGORY_MAP, SATIS_FACTOR_MAP)
            
            # 초기 메시지들을 채팅 세션에 저장
            chat_session.add_message("assistant", context_summary, {"type": "context_summary"})
            chat_session.add_message("assistant", ai_advice, {"type": "initial_advice"})
            
            # 기존 호환성을 위한 세션 저장
            chat_messages = [
                {"role": "system", "content": "당신은 친절한 커리어 코치다. 한국어로 간결하고 실용적으로 답한다."},
                {"role": "assistant", "content": context_summary},
                {"role": "assistant", "content": ai_advice}
            ]
            session['chat_messages'] = chat_messages

    except Exception as e:
        current_app.logger.error(f"AI 조언 생성 페이지 오류: {e}")
        ai_advice = "AI 조언을 생성하기 위한 데이터를 불러오는 데 실패했습니다. 예측 페이지로 돌아가 다시 시도해주세요."

    return render_template('main/advice.html', ai_advice=ai_advice, is_guest=is_guest)

@bp.route('/db-test')
def db_test():
    try:
        status = services.example_db_query()
        return f"<h1>DB 테스트 결과: {status}</h1>"
    except Exception as e:
        current_app.logger.error(f"DB 테스트 실패: {e}")
        return f"<h1>DB 테스트 실패: {e}</h1>"

@bp.route('/ask-ai', methods=['POST'])
@handle_api_exception
def ask_ai():
    from app.chat_session import get_current_chat_session
    
    data = request.get_json()
    user_message = data.get('message')
    client_history = data.get('history', [])
    use_streaming = data.get('streaming', False)  # 스트리밍 옵션

    if not user_message:
        return error_response('메시지가 없습니다.', 400)

    user = get_current_user()
    is_guest = not user

    try:
        if is_guest:
            # 비회원의 경우 간단한 컨텍스트 사용 (세션 관리 없음)
            context_summary = "비회원 사용자의 추가 질문입니다. 일반적인 커리어 조언을 제공해주세요."
            
            if use_streaming:
                return json.dumps({'use_streaming': True, 'message': '스트리밍 모드로 전환합니다.'})
            else:
                # 비회원용 간단한 AI 응답
                reply = services.generate_follow_up_advice(user_message, client_history, context_summary)
                return json_response({'reply': reply})
        else:
            # 회원의 경우 기존 로직 사용
            # 개선된 세션 관리 시스템 사용
            chat_session = get_current_chat_session()
            context_summary = chat_session.get_context_summary()
            
            # 사용자 메시지 저장
            chat_session.add_message("user", user_message)
            
            if use_streaming:
                # 스트리밍 응답은 직접 처리하지 않고 클라이언트에서 별도 요청하도록 함
                return json.dumps({'use_streaming': True, 'message': '스트리밍 모드로 전환합니다.'})
            else:
                # 컨텍스트를 포함하여 서비스 호출
                reply = services.generate_follow_up_advice(user_message, client_history, context_summary)
                
                # AI 응답 저장
                chat_session.add_message("assistant", reply)
                
                # 기존 호환성을 위한 세션 업데이트
                add_chat_message("user", user_message)
                add_chat_message("assistant", reply)
                
                return json_response({'reply': reply})
            
    except Exception as e:
        current_app.logger.error(f"AI 추가 질문 처리 중 오류 발생: {e}")
        return error_response('AI 응답 생성 중 오류가 발생했습니다.', 500)

@bp.route('/ask-ai-stream', methods=['POST'])
def ask_ai_stream():
    """스트리밍 응답을 위한 새로운 엔드포인트"""
    from flask import Response
    from app.chat_session import get_current_chat_session
    import time
    
    data = request.get_json()
    user_message = data.get('message')
    client_history = data.get('history', [])

    if not user_message:
        return json.dumps({'error': '메시지가 없습니다.'}), 400

    user = get_current_user()
    is_guest = not user

    def generate_stream():
        try:
            if is_guest:
                # 비회원의 경우 간단한 컨텍스트 사용
                context_summary = "비회원 사용자의 추가 질문입니다. 일반적인 커리어 조언을 제공해주세요."
                
                full_response = ""
                
                # 비회원용 스트리밍 응답 생성
                for chunk in services.generate_follow_up_advice_stream(user_message, client_history, context_summary):
                    if chunk and chunk.strip():
                        full_response += chunk
                        # SSE 형식으로 전송
                        yield f"data: {json.dumps({'chunk': chunk, 'done': False})}"
                        time.sleep(0.01)  # 스트리밍 효과를 위한 작은 딜레이
                
                # 완료 시그널과 함께 전체 응답 전송
                yield f"data: {json.dumps({'chunk': '', 'done': True, 'full_response': full_response})}"
                
            else:
                # 회원의 경우 기존 로직 사용
                # 개선된 세션 관리 사용
                chat_session = get_current_chat_session()
                context_summary = chat_session.get_context_summary()
                
                # 사용자 메시지 저장 (아직 저장되지 않은 경우)
                if not chat_session.messages or chat_session.messages[-1]['content'] != user_message:
                    chat_session.add_message("user", user_message)
                
                full_response = ""
                
                # 스트리밍 응답 생성 (컨텍스트 포함)
                for chunk in services.generate_follow_up_advice_stream(user_message, client_history, context_summary):
                    if chunk and chunk.strip():
                        full_response += chunk
                        # SSE 형식으로 전송
                        yield f"data: {json.dumps({'chunk': chunk, 'done': False})}"
                        time.sleep(0.01)  # 스트리밍 효과를 위한 작은 딜레이
                
                # AI 응답을 채팅 세션에 저장
                chat_session.add_message("assistant", full_response, {"streaming": True})
                
                # 완료 시그널과 함께 전체 응답 전송
                yield f"data: {json.dumps({'chunk': '', 'done': True, 'full_response': full_response})}"
                
                # 기존 호환성을 위한 세션 업데이트
                messages = session.get('chat_messages', [])
                messages.append({"role": "user", "content": user_message})
                messages.append({"role": "assistant", "content": full_response})
                session['chat_messages'] = messages
            
        except Exception as e:
            current_app.logger.error(f"스트리밍 AI 응답 중 오류: {e}")
            yield f"data: {json.dumps({'error': 'AI 응답 생성 중 오류가 발생했습니다.'})}"

    return Response(
        generate_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # nginx 버퍼링 방지
        }
    )


@bp.route('/chat/new', methods=['POST'])
@handle_api_exception
def new_chat():
    """새로운 채팅 세션 시작"""
    from app.chat_session import clear_chat_session
    clear_chat_session()
    return success_response('새로운 채팅을 시작했습니다.')

@bp.route('/chat/history', methods=['GET'])
def chat_history():
    """현재 채팅 세션의 대화 기록 조회"""
    from app.chat_session import get_current_chat_session
    try:
        chat_session = get_current_chat_session()
        messages = chat_session.get_messages()
        
        # 클라이언트 형식으로 변환
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                'sender': 'ai' if msg['role'] == 'assistant' else 'user',
                'text': msg['content'],
                'timestamp': msg.get('timestamp'),
                'metadata': msg.get('metadata', {})
            })
        
        return json.dumps({
            'messages': formatted_messages,
            'session_info': {
                'created_at': chat_session.created_at.isoformat(),
                'last_activity': chat_session.last_activity.isoformat(),
                'context_summary': chat_session.get_context_summary()
            }
        })
    except Exception as e:
        current_app.logger.error(f"채팅 기록 조회 오류: {e}")
        return json.dumps({'error': '채팅 기록 조회 중 오류가 발생했습니다.'}), 500

@bp.route('/chat/context', methods=['GET'])
def chat_context():
    """채팅 컨텍스트 정보 조회"""
    from app.chat_session import get_current_chat_session
    try:
        chat_session = get_current_chat_session()
        return json.dumps({
            'context_summary': chat_session.get_context_summary(),
            'user_profile': chat_session.user_profile,
            'has_prediction': bool(chat_session.prediction_context),
            'message_count': len(chat_session.messages)
        })
    except Exception as e:
        current_app.logger.error(f"채팅 컨텍스트 조회 오류: {e}")
        return json.dumps({'error': '컨텍스트 조회 중 오류가 발생했습니다.'}), 500

@bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = get_current_user()

    if request.method == 'POST':
        # 프로필 업데이트 로직
        age = request.form.get('age', type=int)
        gender = request.form.get('gender', type=int)
        education = request.form.get('education', type=int)
        monthly_income = request.form.get('monthly_income', type=int)
        job_category = request.form.get('job_category', type=int)
        job_satisfaction = request.form.get('job_satisfaction', type=int)
        
        # 9개 만족도 요인 수집
        satisfaction_factors = {
            'satis_wage': request.form.get('satis_wage', type=int),
            'satis_stability': request.form.get('satis_stability', type=int),
            'satis_growth': request.form.get('satis_growth', type=int),
            'satis_task_content': request.form.get('satis_task_content', type=int),
            'satis_work_env': request.form.get('satis_work_env', type=int),
            'satis_work_time': request.form.get('satis_work_time', type=int),
            'satis_communication': request.form.get('satis_communication', type=int),
            'satis_fair_eval': request.form.get('satis_fair_eval', type=int),
            'satis_welfare': request.form.get('satis_welfare', type=int)
        }

        success = services.update_user_profile(
            user.id, age, gender, education, monthly_income, job_category, job_satisfaction, **satisfaction_factors
        )
        if success:
            flash('프로필이 성공적으로 업데이트되었습니다.')
        else:
            flash('프로필 업데이트에 실패했습니다.')
        return redirect(url_for('main.profile'))

    return render_template('main/profile.html', user=user,
                           job_category_map=JOB_CATEGORY_MAP,
                           satis_factor_map=SATIS_FACTOR_MAP,
                           education_map=EDUCATION_MAP)