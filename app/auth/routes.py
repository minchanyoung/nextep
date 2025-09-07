from flask import render_template, request, redirect, url_for, flash
from datetime import timedelta
from app import services
from app.utils.web_helpers import set_user_session, clear_user_session
from . import bp

@bp.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if services.verify_user(username, password):
            set_user_session(username)
            # 로그인 성공 시 메인 페이지로 리디렉션
            return redirect(url_for('main.index'))
        else:
            # 로그인 실패 시 오류 메시지와 함께 로그인 페이지를 다시 렌더링
            flash('아이디 또는 비밀번호가 올바르지 않습니다.')
            return redirect(url_for('auth.login'))

    # GET 요청 시 로그인 폼을 보여줍니다.
    return render_template('auth/login.html')

@bp.route('/signup', methods=('GET', 'POST'))
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        result = services.create_user(username, password, email)
        if result is True:
            flash('회원가입이 완료되었습니다. 로그인해주세요.')
            return redirect(url_for('auth.login'))
        else:
            flash(result) 
            return redirect(url_for('auth.signup'))
            
    # GET 요청 시 회원가입 폼을 보여줍니다.
    return render_template('auth/signup.html')

@bp.route('/logout')
def logout():
    # 세션을 비워서 로그아웃 처리
    clear_user_session()
    return redirect(url_for('main.index'))
