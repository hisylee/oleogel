from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
try:
  import src.uswheat_volume as us_vol
  import src.rice_oiluptake as rice_ot
  import src.uswheat_variety as us_var
  import src.uswheat_spreadability as us_spread
  import src.uswheat_resistance as us_res
  import src.uswheat_extensibility as us_ext
  import src.buckwheat_2 as buck
  import src.Mixolab_ext as Mixo_ext
  import src.tools as tools
except ImportError as e:
  print(f"[WARNING] 일부 모듈 로드 실패 (tensorflow 등 미설치): {e}")
  us_vol = rice_ot = us_var = us_spread = us_res = us_ext = buck = Mixo_ext = tools = None
import src.cookie_prediction as cookie_pred

# app 변수 정의
app = Flask(__name__)

# 파일 업로드 설정
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB

# 모델에서의 입력 오류 메시지 내용
error_message = "입력값을 확인해주세요. (Please check your inputs.)"

#############################################################################################
#######################################   웹 페이지   #######################################
#############################################################################################

# 메인 페이지
@app.route('/')
def index():
  return render_template('index.html')

@app.route('/terms')
def terms():
  return render_template('terms.html')

@app.route('/why')
def why():
  return render_template('why.html')


# # oil uptake 페이지
@app.route('/rice_oiluptake_res/<res>')
def rice_oiluptake_res(res):
  return render_template('rice_oiluptake.html', result=res)

@app.route('/rice_oiluptake', methods=['GET', 'POST'])
def rice_oiluptake():
  if request.method == 'GET':
    return render_template('rice_oiluptake.html')
  if request.method == 'POST':
    inputList = tools.getInput(4)
    if inputList != -1:
      result_num = rice_ot.result_calc(inputList)
      result_num = result_num.round(1)
      result = "유탕 후 예측된 흡유량은 " + str(result_num) + "% 입니다."
    else:
      result = error_message
    return redirect('{}#result'.format(url_for('rice_oiluptake_res', res = result)))

# 쿠키 물성 예측 페이지
@app.route('/cookie_prediction', methods=['GET', 'POST'])
def cookie_prediction():
  if request.method == 'GET':
    return render_template('cookie_prediction.html')
  if request.method == 'POST':
    try:
      import pandas as pd

      # 0) 샘플 데이터 체험 버튼
      sample_val = request.form.get('sample')
      if sample_val in ('1', '2'):
        sample = cookie_pred.get_sample_data(sample_id=sample_val)
        pred = cookie_pred.result_calc(sample)
        return render_template('cookie_prediction.html', result=pred)

      # 1) 파일 업로드 확인
      file = request.files.get('datafile')
      dat_file = request.files.get('datfile')  # .hdr 에 대응하는 바이너리 파일
      if file is not None and file.filename != '':
        filename = file.filename
        ext = filename.rsplit('.', 1)[-1].lower()

        if ext == 'csv':
          df = pd.read_csv(file)
        elif ext in ('xlsx', 'xls'):
          df = pd.read_excel(file, sheet_name=0, engine='openpyxl')
        elif ext == 'hdr':
          # 초분광 .hdr 파일 처리
          dat_obj = dat_file if (dat_file and dat_file.filename) else None
          spectra = cookie_pred.parse_hdr_file(file, dat_obj)
          if spectra is None:
            result = {'error': '.hdr 파일을 파싱할 수 없습니다. 바이너리 데이터 파일(.dat/.raw/.img)도 함께 업로드해주세요.'}
            return render_template('cookie_prediction.html', result=result)
          pred = cookie_pred.result_calc(spectra)
          return render_template('cookie_prediction.html', result=pred)
        else:
          result = {'error': '지원하지 않는 파일 형식입니다. (.xlsx, .csv, .hdr 가능)'}
          return render_template('cookie_prediction.html', result=result)

        pred = cookie_pred.result_calc(df)
        return render_template('cookie_prediction.html', result=pred)

      # 2) 아무것도 입력 안 한 경우
      result = {'error': '엑셀/CSV 또는 초분광(.hdr) 파일을 업로드해주세요.'}
      return render_template('cookie_prediction.html', result=result)

    except Exception as e:
      result = {'error': '오류가 발생했습니다: ' + str(e)}
      return render_template('cookie_prediction.html', result=result)


#############################################################################################
######################################   파이썬 코드   ######################################
#############################################################################################

# 실행
if __name__ == '__main__':
  #app.run(host='0.0.0.0', port=5000) # 업로드시
  app.run() # 컴에서 실행 시