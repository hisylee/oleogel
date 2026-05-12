import numpy as np
import pandas as pd
import base64
import io
import os
import json
import joblib
import struct

MODEL_NAMES = ['AdaBoost', 'GradientBoosting', 'XGBoost', 'CatBoost']
META_COLS = ['Sample', 'Concentration', 'Gelator', 'Oil']
TARGET_COLS = ['hardness', 'spreadability']

WAVELENGTHS = np.linspace(936, 1716.5, 224)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_BASE_DIR, '..', 'model', 'cookie')
DATA_DIR = os.path.join(_BASE_DIR, '..', 'data', 'cookie-data')

TRAIN_PATHS = {
    'hardness': os.path.join(DATA_DIR, 'Oleogel-HSI-average-cookie-hardness.xlsx'),
    'spreadability': os.path.join(DATA_DIR, 'Oleogel-HSI-average-cookie-spread.xlsx'),
}

# --- Cached model storage ---
_models = {}
_scalers = {}
_loaded = False

# --- scatter 이미지 캐시 (디스크에서 로드한 base64) ---
_scatter_imgs = {}    # {target: {model_name: base64}}
_scatter_params = {}  # {target: {model_name: {left, bottom, width, height}}}
_perf_cache = {}      # {target: {display_name: {r2_train, r2_test, rmse_train, rmse_test}}}


def train_and_save_models():
    """
    모델 학습 + pkl 저장 + scatter 이미지 PNG 저장 (서버에서 1회 실행)
    이후 요청은 PNG 파일만 읽으므로 메모리 부담 없음
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor

    os.makedirs(MODEL_DIR, exist_ok=True)

    for target, path in TRAIN_PATHS.items():
        print(f'\n=== {target} 모델 학습 ===')
        df = pd.read_excel(path, header=0)
        df1 = df.drop(META_COLS, axis=1)
        data = np.array(df1, dtype=np.float32)
        x_data = data[:, :-1]
        y_data = data[:, [-1]]

        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2, random_state=42
        )

        x_scaler = MinMaxScaler()
        x_train_s = x_scaler.fit_transform(x_train)
        x_test_s = x_scaler.transform(x_test)

        y_scaler = MinMaxScaler()
        y_train_s = y_scaler.fit_transform(y_train)
        y_test_s = y_scaler.transform(y_test)

        model_defs = {
            'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            'CatBoost': CatBoostRegressor(n_estimators=100, random_state=42, verbose=0),
        }

        ax_params_all = {}
        perf_all = {}

        for name, model in model_defs.items():
            model.fit(x_train_s, y_train_s.ravel())
            joblib.dump(model, os.path.join(MODEL_DIR, f'{name}_{target}_model.pkl'))

            y_train_pred = model.predict(x_train_s)
            y_test_pred = model.predict(x_test_s)

            r2_tr = r2_score(y_train_s, y_train_pred)
            r2_te = r2_score(y_test_s, y_test_pred)
            rmse_tr = float(np.sqrt(mean_squared_error(y_train_s, y_train_pred)))
            rmse_te = float(np.sqrt(mean_squared_error(y_test_s, y_test_pred)))
            print(f'  {name}: R2(train)={r2_tr:.4f}, R2(test)={r2_te:.4f}')

            dn = 'GBM' if name == 'GradientBoosting' else name
            perf_all[dn] = {
                'r2_train': round(r2_tr, 4), 'r2_test': round(r2_te, 4),
                'rmse_train': round(rmse_tr, 4), 'rmse_test': round(rmse_te, 4),
            }

            # scatter 이미지 생성 (초록 동그라미 없이) & PNG 저장
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.scatter(y_train_s, y_train_pred, c='red', alpha=0.6, s=20,
                       label='Training dataset', zorder=2)
            ax.scatter(y_test_s, y_test_pred, c='blue', alpha=0.6, s=20,
                       label='Testing dataset', zorder=2)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=1, zorder=1)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(dn, fontsize=13, fontweight='bold')
            ax.set_xlabel('Experimental', fontsize=10)
            ax.set_ylabel('Predicted', fontsize=10)
            ax.legend(fontsize=7, loc='upper left')
            plt.tight_layout()
            fig.canvas.draw()

            bbox = ax.get_position()
            ax_params_all[name] = {
                'left': round(bbox.x0, 6), 'bottom': round(bbox.y0, 6),
                'width': round(bbox.width, 6), 'height': round(bbox.height, 6),
            }

            png_path = os.path.join(MODEL_DIR, f'scatter_{name}_{target}.png')
            fig.savefig(png_path, format='png', dpi=120)
            plt.close(fig)
            print(f'  scatter 저장: {png_path}')

        joblib.dump(x_scaler, os.path.join(MODEL_DIR, f'x_scaler_{target}.pkl'))
        joblib.dump(y_scaler, os.path.join(MODEL_DIR, f'y_scaler_{target}.pkl'))

        # ax_params, perf JSON 저장
        with open(os.path.join(MODEL_DIR, f'scatter_ax_params_{target}.json'), 'w') as f:
            json.dump(ax_params_all, f)
        with open(os.path.join(MODEL_DIR, f'perf_{target}.json'), 'w') as f:
            json.dump(perf_all, f)

    print(f'\n모든 모델 및 scatter 이미지가 {MODEL_DIR}에 저장되었습니다.')

    global _loaded
    _loaded = False


def _load_models():
    """pkl 로드 + 디스크의 scatter PNG → base64 캐싱"""
    global _models, _scalers, _loaded, _scatter_imgs, _scatter_params, _perf_cache
    if _loaded:
        return True

    for target in ['hardness', 'spreadability']:
        _models[target] = {}
        for name in MODEL_NAMES:
            p = os.path.join(MODEL_DIR, f'{name}_{target}_model.pkl')
            if os.path.exists(p):
                _models[target][name] = joblib.load(p)

        xp = os.path.join(MODEL_DIR, f'x_scaler_{target}.pkl')
        yp = os.path.join(MODEL_DIR, f'y_scaler_{target}.pkl')
        if os.path.exists(xp) and os.path.exists(yp):
            _scalers[target] = {'x': joblib.load(xp), 'y': joblib.load(yp)}

        # scatter PNG 로드 → base64
        _scatter_imgs[target] = {}
        for name in MODEL_NAMES:
            png_path = os.path.join(MODEL_DIR, f'scatter_{name}_{target}.png')
            if os.path.exists(png_path):
                with open(png_path, 'rb') as f:
                    _scatter_imgs[target][name] = base64.b64encode(f.read()).decode('utf-8')

        # ax_params JSON 로드
        ap_path = os.path.join(MODEL_DIR, f'scatter_ax_params_{target}.json')
        if os.path.exists(ap_path):
            with open(ap_path, 'r') as f:
                _scatter_params[target] = json.load(f)

        # perf JSON 로드
        perf_path = os.path.join(MODEL_DIR, f'perf_{target}.json')
        if os.path.exists(perf_path):
            with open(perf_path, 'r') as f:
                _perf_cache[target] = json.load(f)

    _loaded = bool(
        _models.get('hardness') and _models.get('spreadability')
        and _scalers.get('hardness') and _scalers.get('spreadability')
    )
    return _loaded


def _extract_features(data):
    """입력 데이터에서 224개 파장 특성 추출"""
    if isinstance(data, np.ndarray):
        return data.astype(np.float32)

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        for c in META_COLS:
            if c in df.columns:
                df = df.drop(c, axis=1)
        for c in TARGET_COLS:
            if c in df.columns:
                df = df.drop(c, axis=1)
        df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        return df.values.astype(np.float32)

    return None


def parse_hdr_file(hdr_file_obj, dat_file_obj=None):
    """ENVI .hdr 초분광 파일 파싱"""
    hdr_text = hdr_file_obj.read().decode('utf-8', errors='ignore')
    header = {}
    for line in hdr_text.split('\n'):
        if '=' in line:
            key, val = line.split('=', 1)
            header[key.strip().lower()] = val.strip()

    wavelengths_str = header.get('wavelength', '')
    if '{' in wavelengths_str:
        start = hdr_text.find('wavelength')
        if start >= 0:
            brace_start = hdr_text.find('{', start)
            brace_end = hdr_text.find('}', brace_start)
            if brace_start >= 0 and brace_end >= 0:
                wavelengths_str = hdr_text[brace_start + 1:brace_end]

    wl_values = []
    for w in wavelengths_str.replace('{', '').replace('}', '').split(','):
        w = w.strip()
        if w:
            try:
                wl_values.append(float(w))
            except ValueError:
                pass

    n_bands = int(header.get('bands', len(wl_values) or 224))
    n_samples = int(header.get('lines', 1))
    n_pixels = int(header.get('samples', 1))
    data_type = int(header.get('data type', 4))

    if dat_file_obj is not None:
        raw = dat_file_obj.read()
        dtype_map = {1: np.uint8, 2: np.int16, 3: np.int32,
                     4: np.float32, 5: np.float64, 12: np.uint16}
        dt = dtype_map.get(data_type, np.float32)
        arr = np.frombuffer(raw, dtype=dt)
        total = n_samples * n_pixels * n_bands
        if arr.size >= total:
            arr = arr[:total].reshape(n_samples, n_pixels, n_bands)
            spectra = arr.mean(axis=1).astype(np.float32)
        else:
            spectra = arr[:n_bands].reshape(1, -1).astype(np.float32)

        if spectra.shape[1] != 224 and len(wl_values) == spectra.shape[1]:
            from scipy.interpolate import interp1d
            interp_spectra = []
            for row in spectra:
                f = interp1d(wl_values, row, kind='linear', fill_value='extrapolate')
                interp_spectra.append(f(WAVELENGTHS))
            spectra = np.array(interp_spectra, dtype=np.float32)

        return spectra
    return None


def result_calc(data):
    """초분광 데이터로 쿠키 hardness & spreadability 예측"""
    if not _load_models():
        return {'error': '모델 파일이 없습니다. train_and_save_models()를 먼저 실행하세요.'}

    x_data = _extract_features(data)
    if x_data is None:
        return {'error': '지원하지 않는 데이터 형식입니다.'}
    if x_data.shape[1] != 224:
        return {'error': f'잘못된 형식의 자료가 업로드 되었습니다. 데이터 형식(스펙트럼 특성 224개 필요 - 현재: {x_data.shape[1]}개)을 확인해주세요.'}

    n_samples = x_data.shape[0]

    # 예측
    predictions = {}
    for target in ['hardness', 'spreadability']:
        predictions[target] = {}
        x_s = _scalers[target]['x'].transform(x_data)
        for name, model in _models[target].items():
            yp_s = model.predict(x_s)
            yp = _scalers[target]['y'].inverse_transform(yp_s.reshape(-1, 1))
            if n_samples == 1:
                predictions[target][name] = round(float(yp[0, 0]), 4)
            else:
                predictions[target][name] = [round(float(v), 4) for v in yp.flatten()]

    # 평균 예측값
    if n_samples == 1:
        h_vals = list(predictions['hardness'].values())
        s_vals = list(predictions['spreadability'].values())
    else:
        h_vals = [np.mean(v) for v in predictions['hardness'].values()]
        s_vals = [np.mean(v) for v in predictions['spreadability'].values()]

    h_mean = round(float(np.mean(h_vals)), 4)
    s_mean = round(float(np.mean(s_vals)), 4)

    # 막대그래프
    h_preds = {k: (v if isinstance(v, (int, float)) else np.mean(v))
               for k, v in predictions['hardness'].items()}
    s_preds = {k: (v if isinstance(v, (int, float)) else np.mean(v))
               for k, v in predictions['spreadability'].items()}
    bar_chart_h = _make_bar_chart_single(h_preds, 'Hardness', h_mean)
    bar_chart_s = _make_bar_chart_single(s_preds, 'Spreadability', s_mean)

    # pred_scaled (JS canvas 동그라미 위치 계산용)
    pred_scaled_h = float(_scalers['hardness']['y'].transform([[h_mean]])[0, 0])
    pred_scaled_s = float(_scalers['spreadability']['y'].transform([[s_mean]])[0, 0])

    return {
        'regression': {'hardness': h_mean, 'spreadability': s_mean},
        'predictions_detail': predictions,
        'bar_chart_hardness': bar_chart_h,
        'bar_chart_spread': bar_chart_s,
        'scatter_h_imgs': _scatter_imgs.get('hardness', {}),
        'scatter_h_params': _scatter_params.get('hardness', {}),
        'scatter_s_imgs': _scatter_imgs.get('spreadability', {}),
        'scatter_s_params': _scatter_params.get('spreadability', {}),
        'pred_scaled_h': pred_scaled_h,
        'pred_scaled_s': pred_scaled_s,
        'perf_hardness': _perf_cache.get('hardness', {}),
        'perf_spread': _perf_cache.get('spreadability', {}),
        'n_samples': n_samples,
    }


def _make_bar_chart_single(preds, title, mean_val):
    """단일 타겟 막대그래프"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#4a90d9', '#50c878', '#f0a830', '#e05555']
    names = list(preds.keys())
    display_names = ['GBM' if n == 'GradientBoosting' else n for n in names]
    vals = [float(v) for v in preds.values()]

    ax.bar(display_names, vals, color=colors[:len(names)], width=0.5)
    ax.axhline(y=mean_val, color='#333', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(len(names) - 0.5, mean_val, f'Mean: {mean_val:.2f}',
            ha='right', va='bottom', fontsize=8, color='#333')
    ax.set_ylabel('Predicted Value', fontsize=9)
    ax.set_title(f'{title} - Model Prediction', fontsize=13, fontweight='bold')

    labels = [f'{dn}\n({v:.2f})' for dn, v in zip(display_names, vals)]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)

    vmin = min(vals) if vals else 0
    vmax = max(vals) if vals else 1
    margin = (vmax - vmin) * 0.4 if vmax > vmin else vmax * 0.2
    ax.set_ylim(max(0, vmin - margin), vmax + margin * 0.7)
    plt.tight_layout()
    return _fig_to_base64(fig)


def _fig_to_base64(fig):
    """matplotlib figure → base64 PNG"""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def get_sample_data(sample_id='1'):
    """체험용 샘플 데이터 반환"""
    if sample_id == '2':
        path = os.path.join(DATA_DIR, 'Oleogel-HSI-average-cookie-hardness-new-sample2.xlsx')
    else:
        path = os.path.join(DATA_DIR, 'Oleogel-HSI-average-cookie-hardness-new.xlsx')

    if os.path.exists(path):
        return pd.read_excel(path, header=0)

    np.random.seed(0)
    wavelengths = np.linspace(936, 1720, 224)
    spectrum = 0.5 - (wavelengths - 936) * 0.0003 + np.random.normal(0, 0.01, 224)
    return np.array([np.clip(spectrum, 0, 1).round(6)], dtype=np.float32)
