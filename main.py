# -*- coding: utf-8 -*-
# 依赖: pip install pandas numpy scikit-learn xgboost joblib openpyxl

import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

RANDOM_STATE = 2025
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "result"
RESULT_DIR.mkdir(exist_ok=True)

COLD_FILE = DATA_DIR / "FRP数据-常温-code.xlsx"
HOT_FILE = DATA_DIR / "FRP数据-高温.xlsx"

T0 = 20.0
USE_LOG_TARGET = False  # 若误差偏比例型，可设为 True

def read_data():
    df_cold = pd.read_excel(COLD_FILE)
    df_cold.columns = [str(c).strip() for c in df_cold.columns]
    df_hot = pd.read_excel(HOT_FILE)
    df_hot.columns = [str(c).strip() for c in df_hot.columns]

    rename_cold = {
        'fc(MPa)': 'fc',
        'la(mm)': 'la',
        'd(mm)': 'd',
        'c/mm': 'c_over_mm',
        'τu(MPa)': 'tau_u',
        'BS': 'BS', 'FT': 'FT', 'FM': 'FM',
    }
    df_cold = df_cold.rename(columns=rename_cold)

    rename_hot = {
        'T/◦C': 'T',
        'fc(MPa)': 'fc',
        'la(mm)': 'la',
        'd(mm)': 'd',
        'c/d': 'c_over_d',
        'τu(MPa)': 'tau_u',
        'BS': 'BS', 'FT': 'FT', 'FM': 'FM',
    }
    df_hot = df_hot.rename(columns=rename_hot)

    if 'c_over_mm' in df_cold.columns and 'd' in df_cold.columns:
        df_cold['c_over_d'] = df_cold['c_over_mm'] / df_cold['d']
    elif 'c_over_d' not in df_cold.columns:
        df_cold['c_over_d'] = np.nan

    df_cold['T'] = T0
    if 'T' in df_hot.columns:
        df_hot['T'] = df_hot['T'].astype(float)

    assert 'tau_u' in df_cold.columns, "常温缺少 τu(MPa)"
    assert 'tau_u' in df_hot.columns, "高温缺少 τu(MPa)"
    return df_cold, df_hot

def select_features(df_cold, df_hot):
    base_candidates = ['FT', 'FM', 'BS', 'd', 'la', 'fc', 'c_over_d']
    base_feats = [c for c in base_candidates if (c in df_cold.columns and c in df_hot.columns)]
    return base_feats

def maybe_log(y):
    return np.log1p(y) if USE_LOG_TARGET else y

def maybe_exp(y):
    return np.expm1(y) if USE_LOG_TARGET else y

def train_f_base_xgb(df_cold, feature_cols):
    Xc = df_cold[feature_cols].values
    yc_raw = df_cold['tau_u'].values
    yc = maybe_log(yc_raw)

    model = XGBRegressor(
        n_estimators=800,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        reg_alpha=0.0,
        random_state=RANDOM_STATE,
        tree_method='hist',
        n_jobs=0
    )
    model.fit(Xc, yc)
    return model

def build_residual_dataset(df_hot_part, f_base, feature_cols, scaler_X=None, scaler_T=None, k_inter=3):
    Xh = df_hot_part[feature_cols].values
    yh_raw = df_hot_part['tau_u'].values
    yh = maybe_log(yh_raw)

    y_base_log = f_base.predict(Xh)  # 这是log域的预测
    r_log = yh - y_base_log

    Th = df_hot_part['T'].values.reshape(-1, 1)
    dT = Th - T0
    dT2 = dT ** 2

    # 标准化供 g 使用
    if scaler_X is None:
        scaler_X = StandardScaler().fit(Xh)
    Xh_s = scaler_X.transform(Xh)

    if scaler_T is None:
        scaler_T = StandardScaler().fit(dT)
    dT_s = scaler_T.transform(dT)
    dT2_s = (dT2 - dT2.mean()) / (dT2.std() + 1e-8)

    # 构造特征：X、ΔT、(ΔT)^2、前k个交互（X*ΔT）
    Z_list = [Xh_s, dT_s, dT2_s]
    k = min(k_inter, Xh_s.shape[1])
    if k > 0:
        inter = Xh_s[:, :k] * dT_s  # k个交互
        Z_list.append(inter)
    Z = np.hstack(Z_list)
    return Z, r_log, y_base_log, scaler_X, scaler_T

def train_g_xgb(Z_train, r_train, Z_valid, r_valid):
    # 小模型+强正则，防止过拟合
    model = XGBRegressor(
        n_estimators=1200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=1.0,
        min_child_weight=1.0,
        random_state=RANDOM_STATE,
        tree_method='hist',
        n_jobs=0
    )
    model.fit(Z_train, r_train,
              eval_set=[(Z_valid, r_valid)],
              verbose=False)
    # 可选：根据evals_result手动挑选最佳迭代，这里简化使用最终模型
    return model

def evaluate_hot_logspace(hot_valid, y_base_log, r_hat_log):
    y_true = hot_valid['tau_u'].values
    y_pred_log = y_base_log + r_hat_log
    y_pred = maybe_exp(y_pred_log)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}, y_pred

def main():
    print("读取数据...")
    df_cold, df_hot = read_data()
    feature_cols = select_features(df_cold, df_hot)
    print("基线特征列:", feature_cols)

    print("训练常温基线 f_base (XGBoost, 目标可选log域)...")
    f_base = train_f_base_xgb(df_cold, feature_cols)

    print("划分高温训练/验证集...")
    hot_tr, hot_va = train_test_split(df_hot, test_size=0.3, random_state=RANDOM_STATE)

    print("构造残差数据集（含 ΔT 和 (ΔT)^2 以及交互）...")
    Z_tr, r_tr, yb_tr_log, scaler_X, scaler_T = build_residual_dataset(hot_tr, f_base, feature_cols, k_inter=3)
    Z_va, r_va, yb_va_log, _, _ = build_residual_dataset(hot_va, f_base, feature_cols, scaler_X, scaler_T, k_inter=3)

    # 简单的bias对齐（消除系统偏差）
    b0 = float(np.mean(r_tr))
    r_tr_centered = r_tr - b0
    r_va_centered = r_va - b0

    print("训练残差模型 g（XGBoost，小模型）...")
    g = train_g_xgb(Z_tr, r_tr_centered, Z_va, r_va_centered)

    print("评估在高温验证集上的总体表现...")
    r_hat_tr = g.predict(Z_tr)
    r_hat_va = g.predict(Z_va)

    # 把bias加回去
    r_hat_tr += b0
    r_hat_va += b0

    report, y_hat_va = evaluate_hot_logspace(hot_va, yb_va_log, r_hat_va)
    print("高温验证集指标:", report)

    # 保存产物
    artifacts = {
        'f_base': f_base,
        'g_xgb': g,
        'scaler_X': scaler_X,
        'scaler_T': scaler_T,
        'feature_cols': feature_cols,
        'T0': T0,
        'USE_LOG_TARGET': USE_LOG_TARGET,
        'bias_b0': b0
    }
    dump(artifacts['f_base'], RESULT_DIR / "f_base_xgb.pkl")
    dump(artifacts['g_xgb'], RESULT_DIR / "g_xgb.pkl")
    dump(artifacts['scaler_X'], RESULT_DIR / "scaler_X.pkl")
    dump(artifacts['scaler_T'], RESULT_DIR / "scaler_T.pkl")
    with open(RESULT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            'feature_cols': feature_cols,
            'T0': T0,
            'USE_LOG_TARGET': USE_LOG_TARGET,
            'bias_b0': b0
        }, f, ensure_ascii=False, indent=2)

    # 输出验证集预测
    out = hot_va.copy()
    out['y_pred'] = y_hat_va
    out['error'] = out['y_pred'] - out['tau_u']
    out.to_excel(RESULT_DIR / "hot_valid_predictions.xlsx", index=False)

    with open(RESULT_DIR / "metrics_hot_valid.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"产物已保存至: {RESULT_DIR}")

if __name__ == "__main__":
    from sklearn.metrics import mean_squared_error
    main()