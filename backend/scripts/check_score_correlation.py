"""检查模型预测分数与实际收益的关系"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.predictor import ModelPredictor
from scipy.stats import spearmanr

df = pd.read_parquet("data/test.parquet")
pred = ModelPredictor("lightgbm")

# 多个截面的平均Spearman相关
dates = sorted(df["date"].unique())
corrs_ret = []
corrs_vol = []

for d in dates[::10]:  # 每10个截面取一个
    section = df[df["date"] == d].copy()
    if len(section) < 50:
        continue
    try:
        pred_df = pred.predict(section)
        merged = pred_df.merge(section[["stock_code", "future_return_1w"]], on="stock_code", how="left")
        
        corr_ret, _ = spearmanr(merged["score"], merged["future_return_1w"])
        corrs_ret.append(corr_ret)
        
        vol_col = "volatility_12w"
        if vol_col in section.columns:
            merged2 = pred_df.merge(section[["stock_code", vol_col]], on="stock_code", how="left")
            corr_vol, _ = spearmanr(merged2["score"], merged2[vol_col])
            corrs_vol.append(corr_vol)
    except:
        pass

print("=== 模型预测分数与实际收益/波动率的关系 ===")
print(f"score vs future_return_1w 平均Spearman相关: {np.mean(corrs_ret):.4f}")
print(f"score vs volatility_12w 平均Spearman相关: {np.mean(corrs_vol):.4f}")
print()

# 关键结论
if np.mean(corrs_ret) < 0:
    print("!!! 预测分数与未来收益负相关 !!!")
    print("模型给高分股票的未来收益反而更低")
    print("这就是Top20亏损、Bottom20盈利的原因")
else:
    print("预测分数与未来收益正相关，但Top20仍亏损")
    print("说明高分股票虽然相对收益好，但绝对收益仍为负")
