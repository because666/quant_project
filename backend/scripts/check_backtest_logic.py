"""检查回测合成价格逻辑和模型选股收益"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

# 1. 检查合成价格逻辑
df = pd.read_parquet("data/test.parquet")
print("=== 1. future_return_1w 分布 ===")
print(df["future_return_1w"].describe())
print()

# 2. 模型选股的平均future_return_1w
from src.predictor import ModelPredictor
from src.data_loader import load_factor_columns
pred = ModelPredictor("lightgbm")
factor_cols = load_factor_columns()

# 加载测试集
test_df = df.copy()
available_cols = [c for c in factor_cols if c in test_df.columns]
test_df = test_df.dropna(subset=["future_return_1w"])

# 按截面预测并选Top20
dates = sorted(test_df["date"].unique())
top20_returns = []
all_returns = []

for d in dates[:20]:  # 只看前20个截面
    section = test_df[test_df["date"] == d].copy()
    if len(section) < 20:
        continue
    # 用模型预测
    try:
        pred_df = pred.predict(section)
        top20 = pred_df.head(20)
        top20_ret = top20.merge(section[["stock_code", "future_return_1w"]], on="stock_code", how="left")
        top20_mean = top20_ret["future_return_1w"].mean()
        all_mean = section["future_return_1w"].mean()
        top20_returns.append(top20_mean)
        all_returns.append(all_mean)
    except Exception as e:
        print(f"  截面 {d} 预测失败: {e}")

print("=== 2. 模型选股 vs 全市场平均 future_return_1w ===")
if top20_returns:
    print(f"Top20平均future_return_1w: {np.mean(top20_returns):.4f}")
    print(f"全市场平均future_return_1w: {np.mean(all_returns):.4f}")
    print(f"差值: {np.mean(top20_returns) - np.mean(all_returns):.4f}")
    print(f"Top20正收益截面占比: {np.mean([r > 0 for r in top20_returns]):.2%}")
    print(f"全市场正收益截面占比: {np.mean([r > 0 for r in all_returns]):.2%}")
print()

# 3. 检查合成价格 vs 实际收益
print("=== 3. 合成价格验证 ===")
sample_code = test_df["stock_code"].value_counts().index[0]
sample = test_df[test_df["stock_code"] == sample_code].sort_values("date")
r = sample["future_return_1w"].values
fac = np.ones(len(r))
for j in range(1, len(r)):
    x = r[j - 1]
    fac[j] = fac[j - 1] * (1.0 if np.isnan(x) else (1.0 + x))
synth_close = 100.0 * fac
print(f"样本股票: {sample_code}")
print(f"合成价格: 起始={synth_close[0]:.2f}, 终止={synth_close[-1]:.2f}")
print(f"合成价格变化: {(synth_close[-1]/synth_close[0]-1)*100:.2f}%")
print(f"future_return_1w累计: {(np.nancumprod(1+r)-1)[-1]*100:.2f}%")
print()

# 4. 关键问题：回测引擎在t时刻用close[t]买入，t+1时刻用close[t+1]估值
# close[t] = 100 * prod(1 + future_return_1w[0:t-1])
# close[t+1] = close[t] * (1 + future_return_1w[t])
# 所以持有收益 = close[t+1]/close[t] - 1 = future_return_1w[t]
# 这是正确的！
print("=== 4. 结论 ===")
print("合成价格逻辑：close[t+1] = close[t] * (1 + future_return_1w[t])")
print("持有收益 = future_return_1w[t]，这是未来一周收益率")
print("在t时刻买入，t+1时刻的收益就是future_return_1w[t]")
print("如果模型选的Top20的future_return_1w均值为负，回测必然为负")
