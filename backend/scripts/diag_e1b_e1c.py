import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from src.predictor import ModelPredictor
from src.backtest import BacktestEngine
from pathlib import Path

eng = BacktestEngine('lightgbm', top_n=20, vol_penalty=1.0)
weekly_df = eng.load_weekly_data()

pred_e1b = ModelPredictor('lightgbm', model_path=Path('models/e1b_lightgbm.pkl'))
pred_e1c = ModelPredictor('lightgbm', model_path=Path('models/e1c_lightgbm.pkl'))

test_df = weekly_df[weekly_df['date'] >= '2023-01-01'].copy()
test_dates = sorted(test_df['date'].unique())

diff_count = 0
same_count = 0
for d in test_dates[:10]:
    xs = test_df[test_df['date'] == d]
    pred_b = pred_e1b.predict(xs)
    pred_c = pred_e1c.predict(xs)
    top_e1b = set(pred_b['stock_code'].values[:20])
    top_e1c = set(pred_c['stock_code'].values[:20])
    same = top_e1b == top_e1c
    overlap = len(top_e1b & top_e1c)
    if same:
        same_count += 1
    else:
        diff_count += 1
    print(f'{d}: overlap={overlap}/20, same={same}')

print(f'\n前10周: same={same_count}, diff={diff_count}')

xs = test_df[test_df['date'] == test_dates[0]]
pred_b = pred_e1b.predict(xs)
pred_c = pred_e1c.predict(xs)
scores_b = pred_b['score'].values
scores_c = pred_c['score'].values
print(f'E1b scores range: [{scores_b.min():.6f}, {scores_b.max():.6f}]')
print(f'E1c scores range: [{scores_c.min():.6f}, {scores_c.max():.6f}]')

pred_b_sorted = pred_b.sort_values('stock_code')
pred_c_sorted = pred_c.sort_values('stock_code')
common = set(pred_b_sorted['stock_code']) & set(pred_c_sorted['stock_code'])
pred_b_common = pred_b_sorted[pred_b_sorted['stock_code'].isin(common)].sort_values('stock_code')
pred_c_common = pred_c_sorted[pred_c_sorted['stock_code'].isin(common)].sort_values('stock_code')
diff = np.abs(pred_b_common['score'].values - pred_c_common['score'].values)
print(f'Score diff max: {diff.max():.8f}')
print(f'Score diff mean: {diff.mean():.8f}')
