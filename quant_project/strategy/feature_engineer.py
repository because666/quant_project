# strategy/feature_engineer.py
import pandas as pd
import numpy as np


def calculate_features(clean_data_path):
    """
    从清洗后的数据计算量化因子，生成模型训练用的特征数据
    :param clean_data_path: 清洗后数据的路径（如data/cleaned_data/000001_clean.csv）
    :return: 包含因子和涨跌标签的DataFrame
    """
    # 1. 读取清洗后的股票数据
    df = pd.read_csv(clean_data_path, encoding="utf-8-sig")

    # 数据预处理：确保日期列格式正确，按时间排序
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期").reset_index(drop=True)

    # 2. 计算3个核心因子
    # 因子1：5日收益率 = (今日收盘价 - 5日前收盘价) / 5日前收盘价
    df["5日收益率"] = df["收盘"].pct_change(periods=5)

    # 因子2：收盘价/5日均线 = 收盘价 / 5日移动平均收盘价
    df["5日均线"] = df["收盘"].rolling(window=5).mean()
    df["收盘价_5日均线比率"] = df["收盘"] / df["5日均线"]

    # 因子3：成交量变化率 = (今日成交量 - 5日前成交量) / 5日前成交量
    df["成交量变化率"] = df["成交量"].pct_change(periods=5)

    # 3. 生成训练标签：明日涨(1)、跌(0)
    df["明日涨跌标签"] = np.where(df["收盘"].shift(-1) > df["收盘"], 1, 0)

    # 4. 去除空值（滚动窗口和标签产生的NaN）
    df = df.dropna(subset=["5日收益率", "收盘价_5日均线比率", "成交量变化率", "明日涨跌标签"])

    print(f"因子计算完成，有效数据{len(df)}行")
    return df


def save_feature_data(df, save_path="strategy/feature_data.csv"):
    """保存因子数据供模型训练使用"""
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"因子数据已保存至：{save_path}")


if __name__ == "__main__":
    # 测试：对接清洗后的数据计算因子
    feature_df = calculate_features("../data/cleaned_data/000001_clean.csv")
    save_feature_data(feature_df)
    print("\n因子数据预览：")
    print(feature_df[["日期", "收盘", "5日收益率", "收盘价_5日均线比率", "成交量变化率", "明日涨跌标签"]].head())