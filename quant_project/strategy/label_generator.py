import pandas as pd
import argparse

def calculate_future_return(data: pd.DataFrame, future_days: int = 5) -> pd.DataFrame:
    """计算未来N日收益率"""
    # 未来N日收盘价相对于当前收盘价的收益率
    data["future_close"] = data["close"].shift(-future_days)  # 未来N日收盘价
    data["future_return"] = (data["future_close"] - data["close"]) / data["close"] * 100  # 收益率（%）
    return data

def generate_label(data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """生成标签：未来N日收益率是否大于threshold（默认3%）"""
    # 标签：1=上涨（收益率≥阈值），0=不上涨（收益率<阈值）
    data["label"] = (data["future_return"] >= threshold).astype(int)
    # 删除包含NaN的行（最后N行无未来数据）
    data_clean = data.dropna(subset=["future_return", "label"])
    return data_clean

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于股票数据生成预测标签（未来N日收益率是否≥阈值）")
    parser.add_argument("--input", required=True, help="输入的清洗后数据路径（cleaned_data.csv）")
    parser.add_argument("--output", default="labeled_data.csv", help="输出带标签数据的路径")
    parser.add_argument("--days", type=int, default=5, help="预测未来天数（默认5日）")
    parser.add_argument("--threshold", type=float, default=3.0, help="收益率阈值（默认3%）")
    args = parser.parse_args()

    # 执行标签生成流程
    print(f"开始生成标签：未来{args.days}日收益率≥{args.threshold}%为上涨标签")
    cleaned_data = pd.read_csv(args.input, parse_dates=["date"], index_col="date")
    data_with_return = calculate_future_return(cleaned_data, args.days)
    labeled_data = generate_label(data_with_return, args.threshold)

    # 保存结果并输出统计信息
    labeled_data.to_csv(args.output)
    print(f"标签数据已保存至：{args.output}")
    print(f"\n标签分布统计：")
    label_count = labeled_data["label"].value_counts()
    print(f"上涨标签（1）：{label_count.get(1, 0)} 条（{label_count.get(1, 0)/len(labeled_data)*100:.1f}%）")
    print(f"非上涨标签（0）：{label_count.get(0, 0)} 条（{label_count.get(0, 0)/len(labeled_data)*100:.1f}%）")
