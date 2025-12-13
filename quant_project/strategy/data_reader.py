import pandas as pd
import argparse

def read_stock_data(file_path: str) -> pd.DataFrame:
    """读取股票CSV数据，返回DataFrame"""
    try:
        data = pd.read_csv(file_path, parse_dates=["date"], index_col="date")  # 假设日期列名为date
        return data
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}")
        raise
    except Exception as e:
        print(f"读取数据失败：{str(e)}")
        raise

def show_basic_info(data: pd.DataFrame, stock_name: str = "未知股票") -> None:
    """显示数据基本信息"""
    print("=" * 50)
    print(f"【{stock_name}】数据基本信息")
    print("=" * 50)
    print(f"数据时间范围：{data.index.min().strftime('%Y-%m-%d')} 至 {data.index.max().strftime('%Y-%m-%d')}")
    print(f"数据总条数：{len(data)} 条")
    print(f"字段列表：{list(data.columns)}")
    print("\n前5行数据：")
    print(data.head())
    print("\n数据统计摘要：")
    print(data.describe().round(2))
    print("=" * 50)

if __name__ == "__main__":
    # 命令行参数解析（支持指定文件路径和股票名称）
    parser = argparse.ArgumentParser(description="读取股票CSV数据并显示基本信息")
    parser.add_argument("--file", required=True, help="CSV文件路径（如：data/stock_600036.csv）")
    parser.add_argument("--name", default="未知股票", help="股票名称（可选）")
    args = parser.parse_args()

    # 执行读取和显示
    stock_data = read_stock_data(args.file)
    show_basic_info(stock_data, args.name)
