import pandas as pd
import os

def clean_single_file(input_file_path, output_file_path):
    """
    清洗单个股票数据文件。

    参数：
    input_file_path (str): 原始数据文件的路径，例如 ‘data/raw_data/000001_raw.csv’
    output_file_path (str): 清洗后数据保存的路径，例如 ‘data/cleaned_data/000001_clean.csv’

    返回：
    pandas.DataFrame: 清洗后的DataFrame，同时也会保存为CSV文件。
    """
    try:
        # 1. 读取原始CSV文件
        # 假设原始数据列名是中文，这里指定编码格式。如果列名是英文，可以去掉`names`参数。
        df = pd.read_csv(input_file_path, encoding='utf-8-sig')
        # 如果akshare默认保存的列名是中文，可以参考以下方式指定（根据实际文件调整）：
        # df = pd.read_csv(input_file_path, encoding='utf-8-sig')
        print(f"成功读取文件: {input_file_path}， 原始数据形状: {df.shape}")

        # 2. 删除缺失行 (任何一列为NaN的行都会被删除)
        df_cleaned = df.dropna()
        print(f"删除缺失值后数据形状: {df_cleaned.shape}")

        # 3. 确保日期格式正确
        # 假设你的日期列名是 ‘日期’ 或 ‘date’，请根据数据A生成文件的实际列名修改。
        date_column = '日期'  # 常见列名可能是 ‘date’， ‘datetime’， 请按实际情况修改！
        # 如果列名不对，可以打印 df.columns 查看
        if date_column not in df_cleaned.columns:
            print(f"警告：未找到列名为 ‘{date_column}’ 的列。")
            print(f"现有列名为: {list(df_cleaned.columns)}")
            # 尝试自动寻找包含‘日期’或‘date’的列
            for col in df_cleaned.columns:
                if '日期' in col or 'date' in col.lower():
                    date_column = col
                    print(f"自动识别日期列为: {date_column}")
                    break

        # 将日期列转换为 pandas 的 datetime 格式
        df_cleaned[date_column] = pd.to_datetime(df_cleaned[date_column])
        print(f"日期列 ‘{date_column}’ 格式转换完成。")

        # 4. 按日期排序 (通常是从过去到现在)
        df_cleaned = df_cleaned.sort_values(by=date_column, ascending=True)
        print("数据已按日期排序。")

        # 5. 重置索引（排序后索引会乱，这一步让索引从0开始顺序排列，非必须但建议）
        df_cleaned = df_cleaned.reset_index(drop=True)

        # 6. 保存清洗后的数据到新的CSV文件
        # 确保输出文件夹存在
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        df_cleaned.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"清洗后的数据已保存至: {output_file_path}")

        return df_cleaned

    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file_path}，请检查路径是否正确。")
    except Exception as e:
        print(f"清洗过程中发生未知错误: {e}")

# 以下是测试代码，当你直接运行此脚本时会执行
if __name__ == "__main__":
    # 定义测试路径
    # 请确保 data/raw_data/000001_raw.csv 文件已由数据A生成并存在！
    raw_file = "data/raw_data/000001_raw.csv"
    clean_file = "data/cleaned_data/000001_clean.csv"

    # 调用清洗函数
    cleaned_df = clean_single_file(raw_file, clean_file)

    if cleaned_df is not None:
        # 打印清洗后的前几行数据，用于验证
        print("\n清洗后数据预览:")
        print(cleaned_df.head())
        print(f"\n数据日期范围: {cleaned_df['日期'].min()} 到 {cleaned_df['日期'].max()}")
