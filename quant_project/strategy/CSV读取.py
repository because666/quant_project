# csv_analyzer.py
"""
CSV文件分析器 - 读取CSV文件并显示基本信息
功能:读取CSV文件,显示数据结构、统计信息和数据样本
"""
# 导入必要的库
import pandas as pd # 数据分析和处理
import numpy as np # 数值计算
import os # 用于文件和路径操作,需要检查用户提供的文件路径是否有效
def read_csv_file(file_path):
    """
    读取CSV文件并返回DataFrame对象
    
    参数:
    file_path(str):CSV文件的路径
    
    返回
    pd.DataFrame:包含CSV数据的DataFrame对象
    """
    try:
        df = pd.read_csv(file_path,encoding='utf-8')
        print(f"成功读取文件:{file_path}")
        return df
    except UnicodeDecodeError:
        try:
            df = pd.reaf_csv(file_path,encoding='gbk')
            print(f"使用gbk编码成功读取文件:{file_path}")
            return df
        except Exception as e:
            print(f"编码问题,请检查文件编码格式：{e}")
            return None
    except FileNotFoundError:
        print(f"文件未找到：{file_path}")
        return None
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return None
    
def display_basic_info(df):
    """显示DataFrame的基本信息
    
    参数：
    df (pd.DataFrame):要分析的数据框
    """
    print("\n" + "="*50)
    print("CSV文件基本信息表")
    print("="*50)

    # 显示数据形状：行数和列数
    print(f"数据形状集：{df.shape[0]}行,{df.shape[1]}列")

    # 显示列名
    print(f"\n 列名列表：")
    for i,col in enumerate(df.columns,1):
        print(f"{i}.{col}")
    # 显示数据类型信息
    print(f"\n 数据类型分布")
    dtype_counts = df.dtype.value_counts()
    print(dtype_counts)
    print("\n各列数据类型:")
    for col in df.columns:
        print(f" {col}:{df[col].dtype}")

def display_statistical_info(df):
    """
    显示数值列的统计信息
    
    参数：
    df.(pd.DataFrame):要分析的数据框
    """
    print("\n" + "="*50)
    print("数值列统计信息")
    print("="*50)

    # 选择数值类型的列
    numeric_columns = df.select_dtype(include=[np.number]).columns

    if len(numeric_columns) > 0:
        # 显示描述型统计信息
        print("描述性统计(数列值):")
        print(df.[numeric_columns].describe())

        # 显示每列的基本统计
        print(f"\n各数值列详细信息:")
        for col in numeric_columns:
            print(f"\n🔹{col}:")
            print(f"  非空值：{df.[col].count()}")
            print(f"  唯一值：{df.[col].nunique()}")

            --  




















    
