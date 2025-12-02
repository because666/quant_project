import numpy as np
import pandas as pd
from quant_platform.data.data_source import load_history
from quant_platform.factors.core import default_registry
from quant_platform.models.lstm_model import LSTMModel
from quant_platform.models.base import time_series_split

# 测试模型训练流程
def test_model_training():
    print("测试模型训练流程...")
    
    try:
        # 1. 加载历史数据
        print("1. 加载历史数据...")
        df = load_history("000001", "2020-01-01", "2023-12-31")
        print(f"   成功加载 {len(df)} 条数据")
        
        # 2. 计算因子
        print("2. 计算因子...")
        df = default_registry.apply_all(df)
        print(f"   因子计算完成，数据维度: {df.shape}")
        
        # 3. 准备训练数据
        print("3. 准备训练数据...")
        
        # 选择特征列
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'amount']]
        print(f"   选择特征: {feature_cols}")
        
        # 确保有足够的特征
        if not feature_cols:
            raise ValueError("没有可用的特征列")
        
        # 填充缺失值
        df = df.fillna(0)
        
        # 创建标签：如果下一天收盘价上涨则为1，否则为0
        df['label'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # 移除最后一行，因为它没有标签
        df = df[:-1]
        
        # 准备特征和标签
        X = df[feature_cols].values
        y = df['label'].values
        
        print(f"   特征数据形状: {X.shape}")
        print(f"   标签数据形状: {y.shape}")
        
        # 4. 训练 LSTM 模型
        print("4. 训练 LSTM 模型...")
        
        # 调整数据形状以适应 LSTM 模型：(samples, timesteps, features)
        # 这里我们使用 1 个时间步，即每个样本只包含当前时刻的特征
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        
        # 切分训练集和验证集
        X_train, X_val, y_train, y_val = time_series_split(X, y, train_ratio=0.7)
        
        print(f"   训练集形状: {X_train.shape}, {y_train.shape}")
        print(f"   验证集形状: {X_val.shape}, {y_val.shape}")
        
        # 创建并训练模型
        model = LSTMModel(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=5, batch_size=32)
        
        # 5. 评估模型
        print("5. 评估模型...")
        train_result = model.train(X_train, y_train, X_val, y_val)
        print(f"   训练结果: {train_result.metrics}")
        print(f"   模型保存路径: {train_result.model_path}")
        
        print("\n模型训练测试成功！")
        return True
        
    except Exception as e:
        print(f"\n模型训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_training()