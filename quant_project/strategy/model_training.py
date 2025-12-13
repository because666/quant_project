# strategy/model_training.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_model(feature_path="strategy/feature_data.csv", model_save_path="strategy/model.pkl"):
    """
    训练涨跌预测模型，保存模型文件
    :param feature_path: 因子数据路径
    :param model_save_path: 模型保存路径
    :return: 训练好的模型
    """
    # 1. 读取因子数据
    df = pd.read_csv(feature_path, encoding="utf-8-sig")

    # 2. 定义特征列和标签列
    feature_cols = ["5日收益率", "收盘价_5日均线比率", "成交量变化率"]
    X = df[feature_cols]  # 特征矩阵
    y = df["明日涨跌标签"]  # 标签（1涨0跌）

    # 3. 划分训练集/测试集（时间序列不打乱）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # 4. 训练随机森林分类器
    model = RandomForestClassifier(
        n_estimators=100,  # 决策树数量
        max_depth=6,  # 树的最大深度（防止过拟合）
        random_state=42,
        n_jobs=-1  # 并行计算加速
    )
    model.fit(X_train, y_train)

    # 5. 模型评估
    y_pred = model.predict(X_test)
    print("模型测试集准确率：", round(accuracy_score(y_test, y_pred), 4))
    print("\n分类报告：")
    print(classification_report(y_test, y_pred))

    # 6. 保存模型
    joblib.dump(model, model_save_path)
    print(f"\n模型已保存至：{model_save_path}")
    return model


def generate_signal(model_path="strategy/model.pkl", feature_path="strategy/feature_data.csv",
                    signal_save_path="signal.csv"):
    """
    生成交易信号文件（供回测模块使用）
    :param model_path: 训练好的模型路径
    :param feature_path: 因子数据路径
    :param signal_save_path: 信号文件保存路径
    """
    # 1. 加载模型和因子数据
    model = joblib.load(model_path)
    df = pd.read_csv(feature_path, encoding="utf-8-sig")
    feature_cols = ["5日收益率", "收盘价_5日均线比率", "成交量变化率"]

    # 2. 预测涨跌信号：1=买入（预测涨），0=卖出（预测跌）
    df["交易信号"] = model.predict(df[feature_cols])

    # 3. 保留核心列生成信号文件（日期+收盘价+交易信号）
    signal_df = df[["日期", "收盘", "交易信号"]]
    signal_df.to_csv(signal_save_path, index=False, encoding="utf-8-sig")
    print(f"交易信号文件已生成：{signal_save_path}")
    print("\n信号数据预览：")
    print(signal_df.tail(10))


if __name__ == "__main__":
    # 训练模型
    model = train_model()
    # 生成交易信号
    generate_signal()