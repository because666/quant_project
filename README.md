
# Quant Project - 基于机器学习的量化选股策略

## 📋 项目简介
本项目旨在构建一个基于机器学习的量化选股系统，通过数据获取、模型训练和回测验证，开发有效的股票投资策略。

## 🗂️ 项目结构（具体文件还没有创建完，但是大框架是这样，各个方向负责人依据自己方向需要后续自行更新吧）
-quant_project/
- ├── data/ # 数据获取与清洗组
- │ ├── raw_data/ # 原始数据存储
- │ ├── cleaned_data/ # 清洗后数据存储
- │ ├── data_fetcher.py # 数据获取脚本
- │ └── data_cleaner.py # 数据清洗脚本
- ├── strategy/ # 策略与框架组
- │ ├── model_trainer.py # 模型训练脚本
- │ └── signal_generator.py # 信号生成脚本
- ├── backtest/ # 回测组
- │ └── backtest_engine.py # 回测引擎脚本
- ├── utils/ # 公共工具函数
- ├── config.py # 项目配置文件
- ├── main.py # 主程序入口
- └── requirements.txt # 项目依赖库

## 👥 团队分工
- **数据获取与清洗组**：负责股票数据获取、清洗和预处理
- **策略与框架组**：负责特征工程、模型训练和交易信号生成
- **回测组**：负责策略回测和绩效评估
- **系统集成组**：负责项目整合和流程串联（之前忘记分了，这个方向由组长负责）

## 🚀 快速开始


### 环境配置
```bash
# 克隆项目
git clone https://github.com/你的用户名/quant_project.git
cd quant_project

# 安装依赖
pip install -r requirements.txt
