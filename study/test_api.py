import requests
import json

# 测试API服务器是否正常运行
def test_api():
    print("测试API服务器...")
    
    # 测试数据获取
    print("1. 测试数据获取...")
    fetch_url = "http://localhost:8000/data/fetch"
    fetch_data = {
        "symbols": ["000001"],
        "start": "2020-01-01",
        "end": "2023-12-31"
    }
    
    try:
        response = requests.post(fetch_url, json=fetch_data)
        print(f"   数据获取状态码: {response.status_code}")
        print(f"   数据获取响应: {response.json()}")
    except Exception as e:
        print(f"   数据获取失败: {e}")
    
    # 测试模型训练
    print("\n2. 测试模型训练...")
    train_url = "http://localhost:8000/model/train"
    train_data = {
        "symbol": "000001",
        "train_start": "2020-01-01",
        "train_end": "2023-12-31",
        "label_horizon": 1,
        "model_type": "random_forest"
    }
    
    try:
        response = requests.post(train_url, json=train_data)
        print(f"   模型训练状态码: {response.status_code}")
        print(f"   模型训练响应: {response.json()}")
        
        # 测试LSTM模型
        print("\n3. 测试LSTM模型训练...")
        train_data["model_type"] = "lstm"
        response = requests.post(train_url, json=train_data)
        print(f"   LSTM模型训练状态码: {response.status_code}")
        print(f"   LSTM模型训练响应: {response.json()}")
        
        return True
    except Exception as e:
        print(f"   模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_api()