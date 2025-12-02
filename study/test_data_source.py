from quant_platform.data.data_source import DataSourceManager, EastMoneyDataSource, AkShareDataSource, TongHuaShunDataSource, SinaDataSource

# 测试数据获取功能
def test_data_source():
    print("测试数据获取功能...")
    
    # 分别测试每个数据源
    sources = [
        ("EastMoney", EastMoneyDataSource()),
        ("AkShare", AkShareDataSource()),
        ("Sina", SinaDataSource()),
        ("TongHuaShun", TongHuaShunDataSource())
    ]
    
    for name, source in sources:
        print(f"\n测试 {name} 数据源:")
        try:
            # 测试获取上证指数数据
            df = source.get_kline("000001", "2025-01-01", "2025-12-01")
            print(f"  成功获取数据，共 {len(df)} 条记录")
            print(f"  数据示例:")
            print(f"  {df.tail(2)}")
        except Exception as e:
            print(f"  失败: {e}")
    
    # 测试数据源管理器
    print("\n测试数据源管理器:")
    manager = DataSourceManager()
    
    try:
        # 测试获取上证指数数据
        df = manager.get_kline("000001", "2025-01-01", "2025-12-01")
        print(f"  成功获取数据，共 {len(df)} 条记录")
        print(f"  使用数据源: {manager.last_source}")
        print(f"  数据示例:")
        print(f"  {df.tail(2)}")
        return True
    except Exception as e:
        print(f"  失败: {e}")
        return False

if __name__ == "__main__":
    test_data_source()
