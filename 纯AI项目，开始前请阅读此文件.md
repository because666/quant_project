# 📁 项目文件功能详解与分工手册

## 一、 项目核心架构图

```
数据流向：config/ → data_module/ → strategy_module/ → backtest_module/ → main_app/ & frontend/
```

## 二、 配置文件 (`config/`) - 项目的“大脑与规则库”

| 文件 | 负责人 | 核心功能（具体要做什么） | 必须包含的内容示例 | 验收标准 |
| :--- | :--- | :--- | :--- | :--- |
| **`project_config.py`** | 串联组-组员A | **定义全项目统一的参数**，所有其他文件都必须从这里导入设置，严禁写死。 | ```python<br>STOCK_LIST = ['000001', '000002']<br>START_DATE = '2023-01-01'<br>END_DATE = '2023-12-31'<br>INITIAL_CAPITAL = 100000<br>DATA_PATH = './data_module/outputs/'<br>``` | 1. 其他组员能成功 `import config.project_config`。<br>2. 运行时不报错。 |
| **`strategy_params.yaml`** | 策略组-组员B | **定义策略调参的模板**，供前端界面修改和策略引擎读取。 | ```yaml<br>model_name: "RandomForest"<br>stop_loss: 0.05<br>take_profit: 0.10<br>factors: [‘factor_1’, ‘factor_2’]<br>``` | 1. 是一个合法的YAML文件。<br>2. 策略组能成功读取其中的参数。 |
| **`api_config.yaml`** | 数据组-组员A | **安全存放所有外部数据源的密钥**（如Tushare token）。 | ```yaml<br>tushare:<br>  token: ‘你的token’<br>akshare:<br>  default: true<br>``` | 1. 在 `.gitignore` 中忽略此文件，防止密钥上传。<br>2. 数据获取代码能读取它。 |

## 三、 数据模块 (`data_module/`) - 项目的“原料加工厂”

| 文件 | 负责人 | 核心功能（具体要做什么） | 输入 | 输出 | 验收标准（完成标志） |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`data_fetcher.py`** | 数据组-组员A | **从网络获取原始股票数据**。包含一个主函数，能读取`config`中的股票列表和日期，调用AKShare/Tushare API，循环获取每只股票的日K线数据。 | `config`中的股票列表、起止日期。 | 1. 一个DataFrame。<br>2. 文件：`outputs/raw_data.parquet` | 运行脚本后，能在`outputs/`文件夹下看到 `raw_data.parquet` 文件，且文件大小 > 1KB。 |
| **`data_cleaner.py`** | 数据组-组员B | **清洗和标准化数据**。读取上一步的`raw_data.parquet`，做三件事：1. 处理缺失值（直接删除）。2. 确保`date`列是日期格式。3. 按股票和日期排序。 | `raw_data.parquet` | 1. 一个干净的DataFrame。<br>2. 文件：`outputs/standard_data.parquet` | 能成功读取`raw_data.parquet`，并生成 `standard_data.parquet`。用pandas打开新文件，查看前5行，数据整齐。 |
| **`factor_miner.py`** | 数据组-组员B | **计算基础特征（因子）**。读取`standard_data.parquet`，新增几列，例如：<br>1. `return_5d`: 5日收益率。<br>2. `volume_ma_ratio`: 成交量/5日平均成交量。 | `standard_data.parquet` | 1. 增加了因子列的DataFrame（覆盖原文件或保存为新文件）。<br>2. **必须告知策略组新增的列名清单**。 | 输出的数据文件包含了新的因子列。能打印出新增的列名列表。 |

**🔗 数据模块协作铁律**：
组员A必须等组员B确认能成功读取 `raw_data.parquet` 后，才能变更文件格式。组员B在完成因子计算后，**必须**在群里发布：“数据模块因子就绪，新增列名如下：[‘return_5d’, ‘volume_ma_ratio’]”。

## 四、 策略模块 (`strategy_module/`) - 项目的“智能决策中心”

| 文件 | 负责人 | 核心功能（具体要做什么） | 输入 | 输出 | 验收标准（完成标志） |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`model_trainer.py`** | 策略组-组员A | **训练机器学习模型**。1. 读取数据组提供的含因子数据。2. 定义特征(X)和目标(y)。例如，用当天所有因子预测**未来5天是否上涨**。3. 使用随机森林训练模型。4. 保存模型。 | `standard_data.parquet` (含因子列) | 1. 一个训练好的模型对象。<br>2. 文件：`outputs/trained_model.pkl` | 能运行完成并生成 `.pkl` 模型文件。能在新脚本中加载该模型。 |
| **`factor_optimizer.py`** | 策略组-组员A | **评估因子好坏**。计算每个因子与未来收益的相关性，或输出模型的特征重要性，形成报告。 | `standard_data.parquet`，训练好的模型。 | 因子重要性排名（打印出来或存为文件）。 | 运行后能在终端或日志中看到因子的排名。 |
| **`strategy_engine.py`** | 策略组-组员B | **生成交易信号**。1. 加载训练好的模型(`.pkl`)。2. 读取最新的因子数据。3. 让模型预测，得到每只股票每天的信号：1(买入)、-1(卖出)、0(持有)。 | 1. `trained_model.pkl`<br>2. 最新的因子数据。 | 1. 一个包含`date, symbol, signal`三列的DataFrame。<br>2. 文件：`outputs/signal.csv` | 生成 `signal.csv` 文件。用Excel打开，能看到明确的买卖信号。 |

**🔗 策略模块协作铁律**：
组员A在保存模型后，必须立即告知组员B模型的完整路径。组员B生成`signal.csv`后，**必须**在群里发布：“策略信号已生成，路径为 `strategy_module/outputs/signal.csv`，列格式为 `date, symbol, signal`”。

## 五、 回测模块 (`backtest_module/`) - 项目的“实战模拟与验算中心”

| 文件 | 负责人 | 核心功能（具体要做什么） | 输入 | 输出 | 验收标准（完成标志） |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`backtest_engine.py`** | 回测组-组员A | **模拟交易，计算每日资产**。1. 读取策略信号(`signal.csv`)和历史价格(`standard_data.parquet`)。2. 初始化虚拟资金。3. 按日期循环，根据信号模拟买卖（以收盘价成交）。4. 记录每天的总资产。 | 1. `signal.csv`<br>2. `standard_data.parquet`<br>3. 初始资金(`INITIAL_CAPITAL`)。 | 一个DataFrame，包含`date`和`portfolio_value`两列，代表每日资产净值。 | 运行后，能打印出一条资产曲线的前后几个值（例如，第一天100000，最后一天110000）。 |
| **`performance_analyzer.py`** | 回测组-组员B | **计算核心绩效指标**。根据上述资产曲线计算：<br>1. 总收益率。<br>2. 最大回撤（最惨的时候亏了百分之多少）。<br>3. 胜率（盈利交易次数占比）。 | 每日资产净值DataFrame。 | 一个字典，格式如：`{‘总收益’: 0.10, ‘最大回撤’: -0.05, ‘胜率’: 0.55}`。 | 运行后，能在终端清晰打印出这三个指标的数字。 |
| **`report_generator.py`** | 回测组-组员B | **生成文字建议报告**。根据绩效指标，写一段简单的文字建议，例如：“策略累计收益10%，最大回撤5%，表现稳健。建议未来一个月可继续运行该策略，但需关注市场波动。” | 绩效指标字典。 | 一个文本文件 `report.txt`，包含上述建议。 | 生成 `report.txt` 文件，内容是可读的文字建议。 |

**🔗 回测模块协作铁律**：
组员A在生成资产数据后，必须通过一个规范的函数或变量将其暴露给组员B。组员B在计算绩效和生成报告时，**严禁**手动输入数字，必须全部从组员A的输出自动计算。

## 六、 串联与前端模块 (`main_app/` & `frontend/`) - 项目的“总控台与界面”

| 文件 | 负责人 | 核心功能（具体要做什么） | 输入 | 输出 | 验收标准（完成标志） |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`main.py`** | 串联组-组员A | **程序唯一入口**。只有几行代码，用于启动整个程序（例如启动前端界面或主流程）。 | 无。 | 启动应用。 | 运行 `python main.py` 可以启动程序，不报错。 |
| **`main_pipeline.py`** | 串联组-组员A | **主流程控制器**。定义一个 `run_all()` 函数，按顺序“打印”出如下步骤：1. 获取数据，2. 训练模型，3. 生成信号，4. 运行回测，5. 生成报告。**本周目标：仅用print语句串联。** | 无。 | 控制台按顺序打印的步骤日志。 | 运行 `run_all()` 函数，能在控制台看到清晰的、顺序正确的步骤提示。 |
| **`data_interface.py`** | 串联组-组员A | **提供统一的数据读取接口**。编写如 `get_clean_data()` 函数，内部正确读取 `standard_data.parquet` 并返回。让其他组不用关心复杂路径。 | 无。 | 返回给其他模块的DataFrame。 | 其他组员（如策略组）能通过调用你的函数轻松拿到数据，而不需要自己写路径。 |
| **`risk_manager.py`** | 串联组-组员B | **执行风控规则**。例如，监控沪深300指数，若其收盘价低于60日均线，则返回一个“清仓”信号。 | 指数历史数据（可从网络临时获取）。 | 风控信号：`True`（正常）或`False`（触发风控）。 | 能返回一个布尔值。 |
| **`frontend/`下所有文件** | 串联组-组员B | **搭建图形界面**。使用Streamlit库，创建一个Web界面。**第一版只需实现**：<br>1. 一个标题。<br>2. 一个“开始回测”按钮。<br>3. 按钮按下后，调用 `main_pipeline.run_all()`。 | 用户点击按钮。 | 一个本地运行的网页。 | 运行 `streamlit run main_window.py` 后，能打开一个网页，点击按钮后控制台有反应。 |

**🔗 串联模块协作铁律**：
组员B在开发界面时，按钮绑定的函数名必须与组员A在 `main_pipeline.py` 中定义的函数名完全一致。两人应共同确定这个函数名。

## 七、 给全组的终极行动建议与鼓励

1.  **从“打印版”开始**：每个文件的第一版目标，不是实现完美逻辑，而是**实现正确的“输入-输出”对接**。例如，`data_fetcher.py` 即使只获取一只股票、一天的数据，只要它能生成约定格式的 `raw_data.parquet`，就是伟大的胜利！
2.  **拥抱“丑陋但可运行”的代码**：AI生成的代码可能冗长，但只要它能跑通并完成接口任务，就是好代码。**整洁和优化是之后的事**。
3.  **验收不是感觉，是事实**：你的工作是否完成，**不由你感觉而定，而由“约定的输出文件是否存在且可用”来判定**。请用这个标准要求自己和搭档。
4.  **我们是一个整体**：你手中的文件，就像接力棒。只有你清晰地交给下一个人，比赛才能继续。**主动交棒，主动确认**，是最高效的合作。




---

# 🧠 零基础AI编码提示词规范（必读！）

## 一、 核心：用好AI的“万能公式”

把这五句话**复制**到你的AI对话框（如ChatGPT/DeepSeek/文心一言），然后按你的任务修改【】里的内容。

```
【第一句：定角色】
请你扮演一位经验丰富的Python量化工程师，正在严格遵循一个已有架构的项目进行开发。

【第二句：给上下文】
项目根目录是 `quant_ml_project/`。我的任务是完成其中的一个特定文件，文件完整路径是：【请填写你的文件路径，例如：quant_ml_project/data_module/data_fetcher.py】。

【第三句：说要求】
请为这个文件编写完整、可运行的Python代码，实现以下**具体功能**：
1. 功能点1：【必须非常具体，例如：定义一个名为 `fetch_stock_data` 的函数，接受 `symbol_list`, `start_date`, `end_date` 三个参数】。
2. 功能点2：【例如：使用AKShare库，获取这些股票在指定日期的日线数据（包括open, close, high, low, volume）】。
3. 功能点3：【例如：将获取的所有数据合并，保存到项目内的 `data_module/outputs/raw_data.parquet` 文件】。
请使用pandas和numpy库，代码需添加清晰的中文注释。

【第四句：锁接口（最重要！）】
请务必严格遵守以下项目约定，否则代码将无法与队友对接：
- **输入**：函数参数的形式必须为【例如：(symbol_list: list, start_date: str, end_date: str)】。
- **输出**：函数的返回值必须是【例如：一个pandas DataFrame，同时将数据保存到指定路径】。
- **文件与路径**：生成的数据/模型文件**必须**保存到【例如：`data_module/outputs/raw_data.parquet`】，不能是其他路径。
- **列名规范**：数据表的列名**必须**为：【例如：['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']】。
- **配置读取**：所有参数（如股票列表、开始日期）**必须**从 `quant_ml_project/config/project_config.py` 中导入，不要写死在代码里。

【第五句：避坑指南】
编写代码时请注意：
1. 添加必要的异常处理（如网络请求失败）。
2. 避免使用绝对路径（如C:/Users/...），全部使用相对于项目根目录的路径。
3. 打印一些进度信息（如“正在获取XXX股票数据”）。
```

---

## 二、 针对每个文件的具体提示词示例（直接复制使用）

### 📊 **数据模块**
**文件：`data_fetcher.py` （组员A）**
```
（使用上面的万能公式，【】内内容替换如下）
【文件路径】：quant_ml_project/data_module/data_fetcher.py
【功能点1】：定义一个名为 `fetch_stock_data` 的函数。
【功能点2】：函数从 `config/project_config.py` 导入 `STOCK_LIST`, `START_DATE`, `END_DATE` 作为默认参数。
【功能点3】：使用AKShare库，循环获取`STOCK_LIST`中每只股票从`START_DATE`到`END_DATE`的日线数据。
【功能点4】：将所有数据合并，并保存到 `data_module/outputs/raw_data.parquet`。
【输入】：函数应允许覆盖默认参数，即 `def fetch_stock_data(symbol_list=STOCK_LIST, start_date=START_DATE, end_date=END_DATE):`
【输出】：返回合并后的DataFrame，并保存文件。
【列名】：['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
```

**文件：`data_cleaner.py` （组员B）**
```
【文件路径】：quant_ml_project/data_module/data_cleaner.py
【功能点1】：定义一个名为 `clean_raw_data` 的函数。
【功能点2】：函数读取 `data_module/outputs/raw_data.parquet` 文件。
【功能点3】：进行数据清洗：检查缺失值并删除、将`date`列转换为日期格式、按`symbol`和`date`排序。
【功能点4】：将清洗后的数据保存到 `data_module/outputs/standard_data.parquet`。
【输入】：无需参数。
【输出】：返回清洗后的DataFrame，并保存文件。
【列名】：保持与输入文件一致。
```

### 🤖 **策略模块**
**文件：`model_trainer.py` （组员A）**
```
【文件路径】：quant_ml_project/strategy_module/model_trainer.py
【功能点1】：定义一个名为 `train_model` 的函数。
【功能点2】：从 `data_module/outputs/standard_data.parquet` 读取数据。
【功能点3】：准备特征(X)和目标(y)：使用除`date`, `symbol`, `close`外的列作为特征，以`close`列未来5日的涨跌（1涨0跌）作为目标。
【功能点4】：使用`sklearn`的`RandomForestClassifier`训练一个分类模型。
【功能点5】：将训练好的模型保存到 `strategy_module/outputs/trained_model.pkl`。
【输入】：`train_test_split`的比例等参数可从`config/strategy_params.yaml`读取。
【输出】：返回训练好的模型对象，并保存模型文件。
```

### 📈 **回测模块**
**文件：`backtest_engine.py` （组员A）**
```
【文件路径】：quant_ml_project/backtest_module/backtest_engine.py
【功能点1】：定义一个名为 `run_backtest` 的类或函数。
【功能点2】：从 `strategy_module/outputs/signal.csv` 读取交易信号，从`data_module/outputs/standard_data.parquet`读取价格数据。
【功能点3】：初始化一个虚拟账户，资金从`config/project_config.py`的`INITIAL_CAPITAL`读取。
【功能点4】：按日期循环，根据当天的信号执行买入/卖出操作（简化：忽略手续费，以收盘价成交）。
【功能点5】：记录每天的账户总资产。
【输出】：返回一个包含日期和资产净值的DataFrame。
```

---

## 三、 提交工作成果前的“三步自检法”

生成代码后，**不要直接说“我完成了”**，请按此流程检查：

1.  **第一步：独立运行**
    ```bash
    cd quant_ml_project  # 首先进入项目文件夹
    python 你的文件.py    # 运行你的脚本
    ```
    **成功标志**：没有红色报错，并打印出“数据已保存到XXX”或类似提示。

2.  **第二步：验证输出**
    去文件管理器里找到你的代码承诺要生成的文件（如`raw_data.parquet`），**确认它确实存在**，并且不是空文件（文件大小>1KB）。

3.  **第三步：在群里按格式提交**
    > 【数据组-张三】文件完成并已验证
    > 文件：`data_fetcher.py`
    > 功能：已实现从AKShare获取数据，并读取config配置。
    > 输出文件：`data_module/outputs/raw_data.parquet` （文件大小：XX KB）
    > **接口就绪**：我的搭档@李四，你可以用你的`data_cleaner.py`读取我这个文件进行测试了。

## 四、 最重要的两条“军规”

1.  **路径是生命线**：所有文件路径**必须**和分工表里写的一模一样。如果你改了，你的搭档就会失败。
2.  **先跑通，再优化**：第一个目标是生成一个**能跑起来、能生成约定文件**的“简陋”版本。不要追求完美，先保证“有”。


---





