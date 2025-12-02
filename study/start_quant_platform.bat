@echo off
echo ===== 量化学习平台启动（Windows） =====

REM 创建并激活虚拟环境（如不存在）
if not exist ".venv" (
  python -m venv .venv
)
call .venv\Scripts\activate.bat

echo 安装 Python 依赖...
pip install -r requirements.txt

echo 启动 FastAPI 后端...
start "" cmd /c ".venv\Scripts\activate.bat && uvicorn quant_platform.api.server:create_app --factory --host 0.0.0.0 --port 8000"

echo 启动 React 前端（需要已安装 Node.js 与 pnpm 或 npm）...
cd frontend
if exist "package-lock.json" (
  npm install
) else (
  npm install
)
npm run dev


