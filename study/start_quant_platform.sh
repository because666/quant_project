#!/usr/bin/env bash
set -e

echo "===== 量化学习平台启动（Linux/macOS） ====="

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

echo "安装 Python 依赖..."
pip install -r requirements.txt

echo "启动 FastAPI 后端..."
uvicorn quant_platform.api.server:create_app --factory --host 0.0.0.0 --port 8000 &

echo "启动 React 前端（需要 Node.js）..."
cd frontend
npm install
npm run dev


