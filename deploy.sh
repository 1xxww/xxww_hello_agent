#!/usr/bin/env bash
set -e

echo "🚀 开始自动化部署 Travel Assistant..."

# 1. 系统包及环境检测与安装
if [ -f /etc/debian_version ]; then
    echo "📦 检测到 Debian/Ubuntu 系统，更新包管理器..."
    apt-get update -y
    apt-get install -y python3 python3-pip python3-venv nginx curl
elif [ -f /etc/redhat-release ]; then
    echo "📦 检测到 CentOS/RHEL 系统..."
    yum install -y epel-release || true
    yum install -y python3 python3-pip nginx curl
fi

# 2. 安装 Node.js 20.x
if ! command -v node &> /dev/null; then
    echo "📦 安装 Node.js 20..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs || yum install -y nodejs
fi

# 3. 部署 Python 后端虚拟环境
echo "🐍 配置 Python 后端环境..."
PROJECT_DIR="/opt/HelloAgent"
cd "$PROJECT_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn httpx python-dotenv openai mcp pydantic requests

# 4. 构建前端 React 应用
echo "⚛️ 构建前端应用..."
cd "$PROJECT_DIR/travelAssistant/frontend"
npm install
npm run build

# 5. 配置 Systemd 后端守护进程
echo "⚙️ 配置并启动后端服务..."
cat << 'SERVICE_EOF' > /etc/systemd/system/travel-backend.service
[Unit]
Description=Travel Assistant FastAPI Backend
After=network.target

[Service]
User=root
WorkingDirectory=/opt/HelloAgent/travelAssistant/backend
ExecStart=/opt/HelloAgent/venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5
EnvironmentFile=/opt/HelloAgent/.env

[Install]
WantedBy=multi-user.target
SERVICE_EOF

systemctl daemon-reload
systemctl enable travel-backend
systemctl restart travel-backend

# 6. 配置 Nginx 网页服务
echo "🌐 配置 Nginx 网页服务..."
cat << 'NGINX_EOF' > /etc/nginx/conf.d/travel.conf
server {
    listen 80;
    server_name _;

    location / {
        root /opt/HelloAgent/travelAssistant/frontend/dist;
        index index.html index.htm;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
    }

    location /_AMapService/ {
        proxy_pass http://127.0.0.1:8000/_AMapService/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
NGINX_EOF

# 清理默认配置避免端口冲突并重启 Nginx
rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true
nginx -t
systemctl restart nginx

echo "🎉 部署完成！请在浏览器直接访问：http://114.55.62.246/"
