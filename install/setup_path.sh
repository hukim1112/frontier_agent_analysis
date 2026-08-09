#!/bin/bash

# Resolve project root absolute path based on script location (install/..)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🐍 파이썬 가상환경에 프로젝트 루트 경로 등록 중..."
SP_DIR=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)

if [ -n "$SP_DIR" ] && [ -d "$SP_DIR" ]; then
    echo "$PROJ_DIR" > "$SP_DIR/frontier_agent_lab.pth"
    echo "✅ 가상환경 경로에 frontier_agent_lab.pth가 생성되었습니다 ($SP_DIR)"
else
    # Fallback to user site-packages if sys venv finds nothing
    USER_SP_DIR=$(python -m site --user-site 2>/dev/null)
    if [ -n "$USER_SP_DIR" ]; then
        mkdir -p "$USER_SP_DIR"
        echo "$PROJ_DIR" > "$USER_SP_DIR/frontier_agent_lab.pth"
        echo "✅ 사용자 가상환경 경로에 frontier_agent_lab.pth가 생성되었습니다 ($USER_SP_DIR)"
    else
        echo "⚠️ 가상환경 site-packages 경로를 찾을 수 없어 .pth 등록을 건너뜁니다."
    fi
fi

# Update or insert PROJECT_ROOT to .env
ENV_FILE="$PROJ_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    if ! grep -q "PROJECT_ROOT" "$ENV_FILE" 2>/dev/null; then
        echo "PROJECT_ROOT=\"$PROJ_DIR\"" >> "$ENV_FILE"
        echo "🔑 .env에 PROJECT_ROOT 경로가 추가되었습니다."
    else
        # Remove old PROJECT_ROOT entry first and append the new one to support safe updates when repo is moved
        CLEANED_ENV=$(grep -v "^PROJECT_ROOT=" "$ENV_FILE")
        echo "$CLEANED_ENV" > "$ENV_FILE"
        echo "PROJECT_ROOT=\"$PROJ_DIR\"" >> "$ENV_FILE"
        echo "🔄 .env의 PROJECT_ROOT 경로가 현재 경로로 갱신되었습니다."
    fi
else
    echo "⚠️ .env 파일이 존재하지 않아 PROJECT_ROOT 변수 주입을 건너뜁니다."
fi
