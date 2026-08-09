#!/bin/bash

# 스크립트가 위치한 디렉토리로 이동하여 경로 문제를 방지합니다.
cd "$(dirname "$0")"

echo "🚀 Github Codespaces / WSL 환경 전체 설치를 시작합니다..."

# 1. 시스템 패키지 업데이트 및 한국어 폰트/로케일 설치
echo "🔄 시스템 패키지 업데이트 및 설치 중..."
sudo apt-get update
sudo apt-get install -y \
    fonts-nanum fonts-nanum-coding fonts-nanum-extra \
    language-pack-ko

# 2. 폰트 캐시 갱신 및 로케일 설정
echo "🔤 폰트 캐시 갱신 및 한국어 로케일 설정 중..."
sudo fc-cache -fv
sudo locale-gen ko_KR.UTF-8
sudo update-locale LANG=ko_KR.UTF-8

# 3. Python 의존성 설치 (requirements.txt가 있을 경우)
if [ -f "requirements.txt" ]; then
    echo "🐍 Python 패키지 설치 중..."
    pip install -r requirements.txt
    
    # 4. Chromium 브라우저 및 시스템 의존성 설치
    echo "🌐 Playwright Chromium 브라우저 설치 중..."
    playwright install chromium
    echo "📦 Chromium 시스템 의존성 설치 중..."
    sudo playwright install-deps chromium
else
    echo "⚠️ requirements.txt 파일을 찾을 수 없어 Python 패키지 설치를 건너뜁니다."
fi

# 5. 실행 권한 부여
echo "🔐 스크립트 실행 권한 부여 중..."
if [ -f "install_hangul.sh" ]; then
    chmod +x install_hangul.sh
fi

# 5.5. 파이썬 가상환경에 프로젝트 루트 경로 등록 및 .env 설정
if [ -f "./setup_path.sh" ]; then
    chmod +x ./setup_path.sh
    ./setup_path.sh
else
    echo "⚠️ setup_path.sh 파일을 찾을 수 없어 경로 설정을 건너뜁니다."
fi

# 6. .env 파일 생성 (.env.example 복사)
if [ ! -f "../.env" ]; then
    cp ../.env.example ../.env
    echo "📄 .env 파일이 생성되었습니다. API 키를 설정해 주세요."
    if [ -f "./setup_path.sh" ]; then
        ./setup_path.sh
    fi
else
    echo "📄 .env 파일이 이미 존재합니다. 덮어쓰지 않습니다."
fi

# 7. 완료 메시지
echo "---------------------------------------------------------"
echo "✅ 전체 환경 설치가 성공적으로 완료되었습니다!"
echo "---------------------------------------------------------"

