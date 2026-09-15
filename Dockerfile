# ===============================================================================
# Agent Lab (agent_lab) - GitHub Codespaces & DevContainer Pre-built Base Image
# LangChain, LangGraph, Browser Automation (Playwright/Chromium), noVNC (Port 6080)
# ===============================================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=ko_KR.UTF-8 \
    LC_ALL=ko_KR.UTF-8 \
    DISPLAY=:1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# 1. 필수 시스템 빌드 도구, X11 가상 디스플레이, noVNC, 멀티미디어 및 한글 폰트 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    procps \
    net-tools \
    ffmpeg \
    xvfb \
    x11vnc \
    fluxbox \
    novnc \
    websockify \
    fonts-nanum \
    fonts-nanum-extra \
    fontconfig \
    locales \
    && sed -i '/ko_KR.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen \
    && update-locale LANG=ko_KR.UTF-8 \
    && fc-cache -fv \
    && ln -sf /usr/share/novnc/vnc.html /usr/share/novnc/index.html \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 2. Python 라이브러리 전체 일괄 사전 설치
COPY install/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# 3. Playwright Chromium 브라우저 바이너리 및 시스템 의존성 사전 설치
#    (비-root 사용자 및 Codespaces 기본 계정에서도 권한 문제 없이 공용 사용)
RUN mkdir -p /ms-playwright && \
    playwright install --with-deps chromium && \
    chmod -R 777 /ms-playwright

# 4. 서비스 포트 명시
#    6080: noVNC Web Desktop (Browser View)
#    8000: FastAPI Backend API
#    8080: Chainlit Chat UI
#    8888: Jupyter Notebook
EXPOSE 6080 8000 8080 8888

CMD ["/bin/bash"]
