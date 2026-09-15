#!/bin/bash

# 1. 기존 잔여 잠금 파일 및 프로세스 정리
pkill -f Xvfb 2>/dev/null || true
pkill -f x11vnc 2>/dev/null || true
pkill -f fluxbox 2>/dev/null || true
pkill -f novnc 2>/dev/null || true
pkill -f websockify 2>/dev/null || true

rm -f /tmp/.X1-lock 2>/dev/null || sudo rm -f /tmp/.X1-lock 2>/dev/null || true
rm -f /tmp/.X11-unix/X1 2>/dev/null || sudo rm -f /tmp/.X11-unix/X1 2>/dev/null || true

# 2. 가상 디스플레이 시작 (해상도 1280x800, 24비트 색상)
echo "🖥️ Xvfb 가상 디스플레이(:1)를 시작합니다..."
Xvfb :1 -screen 0 1280x800x24 &
sleep 1
export DISPLAY=:1

# 3. 경량 윈도우 매니저(fluxbox) 실행
echo "🪟 Fluxbox 윈도우 매니저를 시작합니다..."
fluxbox &
sleep 1

# 4. VNC 서버 실행 (5900 포트)
echo "📡 x11vnc 서버를 시작합니다 (포트 5900)..."
x11vnc -display :1 -nopw -forever -shared -rfbport 5900 &
sleep 1

# 5. noVNC 웹 프록시 실행 (6080 포트)
echo "🌐 noVNC 웹 서버를 시작합니다 (포트 6080)..."
echo "💡 Codespaces [Ports] 탭에서 6080번 포트를 열면 브라우저 GUI 화면을 보실 수 있습니다."

if [ -f /usr/share/novnc/utils/novnc_proxy ]; then
    /usr/share/novnc/utils/novnc_proxy --vnc localhost:5900 --listen 6080
elif [ -f /usr/bin/novnc_proxy ]; then
    /usr/bin/novnc_proxy --vnc localhost:5900 --listen 6080
else
    websockify --web /usr/share/novnc/ 6080 localhost:5900
fi