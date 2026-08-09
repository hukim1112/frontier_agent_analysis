#!/bin/bash

echo "🔄 noVNC 및 관련 패키지 설치를 시작합니다..."

sudo apt-get update
sudo apt-get install -y xvfb x11vnc fluxbox novnc websockify

echo "✅ noVNC 설치가 완료되었습니다!"
