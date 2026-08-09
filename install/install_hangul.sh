#!/bin/bash

echo "🔄 한글 폰트 및 로케일 설정을 시작합니다..."

# 1. 패키지 업데이트 및 나눔 폰트 설치
sudo apt-get update
sudo apt-get install -y fonts-nanum fonts-nanum-coding fonts-nanum-extra

# 2. 폰트 캐시 갱신 (설치된 폰트를 시스템에 인식시킴)
sudo fc-cache -fv

# 3. (선택사항) 한국어 로케일 설정
# 시스템 언어 자체를 한국어로 바꾸고 싶을 때 필요합니다.
sudo apt-get install -y language-pack-ko
sudo locale-gen ko_KR.UTF-8
sudo update-locale LANG=ko_KR.UTF-8

echo "✅ 한글 폰트 설치가 완료되었습니다!"
echo "💡 적용을 위해 실행 중인 start_vnc.sh를 종료하고 다시 실행해 주세요."
