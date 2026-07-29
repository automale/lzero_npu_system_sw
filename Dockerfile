# 1. Base Image: 최신 Ubuntu 24.04 LTS

FROM ubuntu:24.04

# 2. Environment Variables
#시간대 설정 및 apt-get 설치 시 사용자 입력창(interactive prompt) 방지

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 3. System Dependencies 설치
# C++ 펌웨어 컴파일을 위한 build-essential과 Python 환경, 기타 유틸리티 설치

RUN apt-get update && apt-get install -y \
build-essential \
python3 \
python3-pip \
python3-venv \
git \
wget \
curl \
vim \
&& rm -rf /var/lib/apt/lists/*

# 4. Workspace 설정

WORKDIR /workspace

#5. Python Virtual Environment 설정
# Ubuntu 24.04의 EXTERNALLY-MANAGED 에러를 피하기 위해 가상환경을 기본으로 사용합니다.

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 6. Python Packages 설치
# NPU 컴파일러 스크립트 작성에 필요한 패키지 설치

RUN pip install --upgrade pip && \
pip install numpy gguf

# 7. 컨테이너 실행 시 기본 명령
CMD ["/bin/bash"]