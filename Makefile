# 도커 이미지 및 컨테이너 이름 설정
IMAGE_NAME = npu-compiler:1.0
CONTAINER_NAME = npu-compiler-env
WORKSPACE = $(PWD)

.PHONY: build run start stop clean prune help

# 기본 안내
help:
	@echo "=========================================================="
	@echo " NPU Compiler Docker Environment Makefile"
	@echo "=========================================================="
	@echo "사용 가능한 명령어:"
	@echo "  make build  - Docker 이미지 빌드"
	@echo "  make run    - 새 컨테이너 생성 및 실행 (현재 디렉토리 마운트)"
	@echo "  make start  - 중지된 컨테이너 다시 켜기 및 접속"
	@echo "  make stop   - 실행 중인 컨테이너 끄기"
	@echo "  make clean  - 컨테이너 삭제"
	@echo "  make prune  - 컨테이너 및 이미지 완전 삭제 (초기화)"
	@echo "=========================================================="

# 이미지 빌드
build:
	docker build -t $(IMAGE_NAME) .

# 컨테이너를 새로 생성하고 쉘로 접속
run:
	docker run -it --name $(CONTAINER_NAME) -v "$(WORKSPACE):/workspace" $(IMAGE_NAME)

# 껐던 컨테이너를 다시 켜고 접속
start:
	docker start -i $(CONTAINER_NAME)

# 실행 중인 컨테이너 끄기
stop:
	docker stop $(CONTAINER_NAME)

# 컨테이너 삭제
clean: stop
	docker rm $(CONTAINER_NAME)

# 컨테이너와 이미지 모두 삭제
prune: clean
	docker rmi $(IMAGE_NAME)

### 💡 사용 방법
# 터미널에서 아래 명령어들만 입력하시면 됩니다. (Makefile은 탭(Tab) 들여쓰기가 필수인데, 위 코드는 정확히 적용되어 있으니 그대로 복사해 쓰시면 됩니다.)
#1. **최초 1회 셋업:** `make build`를 쳐서 이미지를 굽습니다.
#2. **첫 실행:** `make run`을 치면 컨테이너가 생성되면서 바로 파이썬 환경으로 들어갑니다.
#3. **끄기:** 쉘에서 `exit`으로 나오시거나 터미널을 닫으면 됩니다. 백그라운드에 남아있다면 `make stop`으로 완전히 끕니다.
#4. **다시 켜기:** 다음 날 다시 작업하실 때는 `make start`만 치시면 기존에 만들어둔 컨테이너가 켜지면서 바로 접속됩니다! 