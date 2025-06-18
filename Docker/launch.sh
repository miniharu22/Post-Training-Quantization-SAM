#!/bin/bash

# 기본 실행 모드를 초기화
mode="nothing"

# 아래 명령어 옵션들을 Parse
# -b(build) 또는 -r(run) 옵션을 처리
while getopts "br" opt; do
  case $opt in
    # -b 옵션 : Build mode 설정
    b)              
      mode="build"
      ;;
    # -r 옵션 : Launch mode 설정
    r)              
      mode="run"
      ;;
    # 정의되지 않은 옵션에 대해 exit 처리
    *)              
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# 선택된 모드가 build나 run이 아닌 경우, 에러 출력하고 종료
if [ "$mode" != "build" ] && [ "$mode" != "run" ]; then
  echo "Unsupported mode use ['-b' / '-r']" >&2
  exit 1
fi

# 모드에 따라 Docker 명령어를 실행   
if [ "$mode" == "build" ]; then
  # Docker Image를 빌드    
  docker build -t sam:trt .

elif [ "$mode" == "run" ]; then
  # x 서버 접근 권한을 임시로 허용   
  # GUI 프로그램이 컨테이너에서 표시되도록 설정
  xhost +
  # GPU를 사용하여 컨테이너를 실행합니다
  # --gpus all     : 모든 GPU에 접근
  # -it            : 인터랙티브 터미널
  # --rm           : 종료 시 컨테이너 자동 삭제
  # --net=host     : 호스트 네트워크 사용
  # -e DISPLAY     : 호스트의 DISPLAY 환경변수 전달
  # -v $(pwd):/workspace : 현재 디렉터리를 컨테이너의 /workspace로 마운트
  docker run --gpus all -it --rm --net=host -e DISPLAY=$DISPLAY -v $(pwd):/workspace sam:trt
fi

