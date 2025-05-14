
```plaintext
project_root/
├── ai_server/                         # AI 서버 (Docker 기반, 영상 인식 처리)
│   ├── model_loader.py
│   ├── tcp_server.py                 # Main Server와 TCP 통신
│   ├── docker-compose.yml                  # 개별 빌드용 이미지 정의
│   └── filesystem/
│   │   ├── file_writer.py           # 수신한 이미지 파일 저장 로직
│   │   └── file_cleanup.py          # 임시 파일 삭제 및 디스크 관리
│   └── detection/
│       ├── object_detector.py        # 추론 모듈 (Python or C++)
│       └── tracking_module.py        # 추론 후처리 (Python)
│
├── docs/                             # 문서 디렉토리 (명세서, 다이어그램 등)
│   ├── interface_specifications.md
│   └── system_architecture.md
│
├── gui/                              # PyQt 기반 GUI
│   ├── shared/                       # 공통 위젯 및 유틸
│   ├── manager_gui/
│   │   ├── main.py                   # 관리자 GUI 실행 진입점
│   │   ├── monitor_panel.py
│   │   └── task_control_panel.py
│   └── staff_gui/
│       ├── main.py                   # 작업자 GUI 실행 진입점
│       ├── task_form.py
│       └── task_history.py
│
├── main_server/                      # 중앙 통합 서버
│   ├── interfaces/                  # 메시지, 서비스 정의 (ROS2 .msg/.srv)
│   ├── ros_nodes/                     
│   │   ├── service_node.py
│   │   ├── task_manager.py
│   │   └── logger.cpp                  # 빠른 로깅용 C++ 모듈
│   ├── network/ 
│   │   ├── tcp_handler.py               # TCP 통신 처리 (Python)
│   │   ├── emergency_handler.py
│   │   └── message_router.rs             # 고성능 메시지 큐 처리 (Rust)
│   └── parameters_config.yaml
│
├── mobile_controller/               # 로봇 탑재 라즈베리파이
│   ├── ros_nodes/
│   │   └── controller_node.py            # ROS2 엔트리 포인트
│   ├── sensor.cpp                      # 모든 센서 통합 처리 (C++)
│   ├── actuator.cpp                     # 모든 액추에이터 통합 제어 (Python)
│   ├── camera_streamer.cpp            # 영상 캡처 및 UDP 전송
│   ├── image_encoder.rs              # Rust 최적화 모듈 - 이미지 인코딩 고속 처리 (Rust)
│   └── emergency_monitor.cpp              # 긴급 상황 감지
│
├── launch/                          # ROS2 launch 파일 모음
│   ├── mobile.launch.py
│   └── main_server.launch.py
│
├── shared_interfaces/              # 공통 ROS2 인터페이스 정의
│   ├── msg/
│   │   ├── BatteryStatus.msg
│   │   ├── TaskResult.msg
│   │   └── Emergency.msg
│   └── srv/
│       └── ManualOverride.srv
│
└── README.md
```