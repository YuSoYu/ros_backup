# Roscars 시스템 인터페이스 명세서

## 1. TCP Binary 통신

| 인터페이스 ID | 이름                | 방향 (송신 → 수신)      | 설명                          |
|----------------|---------------------|--------------------------|-------------------------------|
| AU            | 로그인 인증 요청    | Staff GUI → Main Server  | 사용자 로그인                  |
| IS            | 상품 정보 조회 요청 | Staff GUI → Main Server  | QR코드 기반 상품 조회          |
| IR            | 상품 요청           | Staff GUI → Main Server  | 재고 요청                      |
| CT            | 작업 생성           | Staff GUI → Main Server  | 로봇 배송 작업 요청            |
| CK            | 작업 취소           | Staff GUI → Main Server  | 진행 중 작업 중단              |
| TR            | 작업 결과 확인 요청 | Staff GUI → Main Server  | 작업 완료 여부 조회            |
| NR            | 경로 이동 요청      | Staff GUI → Main Server  | 특정 위치로 이동 지시          |
| SC            | 재고 스캔 요청      | Main Server → Staff GUI  | 수동 스캔 트리거              |
| DL            | 응답 (공통 응답 메시지) | Main Server → Staff GUI | 결과 응답 (성공/실패)          |
| IN            | AI 인식 결과 수신   | AI Server → Main Server  | Object Detection 결과 전송    |
| ER            | 에러 상태 리포트 전송 | Main Server → Staff GUI  | 장애 알림 (예: 충돌 회피 실패 등) |
| LS            | 로그 조회 요청       | Staff GUI → Main Server  | 작업 이력 외 로봇 이벤트 로그 조회 요청 (`RosCarEventLog`) |
| AS            | AI 모듈 상태 요청    | Main Server → AI Server  | Object Detector 상태 확인 (예: 작동 여부, 처리 지연) |

### [AU] 로그인 인증 요청
**방향:** Staff GUI → Main Server  
**설명:** 사용자 로그인을 위해 인증 요청을 전송합니다.

**Request**  
| Offset | Length | Name      | Type    | Description       |  
|--------|--------|-----------|---------|-------------------|  
| 0      | 2      | Cmd       | char[2] | "AU"              |  
| 2      | 32     | user_name | char[32]| 사용자 이름       |  
| 34     | 32     | password  | char[32]| 비밀번호          |

**Response**  
| Offset | Length | Name      | Type    | Description       |  
|--------|--------|-----------|---------|-------------------|  
| 0      | 2      | Cmd       | char[2] | "DL"              |  
| 2      | 1      | status    | uint8   | 0x00 성공 / 0x01 실패 |  
| 3      | 4      | user_id   | uint32  | 사용자 ID (성공 시) |  
| 7      | 1      | user_role | uint8   | 0x00 WORKER / 0x01 ADMIN |

### [IS] 상품 정보 조회 요청
**방향:** Staff GUI → Main Server  
**설명:** QR코드를 기반으로 상품 정보를 조회합니다.

**Request**  
| Offset | Length | Name          | Type    | Description       |  
|--------|--------|---------------|---------|-------------------|  
| 0      | 2      | Cmd           | char[2] | "IS"              |  
| 2      | 16     | qrcode_data   | char[16]| QR코드 데이터     |

**Response**  
| Offset | Length | Name      | Type    | Description       |  
|--------|--------|-----------|---------|-------------------|  
| 0      | 2      | Cmd       | char[2] | "DL"              |  
| 2      | 1      | status    | uint8   | 0x00 성공 / 0x01 실패 |  
| 3      | 32     | name      | char[32]| 상품 이름         |  
| 35     | 4      | size      | uint32  | 사이즈            |  
| 39     | 16     | color     | char[16]| 색상 이름         |  
| 55     | 4      | quantity  | uint32  | 재고 수량         |  
| 59     | 16     | location  | char[16]| 위치 코드         |

### [IR] 상품 요청
**방향:** Staff GUI → Main Server  
**설명:** 재고 요청을 위해 상품 정보를 전달합니다.

**Request**  
| Offset | Length | Name        | Type    | Description       |  
|--------|--------|-------------|---------|-------------------|  
| 0      | 2      | Cmd         | char[2] | "IR"              |  
| 2      | 4      | item_id     | uint32  | 상품 ID           |  
| 6      | 4      | quantity    | uint32  | 요청 수량         |  
| 10     | 16     | location    | char[16]| 요청 위치         |

**Response**  
| Offset | Length | Name      | Type    | Description       |  
|--------|--------|-----------|---------|-------------------|  
| 0      | 2      | Cmd       | char[2] | "DL"              |  
| 2      | 1      | status    | uint8   | 0x00 성공 / 0x01 실패 |  
| 3      | 4      | task_id   | uint32  | 생성된 작업 ID (성공 시) |

### [CT] 작업 생성
**방향:** Staff GUI → Main Server  
**설명:** 배송을 위한 새로운 작업 생성 요청을 전송합니다.

**Request**  
| Offset | Length | Name      | Type    | Description       |  
|--------|--------|-----------|---------|-------------------|  
| 0      | 2      | Cmd       | char[2] | "CT"              |  
| 2      | 1      | roscar_id  | uint8   | 대상 로봇 ID      |  
| 3      | 1      | task_type | uint8   | 작업 종류         |  
| 4      | 1      | from_zone | uint8   | 출발 존           |  
| 5      | 1      | to_zone   | uint8   | 목적 존           |

**Response**  
| Offset | Length | Name      | Type    | Description       |  
|--------|--------|-----------|---------|-------------------|  
| 0      | 2      | Cmd       | char[2] | "DL"              |  
| 2      | 1      | status    | uint8   | 0x00 성공 / 0x01 실패 |  
| 3      | 1      | roscar_id  | uint8   | 요청한 로봇 ID    |  
| 4      | 4      | task_id   | uint32  | 생성된 작업 ID (성공 시) |

### [CK] 작업 취소
**방향:** Staff GUI → Main Server  
**설명:** 현재 진행 중인 작업 취소 요청을 전송합니다.

**Request**  
| Offset | Length | Name      | Type    | Description       |  
|--------|--------|-----------|---------|-------------------|  
| 0      | 2      | Cmd       | char[2] | "CK"              |  
| 2      | 1      | roscar_id  | uint8   | 대상 로봇 ID      |  
| 3      | 4      | task_id   | uint32  | 취소할 작업 ID    |

**Response**  
| Offset | Length | Name      | Type    | Description       |  
|--------|--------|-----------|---------|-------------------|  
| 0      | 2      | Cmd       | char[2] | "CK"              |  
| 2      | 1      | roscar_id  | uint8   | 요청한 로봇 ID    |  
| 3      | 1      | status    | uint8   | 0x00 성공 / 0x01 실패 |

### [TR] 작업 결과 확인 요청
**방향:** Staff GUI → Main Server  
**설명:** 특정 작업의 완료 여부를 확인하기 위한 요청을 전송합니다.

**Request**  
| Offset | Length | Name      | Type    | Description       |  
|--------|--------|-----------|---------|-------------------|  
| 0      | 2      | Cmd       | char[2] | "TR"              |  
| 2      | 1      | roscar_id  | uint8   | 대상 로봇 ID      |

**Response**  
| Offset | Length | Name      | Type    | Description       |  
|--------|--------|-----------|---------|-------------------|  
| 0      | 2      | Cmd       | char[2] | "TR"              |  
| 2      | 1      | roscar_id  | uint8   | 대상 로봇 ID      |  
| 3      | 1      | status    | uint8   | 0x00 완료 / 0x01 실패 또는 미완료 |

### [NR] 경로 이동 요청
**방향:** Staff GUI → Main Server  
**설명:** 로봇에게 현재 위치에서 목적지까지 경로 이동 요청을 전송합니다.

**Request**  
| Offset | Length | Name       | Type    | Description       |  
|--------|--------|------------|---------|-------------------|  
| 0      | 2      | Cmd        | char[2] | "NR"              |  
| 2      | 1      | roscar_id   | uint8   | 대상 로봇 ID      |  
| 3      | 4      | from_point | float32 | 현재 위치 (X 또는 ID) |  
| 7      | 4      | to_point   | float32 | 목적지 위치       |

**Response**  
| Offset | Length | Name      | Type    | Description       |  
|--------|--------|-----------|---------|-------------------|  
| 0      | 2      | Cmd       | char[2] | "NR"              |  
| 2      | 1      | roscar_id  | uint8   | 대상 로봇 ID      |  
| 3      | 1      | status    | uint8   | 0x00 성공 / 0x01 실패 |

### [SC] 재고 스캔 요청
**방향:** Main Server → Staff GUI  
**설명:** 수동 또는 관리자 요청에 따라 재고 스캔 명령을 전달합니다.

**Request**  
| Offset | Length | Name      | Type    | Description       |  
|--------|--------|-----------|---------|-------------------|  
| 0      | 2      | Cmd       | char[2] | "SC"              |  
| 2      | 1      | roscar_id  | uint8   | 재고 확인 요청 로봇 ID |  
| 3      | 1      | scan_type | uint8   | 0x00 수동, 0x01 자동 |

### [DL] 공통 응답 메시지
**방향:** Main Server → Staff GUI  
**설명:** 요청 명령에 대한 공통적인 결과 응답 포맷입니다.

**Response**  
| Offset | Length | Name      | Type    | Description       |  
|--------|--------|-----------|---------|-------------------|  
| 0      | 2      | Cmd       | char[2] | "DL"              |  
| 2      | 1      | status    | uint8   | 0x00 성공 / 0x01 실패 |  
| 3      | 1      | roscar_id  | uint8   | 응답 대상 로봇 ID |

### [IN] AI 인식 결과 수신
**방향:** AI Server → Main Server  
**설명:** 객체 인식 결과를 전송합니다 (예: 사람 감지).

**Response**  
| Offset | Length | Name        | Type    | Description       |  
|--------|--------|-------------|---------|-------------------|  
| 0      | 2      | Cmd         | char[2] | "IN"              |  
| 2      | 1      | roscar_id    | uint8   | 인식 로봇 ID      |  
| 3      | 1      | result_code | uint8   | 0x00 사람, 0x01 핑키 등 |

### [ER] 에러 상태 리포트
**방향:** Main Server → Staff GUI  
**설명:** 로봇 동작 중 에러 또는 충돌 감지 시 알림을 전송합니다.

**Response**  
| Offset | Length | Name       | Type    | Description       |  
|--------|--------|------------|---------|-------------------|  
| 0      | 2      | Cmd        | char[2] | "ER"              |  
| 2      | 1      | roscar_id   | uint8   | 대상 로봇 ID      |  
| 3      | 1      | error_code | uint8   | 오류 종류 코드    |  
| 4      | n      | message    | char[n] | 오류 설명 (ASCII 문자열) |

### [LS] 로그 조회 요청
**방향:** Staff GUI → Main Server  
**설명:** 작업 이력 외에 로봇 이벤트 로그(`RosCarEventLog`)를 조회하기 위한 요청을 전송합니다. 직원도 로그 확인이 필요합니다.

**Request**  
| Offset | Length | Name      | Type    | Description       |  
|--------|--------|-----------|---------|-------------------|  
| 0      | 2      | Cmd       | char[2] | "LS"              |  
| 2      | 4      | user_id   | uint32  | 요청 사용자 ID    |  
| 6      | 1      | log_type  | uint8   | 로그 종류 (예: 0x00=로봇 이벤트) |  
| 7      | 8      | start_time| uint64  | 조회 시작 시간 (Unix 타임스탬프) |  
| 15     | 8      | end_time  | uint64  | 조회 종료 시간 (Unix 타임스탬프) |

**Response**  
| Offset | Length | Name      | Type    | Description       |  
|--------|--------|-----------|---------|-------------------|  
| 0      | 2      | Cmd       | char[2] | "DL"              |  
| 2      | 1      | status    | uint8   | 0x00 성공 / 0x01 실패 |  
| 3      | 4      | count     | uint32  | 반환된 로그 수    |  
| 7      | n      | log_data  | bytes[] | 로그 데이터 (JSON 직렬화) |

### [AS] AI 모듈 상태 요청
**방향:** Main Server → AI Server  
**설명:** AI Server의 Object Detector 상태(예: 작동 여부, 처리 지연)를 확인하여 시스템 안정성을 모니터링합니다.

**Request**  
| Offset | Length | Name      | Type    | Description       |  
|--------|--------|-----------|---------|-------------------|  
| 0      | 2      | Cmd       | char[2] | "AS"              |  
| 2      | 1      | module_id | uint8   | 모듈 ID (예: 0x00=Object Detector) |

**Response**  
| Offset | Length | Name      | Type    | Description       |  
|--------|--------|-----------|---------|-------------------|  
| 0      | 2      | Cmd       | char[2] | "AS"              |  
| 2      | 1      | module_id | uint8   | 모듈 ID           |  
| 3      | 1      | status    | uint8   | 0x00 정상 / 0x01 비정상 |  
| 4      | 4      | latency   | float32 | 처리 지연 (초 단위) |

## 2. UDP 통신

| 인터페이스 ID | 이름            | 방향 (송신 → 수신)      | 설명                        |
|----------------|-----------------|--------------------------|-----------------------------|
| VF            | 영상 프레임 전송 | Video Sender → AI Server | Pi Camera 영상 전송 (MediaFrame) |

### [VF] 영상 프레임 전송
**방향:** Video Sender → AI Server  
**설명:** 로봇이 수집한 영상 데이터를 실시간으로 AI 서버에 전송합니다.

**Packet**  
| Offset | Length      | Name        | Type    | Description       |  
|--------|-------------|-------------|---------|-------------------|  
| 0      | 1           | roscar_id    | uint8   | 송신 로봇 ID      |  
| 1      | 4           | frame_size  | uint32  | 이미지 바이트 길이 |  
| 5      | frame_size  | image_bytes | bytes[] | 이미지 바이너리 데이터 |

## 3. ROS2 Topic (.msg)

| Topic 이름             | 방향 (송신 → 수신)      | 설명                          |
|-------------------------|--------------------------|-------------------------------|
| /roscar/status/battery   | Mobile Controller → Main Server | 배터리 잔량 보고              |
| /roscar/status/charge    | Main Server → Mobile Controller | 충전 위치 이동 명령           |
| /roscar/navigation/goal  | Main Server → Mobile Controller | 경로 이동 명령                |
| /inventory/result/admin | Main Server → Admin GUI | 재고 파악 결과                |
| /alert/emergency/admin  | Mobile Controller → Main Server | 긴급 상황 보고                |
| /dashboard/status/update | Main Server → Mobile Controller | 대시보드 상태 갱신            |
| /roscar/status/log       | Main Server → Admin GUI | 로봇 상태 로그 전송            |
| /roscar/pose/update      | Mobile Controller → Main Server | 로봇 실시간 위치 보고         |
| /roscar/sensor/fusion    | Mobile Controller → Main Server | IR/초음파/IMU 센서 통합 보고  |
| /roscar/task/progress    | Mobile Controller → Main Server | 작업 진행률 실시간 보고       |
| /roscar/task/complete    | Mobile Controller → Main Server | 작업 완료 결과 보고           |
| /roscar/emergency/state  | Main Server → Mobile Controller | 비상 명령 송신 (정지/재시작 등) |
| /roscar/obstacle/response | Main → Mobile       | AI 인식 기반 정지/회피 명령   |
| /roscar/avoidance/cmd    | Main → Mobile       | 장애물 회피용 로컬 제어 명령  |
| /roscar/precision_stop/cmd | Main → Mobile    | 정밀 정지 명령                |
| /roscar/precision_stop/result | Mobile → Main  | 정밀 정지 결과 보고           |
| /roscar/obstacle/detected | Mobile → Main     | 장애물 감지 이벤트 보고       |
| /log/event/admin        | Main Server → Admin GUI | 실시간 로그 이벤트 전송       |

### [/roscar/status/battery] 배터리 잔량 보고
**방향:** Mobile Controller → Main Server  
**설명:** 로봇의 현재 배터리 잔량 및 충전 상태를 주기적으로 보고합니다.

**Message Definition**  
| Field Name      | Type                   | Description       |  
|-----------------|------------------------|-------------------|  
| roscar_id        | uint8                 | 로봇 ID           |  
| battery_percent | float32               | 현재 배터리 잔량 (%) |  
| is_charging     | bool                  | 충전 중 여부      |  
| stamp           | builtin_interfaces/Time | 타임스탬프        |

### [/roscar/status/charge] 충전 위치 이동 명령
**방향:** Main Server → Mobile Controller  
**설명:** 로봇에게 충전소로 이동할 좌표를 지정합니다.

**Message Definition**  
| Field Name | Type                   | Description       |  
|------------|------------------------|-------------------|  
| roscar_id   | uint8                 | 로봇 ID           |  
| goal_x     | float32               | 목적지 X좌표      |  
| goal_y     | float32               | 목적지 Y좌표      |  
| theta      | float32               | 회전 각도         |  
| stamp      | builtin_interfaces/Time | 타임스탬프        |

### [/roscar/navigation/goal] 경로 이동 명령
**방향:** Main Server → Mobile Controller  
**설명:** 로봇에게 일반적인 목적지 이동 명령을 전송합니다.

**Message Definition**  
| Field Name | Type                   | Description       |  
|------------|------------------------|-------------------|  
| roscar_id   | uint8                 | 로봇 ID           |  
| goal_x     | float32               | 목적지 X좌표      |  
| goal_y     | float32               | 목적지 Y좌표      |  
| theta      | float32               | 회전 각도         |  
| stamp      | builtin_interfaces/Time | 타임스탬프        |

### [/inventory/result/admin] 재고 파악 결과 전송
**방향:** Main Server → Admin GUI  
**설명:** 로봇이 검수한 재고 항목의 결과를 전송합니다.

**Message Definition**  
| Field Name | Type                   | Description       |  
|------------|------------------------|-------------------|  
| roscar_id   | uint8                 | 로봇 ID           |  
| item_id    | uint32                | 제품 ID           |  
| location   | string                | 제품 위치 코드    |  
| status     | uint8                 | 상태 코드 (예: 0=정상) |  
| stamp      | builtin_interfaces/Time | 타임스탬프        |

### [/alert/emergency/admin] 긴급 상황 보고
**방향:** Mobile Controller → Main Server  
**설명:** 로봇 측에서 감지한 긴급 이벤트를 보고합니다.

**Message Definition**  
| Field Name | Type                   | Description       |  
|------------|------------------------|-------------------|  
| roscar_id   | uint8                 | 로봇 ID           |  
| event_code | uint8                 | 이벤트 코드 (예: 0x01 EMERGENCY_STOP) |  
| message    | string                | 상세 메시지       |  
| stamp      | builtin_interfaces/Time | 타임스탬프        |

### [/dashboard/status/update] 대시보드 상태 갱신
**방향:** Main Server → Mobile Controller  
**설명:** 로봇의 현재 작업 정보 및 위치 상태를 갱신합니다.

**Message Definition**  
| Field Name | Type                   | Description       |  
|------------|------------------------|-------------------|  
| roscar_id   | uint8                 | 로봇 ID           |  
| task_id    | uint32                | 현재 작업 ID      |  
| pose_x     | float32               | 위치 X좌표        |  
| pose_y     | float32               | 위치 Y좌표        |  
| stamp      | builtin_interfaces/Time | 타임스탬프        |

### [/roscar/status/log] 로봇 상태 로그 전송
**방향:** Main Server → Admin GUI  
**설명:** 로봇의 상태 변화나 이벤트 로그를 전달합니다.

**Message Definition**  
| Field Name | Type                   | Description       |  
|------------|------------------------|-------------------|  
| roscar_id   | uint8                 | 로봇 ID           |  
| log_type   | uint8                 | 로그 종류 (상태/오류 등) |  
| message    | string                | 로그 메시지 내용  |  
| stamp      | builtin_interfaces/Time | 타임스탬프        |

### [/roscar/pose/update] 로봇 실시간 위치 보고
**방향:** Mobile Controller → Main Server  
**설명:** 로봇의 실시간 위치를 보고합니다.

**Message Definition**  
| Field Name | Type                   | Description       |  
|------------|------------------------|-------------------|  
| roscar_id   | uint8                 | 로봇 ID           |  
| pose_x     | float32               | 위치 X좌표        |  
| pose_y     | float32               | 위치 Y좌표        |  
| stamp      | builtin_interfaces/Time | 타임스탬프        |

### [/roscar/sensor/fusion] IR/초음파/IMU 센서 통합 보고
**방향:** Mobile Controller → Main Server  
**설명:** IR, 초음파, IMU 센서의 통합 데이터를 보고합니다.

**Message Definition**  
| Field Name      | Type                   | Description       |  
|-----------------|------------------------|-------------------|  
| roscar_id        | uint8                 | 로봇 ID           |  
| lidar_raw       | string                | LiDAR 데이터 (JSON) |  
| imu_data        | string                | IMU 데이터 (JSON) |  
| ultrasonic_data | string                | 초음파 데이터 (JSON) |  
| stamp           | builtin_interfaces/Time | 타임스탬프        |

### [/roscar/task/progress] 작업 진행률 실시간 보고
**방향:** Mobile Controller → Main Server  
**설명:** 작업 진행률을 실시간으로 보고합니다.

**Message Definition**  
| Field Name | Type                   | Description       |  
|------------|------------------------|-------------------|  
| roscar_id   | uint8                 | 로봇 ID           |  
| task_id    | uint32                | 작업 ID           |  
| progress   | uint8                 | 진행률 (0~100)    |  
| stamp      | builtin_interfaces/Time | 타임스탬프        |

### [/roscar/task/complete] 작업 완료 결과 보고
**방향:** Mobile Controller → Main Server  
**설명:** 작업 완료 결과를 보고합니다.

**Message Definition**  
| Field Name | Type                   | Description       |  
|------------|------------------------|-------------------|  
| roscar_id   | uint8                 | 로봇 ID           |  
| task_id    | uint32                | 작업 ID           |  
| success    | bool                  | 성공 여부         |  
| stamp      | builtin_interfaces/Time | 타임스탬프        |

### [/roscar/emergency/state] 비상 명령 송신
**방향:** Main Server → Mobile Controller  
**설명:** 비상 명령(정지/재시작 등)을 송신합니다.

**Message Definition**  
| Field Name | Type                   | Description       |  
|------------|------------------------|-------------------|  
| roscar_id   | uint8                 | 로봇 ID           |  
| command    | uint8                 | 0x00 정지 / 0x01 재시작 |  
| stamp      | builtin_interfaces/Time | 타임스탬프        |

### [/roscar/obstacle/response] AI 인식 기반 정지/회피 명령
**방향:** Main → Mobile  
**설명:** AI 인식 결과를 기반으로 정지 또는 회피 명령을 전송합니다.

**Message Definition**  
| Field Name | Type                   | Description       |  
|------------|------------------------|-------------------|  
| roscar_id   | uint8                 | 로봇 ID           |  
| command    | uint8                 | 0x00 정지 / 0x01 회피 |  
| stamp      | builtin_interfaces/Time | 타임스탬프        |

### [/roscar/avoidance/cmd] 장애물 회피용 로컬 제어 명령
**방향:** Main → Mobile  
**설명:** 장애물 회피를 위한 로컬 제어 명령을 전송합니다.

**Message Definition**  
| Field Name | Type                   | Description       |  
|------------|------------------------|-------------------|  
| roscar_id   | uint8                 | 로봇 ID           |  
| direction  | float32               | 회피 방향 (각도)  |  
| speed      | float32               | 회피 속도         |  
| stamp      | builtin_interfaces/Time | 타임스탬프        |

### [/roscar/precision_stop/cmd] 정밀 정지 명령
**방향:** Main → Mobile  
**설명:** 정밀 정지 명령을 전송합니다.

**Message Definition**  
| Field Name | Type                   | Description       |  
|------------|------------------------|-------------------|  
| roscar_id   | uint8                 | 로봇 ID           |  
| target_x   | float32               | 정지 목표 X좌표   |  
| target_y   | float32               | 정지 목표 Y좌표   |  
| stamp      | builtin_interfaces/Time | 타임스탬프        |

### [/roscar/precision_stop/result] 정밀 정지 결과 보고
**방향:** Mobile → Main  
**설명:** 정밀 정지 결과를 보고합니다.

**Message Definition**  
| Field Name  | Type                   | Description       |  
|-------------|------------------------|-------------------|  
| roscar_id    | uint8                 | 로봇 ID           |  
| success     | bool                  | 성공 여부         |  
| deviation   | float32               | 오차 (cm 단위)    |  
| stamp       | builtin_interfaces/Time | 타임스탬프        |

### [/roscar/obstacle/detected] 장애물 감지 이벤트 보고
**방향:** Mobile → Main  
**설명:** 장애물 감지 이벤트를 보고합니다.

**Message Definition**  
| Field Name | Type                   | Description       |  
|------------|------------------------|-------------------|  
| roscar_id   | uint8                 | 로봇 ID           |  
| distance   | float32               | 장애물 거리 (m)   |  
| direction  | float32               | 장애물 방향 (각도) |  
| stamp      | builtin_interfaces/Time | 타임스탬프        |

### [/log/event/admin] 실시간 로그 이벤트 전송
**방향:** Main Server → Admin GUI  
**설명:** 실시간 로그 이벤트(`DeliveryEventLog`, `RosCarEventLog`)를 전송하여 관리자가 실시간 모니터링을 할 수 있도록 합니다.

**Message Definition**  
| Field Name  | Type                   | Description         |  
|-------------|------------------------|---------------------|  
| event_id    | uint32                | 이벤트 ID           |  
| event_type  | uint8                 | 이벤트 종류 (예: 0x00=EMERGENCY_STOP) |  
| event_data  | string                | 이벤트 상세 데이터 (JSON 직렬화) |  
| stamp       | builtin_interfaces/Time | 타임스탬프          |

## 4. ROS2 Service (.srv)

| Service 이름         | 방향 (송신 → 수신)      | 설명                  |
|-----------------------|--------------------------|-----------------------|
| /inventory/start_check | Admin GUI → Main Server | 재고 파악 모드 시작    |
| /task/query_status    | GUI → Main Server       | 작업 상태 요청        |

### [/inventory/start_check] 재고 파악 모드 시작
**방향:** Admin GUI → Main Server  
**설명:** 관리자에 의해 재고 확인 절차를 시작하도록 요청합니다.

**Request**  
| Field Name    | Type    | Description       |  
|---------------|---------|-------------------|  
| admin_user_id | uint32  | 요청 관리자 ID    |  
| mode          | uint8   | 0: 수동, 1: 자동  |  
| trigger_msg   | string  | 추가 메시지 (선택) |

**Response**  
| Field Name | Type   | Description       |  
|------------|--------|-------------------|  
| success    | bool   | 요청 처리 결과    |  
| message    | string | 처리 상태 또는 오류 메시지 |

### [/task/query_status] 작업 상태 요청
**방향:** GUI → Main Server  
**설명:** 작업 ID 기준으로 현재 상태 확인 요청을 전송합니다.

**Request**  
| Field Name | Type   | Description       |  
|------------|--------|-------------------|  
| task_id    | uint32 | 조회할 작업 ID    |

**Response**  
| Field Name   | Type   | Description       |  
|--------------|--------|-------------------|  
| task_id      | uint32 | 작업 ID           |  
| delivery_id  | uint32 | 연관된 배송 ID    |  
| task_status  | uint8  | 상태 코드 (0: TO_DO 등) |  
| progress     | uint8  | 진행률 (0~100)    |  
| current_zone | string | 현재 위치 존 (예: R2-B1) |  
| message      | string | 설명 또는 상태 메시지 |

## 5. ROS2 Action (.action)

| Action 이름            | 방향 (송신 → 수신)      | 설명                          |
|-------------------------|--------------------------|-------------------------------|
| /navigation/move_to_goal | Main Server → Mobile Controller | 경로 이동 수행 및 피드백      |
| /delivery/start         | Main Server → Mobile Controller | 배송 작업 전체 흐름 실행      |
| /inventory/scan_items   | Main Server → Mobile Controller | 자동 재고 스캔 실행 및 상태 보고 |
| /maintenance/charge     | Main Server → Mobile Controller | 충전 프로세스 수행 및 완료 보고 |
| /security/patrol        | Main Server → Mobile Controller | 비업무 시간대 순찰 수행 및 보고 |

### [/navigation/move_to_goal] 경로 이동 수행
**방향:** Main Server → Mobile Controller  
**설명:** 로봇이 목표 위치까지 이동하며 도중 피드백을 전송합니다.

**Goal**  
| Field Name | Type   | Description       |  
|------------|--------|-------------------|  
| roscar_id   | uint8  | 로봇 ID           |  
| goal_x     | float32 | 목적지 X좌표      |  
| goal_y     | float32 | 목적지 Y좌표      |  
| theta      | float32 | 목표 방향 각도    |

**Feedback**  
| Field Name | Type   | Description       |  
|------------|--------|-------------------|  
| progress   | uint8  | % 기준 진행률     |  
| current_x  | float32 | 현재 위치 X       |  
| current_y  | float32 | 현재 위치 Y       |

**Result**  
| Field Name | Type   | Description       |  
|------------|--------|-------------------|  
| success    | bool   | 성공 여부         |  
| message    | string | 종료 메시지 또는 에러 |

### [/delivery/start] 배송 작업 전체 흐름 실행
**방향:** Main Server → Mobile Controller  
**설명:** 지정된 작업 ID 기반으로 로봇이 배송 전체를 수행합니다.

**Goal**  
| Field Name | Type   | Description       |  
|------------|--------|-------------------|  
| task_id    | uint32 | 배송 작업 ID      |  
| roscar_id   | uint8  | 로봇 ID           |

**Feedback**  
| Field Name | Type   | Description       |  
|------------|--------|-------------------|  
| phase      | string | 현재 수행 단계    |  
| location   | string | 현재 위치 (존 이름 등) |

**Result**  
| Field Name | Type   | Description       |  
|------------|--------|-------------------|  
| success    | bool   | 성공 여부         |  
| message    | string | 종료 메시지 또는 에러 |

### [/inventory/scan_items] 자동 재고 스캔 실행 및 상태 보고
**방향:** Main Server → Mobile Controller  
**설명:** 로봇이 지정 존 또는 전체 존을 순회하며 자동 재고 스캔을 실행합니다.

**Goal**  
| Field Name | Type   | Description       |  
|------------|--------|-------------------|  
| scan_zone  | string | 스캔할 존 또는 범위 |  
| roscar_id   | uint8  | 로봇 ID           |

**Feedback**  
| Field Name  | Type   | Description       |  
|-------------|--------|-------------------|  
| scanned_item| uint32 | 인식된 아이템 수  |  
| zone        | string | 현재 스캔 위치 존 |

**Result**  
| Field Name  | Type   | Description       |  
|-------------|--------|-------------------|  
| total_items | uint32 | 총 인식된 항목 수 |  
| success     | bool   | 성공 여부         |

### [/maintenance/charge] 충전 프로세스 수행
**방향:** Main Server → Mobile Controller  
**설명:** 로봇이 충전소로 이동하여 충전 프로세스를 수행하고 단계별 피드백을 제공합니다.

**Goal**  
| Field Name  | Type   | Description       |  
|-------------|--------|-------------------|  
| roscar_id    | uint8  | 로봇 ID           |  
| charge_zone | string | 충전소 위치 (존 이름) |

**Feedback**  
| Field Name      | Type   | Description       |  
|-----------------|--------|-------------------|  
| phase           | string | 현재 단계 (예: 이동 중, 충전 중) |  
| battery_percent | float32 | 현재 배터리 잔량 (%) |

**Result**  
| Field Name | Type   | Description       |  
|------------|--------|-------------------|  
| success    | bool   | 성공 여부         |  
| message    | string | 종료 메시지 또는 에러 |

### [/security/patrol] 비업무 시간대 순찰 수행
**방향:** Main Server → Mobile Controller  
**설명:** 비업무 시간대에 매장 내부를 순찰하며 이상 상황을 보고합니다.

**Goal**  
| Field Name  | Type   | Description       |  
|-------------|--------|-------------------|  
| roscar_id    | uint8  | 로봇 ID           |  
| patrol_zone | string | 순찰할 구역 (예: 전체 또는 특정 존) |

**Feedback**  
| Field Name  | Type   | Description       |  
|-------------|--------|-------------------|  
| current_zone| string | 현재 순찰 중인 구역 |  
| event       | string | 발견된 이벤트 (예: 장애물 감지) |

**Result**  
| Field Name | Type   | Description       |  
|------------|--------|-------------------|  
| success    | bool   | 성공 여부         |  
| message    | string | 종료 메시지 또는 에러 |