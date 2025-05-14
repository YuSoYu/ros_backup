# 1. 사용자 단말 초기화 및 인증 (user_authentication)

```plantuml
@startuml
title 사용자 단말 초기화 및 인증 공통 흐름

actor User
participant "GUI (Staff / Manager)" as GUI
participant "Main Service" as Main
database "User (user_id, user_name, user_role, password, can_call_roscar)" as DB

== GUI 실행 및 서버 연결 ==

User -> GUI : 프로그램 실행
activate GUI
GUI -> Main : 서버 연결 요청
activate Main
Main --> GUI : 서버 연결 결과\nsuccess=true
deactivate Main

alt 서버 연결 성공
    GUI -> GUI : 로그인 화면 표시

    User -> GUI : ID, Password 입력
    GUI -> Main : LoginRequest(user_name, password)
    activate Main

    ' 사용자 조회
    Main -> DB : SELECT * FROM User WHERE user_name = ?
    activate DB
    DB --> Main : user_id, user_name, user_role, can_call_roscar 등
    deactivate DB

    Main -> Main : 비밀번호 해시 비교

    alt 비밀번호 일치
        alt user_role == WORKER
            Main -> DB : SELECT * FROM gui_layouts WHERE user_id = ?
            activate DB
            DB --> Main : layoutData (압축된 JSON)
            deactivate DB
            Main --> GUI : LoginResponse(success=true, user_role=WORKER)
        else user_role == ADMIN
            Main -> DB : SELECT * FROM admin_dashboard_data WHERE user_id = ?
            activate DB
            DB --> Main : dashboardData\n(statusSummary, taskStats, roscarHealth)
            deactivate DB
            Main --> GUI : LoginResponse(success=true, user_role=ADMIN)
        end
    else 비밀번호 불일치
        Main --> GUI : LoginResponse(success=false, reason="Invalid credentials")
    end
    deactivate Main
else 서버 연결 실패
    GUI -> GUI : 오류 팝업 표시
end
deactivate GUI
@enduml
```

---

# 2. 재고 조회 및 요청 (inventory_request_flow)

## 상품 정보 조회

```plantuml
title 재고 조회 및 요청 흐름

actor User
participant "Staff GUI" as Staff
participant "Main Service" as Main
database "QRCode, Inventory, ShoesModel, Location" as DB

User -> Staff : QR 스캔
activate Staff
Staff -> Staff : QR 파싱 → qrcode_data=QR20250430X001

Staff -> Main : SearchItemByQRCode(qrcode_data)
activate Main
Main -> DB : SELECT name, size, color, quantity, location\nFROM 조인된 테이블\nWHERE qrcode_data = ?
activate DB
DB --> Main : 상품 정보 반환\n(Nike Air Zoom, 270, black, 5, R3-S2-B1)
deactivate DB

Main --> Staff : SearchItemResponse(item, size, quantity, location)
deactivate Main

Staff -> Staff : GUI에 상품 정보 표시
```

## 상품 요청

```plantuml
@startuml
title 재고 조회 및 요청 흐름

actor User
participant "Staff GUI" as Staff
participant "Main Service" as Main
participant "Mobile Controller" as Mobile
database "QRCode, Inventory, ShoesModel, Location" as DB


User -> Staff : [요청] 클릭
Staff -> Main : InventoryRequest(item_id, quantity, location)
activate Main

Main -> DB : INSERT Task, Delivery (TO_DO 상태)
activate DB
DB --> Main : task_id=TASK-789
deactivate DB

Main -> Mobile : PickupCommand(task_id, item_id, from, to)
activate Mobile
Mobile -> Mobile : 경로 이동 및 집재
Mobile -> Main : ArrivalReport(task_id, status=ARRIVED)
deactivate Mobile

Main --> Staff : TaskStatusUpdate(task_id, status=배송중)
deactivate Main

Staff -> Staff : 배송중 상태 표시

User -> Staff : [담기] 클릭
Staff -> Staff : 로컬 캐시 저장 + UI 갱신
deactivate Staff

@enduml
```

---

# 3. 작업 생성 및 로봇 할당 (task_assignment_flow)

```plantuml
@startuml
title 작업 생성 및 로봇 할당 흐름

actor User
participant "Staff GUI" as Staff
participant "Main Service" as Main
participant "Mobile Controller" as Mobile
database "Task, Delivery, RosCars" as DB


User -> Staff : 작업 요청 클릭
activate Staff
Staff -> Main : CreateTaskRequest(item_id, from, to)
deactivate Staff

activate Main
Main -> DB : INSERT INTO Delivery + Task (status=TO_DO)
activate DB
DB --> Main : delivery_id, task_id
deactivate DB


Main -> DB : SELECT * FROM RosCars WHERE status='STANDBY'
activate DB
DB --> Main : [RB01, RB02]
deactivate DB
Main -> Main : best_roscar = RB01


Main -> Mobile : AssignTask(task_id, item_id, from, to)
activate Mobile
Mobile -> Main : TaskAck(result=success)

alt 수락 성공
    Main -> DB : UPDATE Task, Delivery SET status='IN_PROGRESS'
    activate DB
    DB --> Main : OK
    deactivate DB
else 수락 실패
    Main -> Main : 재할당 또는 대기 큐 등록
end
deactivate Mobile


loop 작업 진행 중
    Mobile -> Main : TaskProgressUpdate(task_id, 위치, 진행률)
    Main -> DB : UPDATE Task SET progress, location
end

Mobile -> Main : TaskCompleteReport(task_id, result=SUCCESS)
Main -> DB : UPDATE Task, Delivery 상태='DONE', 완료시간 저장
DB --> Main : OK

Main --> Staff : CreateTaskResponse(task_id, result=SUCCESS)
deactivate Main
@enduml
```

---

# 4. 로봇 상태 모니터링 및 관리 (roscar_monitoring_and_management)

```plantuml
@startuml
title 로봇 상태 확인 / 등록 / 삭제 + 위치 실시간 확인 흐름

actor Manager
participant "Manager GUI" as GUI
participant "Main Service" as Main
participant "Mobile Controller" as Mobile
database "RosCars DB" as DB


Manager -> GUI : [메뉴 클릭 or 정보 입력]
activate GUI
GUI -> Main : Request(type, payload)
deactivate GUI

activate Main
alt type == "status"
    Main -> Mobile : GetStatus(roscar_id)
    activate Mobile
    Mobile --> Main : RobotStatus(...)
    deactivate Mobile
    Main --> GUI : RobotStatusResponse(...)

else type == "register"
    Main -> DB : INSERT INTO RosCars(...)
    activate DB
    DB --> Main : OK
    deactivate DB
    Main --> GUI : RegisterRobotResponse(success=true)

else type == "delete"
    Main -> DB : DELETE FROM RosCars WHERE roscar_id=...
    activate DB
    DB --> Main : OK
    deactivate DB
    Main --> GUI : DeleteRobotResponse(success=true)
end
deactivate Main

Manager -> GUI : [위치 확인 탭 클릭]

loop 매 5초마다
    GUI -> Main : RequestRobotLocation(roscar_id)
    activate Main
    Main -> Mobile : GetLocation(roscar_id)
    activate Mobile
    Mobile --> Main : LocationInfo(x, y, z)
    deactivate Mobile
    Main --> GUI : LocationInfo(...)
    deactivate Main
end
@enduml
```

---

# 5. 로그 및 기록 조회 흐름 (log_and_audit_flow)

```plantuml
@startuml
title 로그 및 기록 조회 흐름

actor Manager
participant "Manager GUI" as GUI
participant "Main Service" as Main
database "Log DB (Task, Inventory, RosCar, Delivery)" as DB

loop 각 로그 탭 클릭 시
    Manager -> GUI : 탭 선택 (작업/재고/로봇/출고)
    GUI -> Main : RequestLogs(log_type, 기간, 키워드 등)
    Main -> DB : SELECT * FROM 로그 테이블\nWHERE 조건 (기간, 타입, 키워드)
    DB --> Main : 로그 결과 N건 반환
    Main --> GUI : LogResponse(type, count=N)
    GUI -> GUI : 로그 테이블 렌더링
end

note right of DB
[작업 로그]
event_id=301, task_id=TASK-1001, event=CANCEL  
event_id=302, task_id=TASK-1010, event=CANCEL

[로봇 이벤트 로그]  
event_id=RE-001, roscar_id=RB-01, EMERGENCY_STOP  
event_id=RE-002, roscar_id=RB-03, COLLISION_AVOID
end note

@enduml
```

---

# 6. 로봇 센서 및 상태 보고 흐름 (sensor_and_state_reporting)

```plantuml
@startuml
title 센서 및 상태 보고 흐름
participant "Mobile Controller" as Mobile
participant "Main Service" as Main
participant "Staff PC" as StaffPC
participant "Manager PC" as ManagerPC
database "roscar_status_log" as DB

activate Mobile
Mobile -> Main : 상태 보고\n(SensorData, BatteryStatus, RobotState)\nroscar_id=RB02, task_id=TASK-101,\nbattery=72%, status=DRIVING,\nsensors=OK, pressure=1, time=14:03:22
activate Main

Main -> Main : 내부 상태 업데이트\n(센서, 배터리, 동작)


Main -> DB : INSERT INTO roscar_status_log\n(roscar_id, task_id, battery, status, sensors, time)
DB --> Main : OK


Main -> StaffPC : SendStatusTCP\n(roscar_id=RB02, battery=72%, status=DRIVING, pressure=1)
Main -> ManagerPC : SendStatusROS\n(roscar_id=RB02, sensors=OK, location=x,y)

deactivate Main
deactivate Mobile
@enduml
```

---

# 7. AI 기반 인식 및 추론 흐름 (ai_perception_pipeline)

```plantuml
@startuml
title AI 기반 인식 및 추론 흐름
participant "Video Sender" as Video
participant "Object Detector" as Detector
participant "AI Module" as AI
participant "Main Service" as Main
database "InferenceResult DB" as DB


activate Video
Video -> Detector : MediaFrame(roscar_id=RB02, frame_id=F123, time=14:05:02)
deactivate Video

activate Detector
Detector -> AI : AnalyzeFrame(F123.jpg, meta: roscar_id, time)
deactivate Detector


activate AI
AI -> AI : YOLOv8 추론 실행
AI --> Detector : AIResult(object=Human, confidence=91%)
deactivate AI

activate Detector
Detector -> Main : InferenceResult(roscar_id, object, confidence, location)
deactivate Detector

activate Main
Main -> Main : AnalyzeInferenceResult()

alt object == Human
    Main -> MobileController : EmergencyStop()
else object == 협업로봇
    Main -> MobileController : Pause()
else 기타
    Main -> Main : UpdateTaskPhase()
end

Main -> DB : INSERT INTO InferenceResult(roscar_id, object, confidence, location, time)
deactivate Main
@enduml
```

---

# 8. 로봇 제어 및 동작 흐름 (roscar_control_and_motor_flow)

```plantuml
@startuml
@startuml
title 로봇 제어 및 동작 흐름

participant "Main Service" as Main
participant "Mobile Controller" as Mobile
participant "Motor Controller" as Motor
participant "Staff PC" as Staff
participant "Manager PC" as Manager


Main -> Mobile : MovementCommand(roscar_id=RB02, destination, speed)
Mobile -> Motor : ExecuteMovement(path=[...])


loop 장애물 감지 중
    Motor -> Mobile : ObstacleDetected(distance, direction)
    Mobile -> Main : ObstacleAlert(roscar_id, distance, direction)
    Main -> Mobile : AvoidanceCommand(rotate, direction)
    Mobile -> Motor : ExecuteAvoidance(rotate, direction)
end


Motor -> Mobile : MovementFeedback(position, result)
Mobile -> Main : CommandAck(roscar_id=RB02, result)

alt 이동 성공
    Main -> Main : 완료 처리 및 상태 갱신
else 이동 실패
    par 에러 보고
        Main -> Manager : SendErrorReport(roscar_id, error, time)
        Main -> Staff : SendErrorReport(roscar_id, error, time)
    end
end

@enduml
```

# 9. 재고 파악 모드 및 수동 검수 기능 (inventory_verification_mode)

```plantuml
@startuml
title 재고 파악 모드 및 수동 검수 흐름

actor Manager
participant "Manager GUI" as GUI
participant "Main Service" as Main
participant "Mobile Controller" as Mobile
database "Inventory DB (QR, Inventory, ShoesModel, Log)" as DB


Manager -> GUI : [재고 파악 모드 실행] 클릭
activate GUI
GUI -> Main : StartInventoryCheckMode(triggered_by=admin01)
deactivate GUI

activate Main
Main -> Mobile : ExecuteInventoryScan(mode=AUTO_SCAN)
deactivate Main


activate Mobile
Mobile -> Mobile : Scan QRCode / ArUco ID

Mobile -> DB : SELECT * FROM QRCode\nJOIN Inventory\nJOIN ShoesModel\nWHERE qrcode_data='QR20250430X001'
activate DB
DB --> Mobile : name=Nike Air Zoom, size=270, qty=5, location=R3-S2
deactivate DB

Mobile -> DB : INSERT INTO InventoryEventLog(item_id, qty, roscar_id, timestamp)
activate DB
DB --> Mobile : OK
deactivate DB

Mobile -> Main : InventoryScanResult(name, size, qty, location)
deactivate Mobile

activate Main
Main -> GUI : InventoryScanResult(상세 정보 표시)
deactivate Main

@enduml
```