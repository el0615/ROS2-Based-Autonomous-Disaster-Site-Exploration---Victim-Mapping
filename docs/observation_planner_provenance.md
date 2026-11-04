# Code Provenance — observation_goal_planner

## 1. 원 출처와 참고 범위

- Fraunhofer IPA, `ipa320/ipa_coverage_planning`, `noetic_dev`, `convex_sensor_placement_explorator.cpp`:
  https://raw.githubusercontent.com/ipa320/ipa_coverage_planning/noetic_dev/ipa_room_exploration/common/src/convex_sensor_placement_explorator.cpp
  참고: 공간 후보, 센서 자세, 셀-후보 가시성 관계로 관측 위치를 평가하는 아이디어.
- ETH ASL, `ethz-asl/nbvplanner`, README:
  https://raw.githubusercontent.com/ethz-asl/nbvplanner/master/README.md
  참고: 센서 관측이 갱신되면 남은 관측 이득을 고려해 다음 관측 위치를 재평가하는 아이디어.
- ROS2 Humble 공식 API:
  https://raw.githubusercontent.com/ros2/geometry2/humble/tf2_ros_py/tf2_ros/buffer.py
  https://raw.githubusercontent.com/ros2/geometry2/humble/tf2_ros_py/tf2_ros/transform_listener.py
  https://raw.githubusercontent.com/ros2/common_interfaces/humble/visualization_msgs/msg/Marker.msg
  https://raw.githubusercontent.com/ros2/common_interfaces/humble/nav_msgs/msg/OccupancyGrid.msg
  https://raw.githubusercontent.com/ros2/rclpy/humble/rclpy/rclpy/impl/rcutils_logger.py

참고 소스의 확인일: 2026-10-30. 이 문서는 저장소 유지보수 상태나 설치 호환성을 보증하지 않는다.

## 2. 직접 구현·복사·재사용 분류

신규 Python 파일 두 개와 신규 테스트 두 개는 이번 프로젝트를 위해 작성했다. IPA/ETH의 C++ 코드, 최적화 솔버, ROS1 노드, ROS2 포팅 코드는 이 ZIP에 복사하거나 포함하지 않았다.

기존 사용자 프로젝트의 `hospital_coverage_core.Geometry`, `load_bundle`, `planar_pose`와 `hospital_zone_visualizer.verify_map`을 import해 재사용한다. 해당 원본 파일은 수정하거나 새 ZIP에서 덮어쓰지 않는다. 벡터 DDA는 기존 `Coverage.trace`와 같은 셀·모서리 규칙을 따르도록 새로 작성했으며, 기존 관측 코드와 수치 비교 테스트를 넣었다.

NumPy, SciPy, ROS2/tf2/Marker API는 설치된 의존성으로 호출한다.

## 3. 원 알고리즘과 다른 부분

- IPA의 convex relaxation, 정수계획, set-cover 최적화, 방 전체 TSP 방문 경로는 구현하지 않았다.
- ETH NBV의 3D OctoMap/RRT 재계획 트리는 구현하지 않았다.
- 단일 2D 방과 주변 복도에서 타일별 관측 후보를 샘플링하고, 한 번에 다음 위치 하나를 평가하는 근사적 NBV형 정책이다.
- 원본 지도의 Unknown 대신 사용자의 **이번 세션에 아직 관측하지 않은 구역 셀**을 정보 이득 대상으로 삼는다.
- 실제 관측률은 기존 Coverage에서만 읽는다. 예측 결과는 원래 관측 마스크에 쓰지 않는다.
- Nav2의 실제 경로 계산·Goal 전송은 이번 구현에 포함하지 않았다. scipy 그래프상의 경로 추정과 최신 전체 global costmap의 제한적 장애물 검사만 한다.

## 4. 변경 이유

사용자 환경은 ROS2 Humble, 2D LiDAR, J100, 사전 병원 지도와 승인된 27구역이다. 완전 미지 환경 탐색이나 바닥 전수 주행보다, 기존 방을 실제로 얼마나 다시 관측했는지 판단하는 기능이 핵심이다. 원본 코드를 직접 포팅하기 전에 후보 생성과 실제 Coverage 연동을 분리해 검증하고, 실패 시 원래 Nav2/Coverage를 유지하기 위해 독립 표시 노드로 구현했다.

## 5. 검증 및 재현

`--self-test`: 30개 단위 테스트와 승인된 27구역 데이터 검사, 실제 원본 지도 기반 R14-T 합성 시나리오 후보 생성. 별도 strict mock ROS 테스트에서는 노드 생성, 입력 타임스탬프/세션 매칭, TF 대기, INFO/WARN 호출 위치, 마커/JSON, stale/시각역행/지도변경, 실제 90% 게이트를 확인한다.

`results/observation_planner/offline_reference_examples.json`의 좌표와 관측 마스크는 **오프라인 합성 조건**이다. 사용자 현재 위치·실측 세션 결과가 아니다.

실제 ROS2 Humble, DDS, tf2 전송, Gazebo 주행 및 신체 정지 검증은 제작 환경에서 수행하지 않았다. Python 3.10 문법 적합성은 AST로 확인했고 테스트 실행 인터프리터는 검증 JSON에 기재했다.

## 6. 라이선스·책임 범위 및 후속 작업

이번 산출물은 IPA/ETH의 코드나 라이선스를 재배포하지 않는다. 향후 그 코드 자체를 복사·수정·배포한다면 각 저장소 LICENSE와 헤더, IPA의 이용조건을 별도로 검토하고 출처를 유지해야 한다. 이번 결과를 ‘IPA 공식 구현 적용 완료’로 기록해서는 안 된다.

예측 가시성·보수적 격자 연결성은 실제 안전/도달 가능성의 증명이 아니다. 후속은 Nav2 ComputePathToPose 검증, 실제 관측 이득 비교, 실패/진전없음 관리, 90% 후 취소와 정지 검증, 방간 방문 순서 통합이다. LLM, 피해자 수색 완료, 원본 벽 삭제는 포함하지 않는다.
