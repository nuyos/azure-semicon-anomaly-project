# producer_streaming.py
import os
import time
import uuid
import pandas as pd
import numpy as np
import json
from datetime import datetime
from azure.eventhub import EventHubProducerClient, EventData

# --- 1. Azure Event Hubs 설정 ---
# (보안을 위해 실제 값은 코드에 하드코딩하지 않고, 환경변수에서 불러옵니다)
EVENTHUB_CONN_STR = os.environ.get("EVENTHUB_CONNECTION_STRING")
EVENTHUB_NAME = os.environ.get("EVENTHUB_NAME")

# --- 2. 시뮬레이션 설정 ---
SECOM_DATA_PATH = "assets/uci-secom.csv" # 590개 센서 데이터 파일
SECOM_LABELS_PATH = "assets/secom_labels.data" # Pass/Fail, Timestamp 데이터 파일

# 1초당 생성할 총 메시지 수 (시뮬레이션 속도 조절)
# 예: 100이면 A, B라인 합쳐 초당 200건 전송
MESSAGES_PER_SECOND_PER_LINE = 15 
SIMULATION_LINES = ['A-Line', 'B-Line']   # 가상 공정 라인

def load_base_data():
    """
    assets/ 폴더의 파일 2개를 로드하고 병합하여
    '원본 딕셔너리 리스트'를 반환합니다.
    """
    print("SECOM 원본 데이터 로드 중...")
    try:
        # 1. 레이블 파일 로드
        # - 구분자: 공백
        # - 헤더: 없음
        labels_df = pd.read_csv(
            SECOM_LABELS_PATH, 
            sep=" ",  # 공백으로 분리됨
            header=None, 
            names=["Pass/Fail", "Timestamp_Str"]
        )
        
        # 2. 센서 데이터 로드
        # - 구분자: 콤마
        # - 헤더: 있음 (0번째 줄)
        sensors_df = pd.read_csv(
            SECOM_DATA_PATH, 
            sep=",",
            header=0  # 0번째 줄이 헤더임
        )
        
        # 3. 데이터 전처리 및 병합
        
        # (A) sensors_df에서 불필요한 'Time' 컬럼 제거
        #     (labels_df의 타임스탬프를 기준으로 할 것이므로)
        if 'Time' in sensors_df.columns:
            sensors_df = sensors_df.drop(columns=["Time"])
        
        # (B) sensors_df의 컬럼명을 '0', '1'... -> 'Sensor_1', 'Sensor_2'...로 변경
        #     (컬럼 개수가 590개라고 가정)
        sensors_df.columns = [f"Sensor_{i+1}" for i in range(sensors_df.shape[1])]
        
        # 4. (기존 로직) 두 데이터프레임을 side-by-side로 합치기
        #    (두 파일의 행 순서가 동일하다고 가정)
        base_df = pd.concat([labels_df, sensors_df], axis=1)
        
        # 5. (이하 기존 코드와 동일)
        
        # 원본 타임스탬프는 사용하지 않음 (실시간으로 생성)
        base_df = base_df.drop(columns=["Timestamp_Str"])
        
        # JSON 변환을 위해 pandas의 NaN을 Python의 None(Null)으로 변경
        base_df = base_df.astype(object).where(pd.notnull(base_df), None)
        
        # 성능 최적화를 위해 딕셔너리 리스트로 변환
        base_records = base_df.to_dict('records')
        
        print(f"원본 데이터 로드 완료. (총 {len(base_records)}개 레코드)")
        return base_records

    except FileNotFoundError:
        print(f"[오류] '{SECOM_DATA_PATH}' 또는 '{SECOM_LABELS_PATH}' 파일을 찾을 수 없습니다.")
        print("assets 폴더에 파일이 정확히 있는지 확인해주세요.")
        return None
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        return None

def create_simulated_event(base_records, line_id):
    """
    원본 레코드 중 하나를 무작위로 선택하고 '뻥튀기'(Augmentation)하여
    실시간 이벤트(JSON 문자열)를 1건 생성합니다.
    """
    
    # 1. 원본 데이터에서 무작위 샘플 1건 추출 (매우 빠름)
    record_index = np.random.randint(0, len(base_records))
    record = base_records[record_index].copy() # 원본 수정을 막기 위해 .copy()
    
    # 2. '뻥튀기' (Augmentation) - 실시간 데이터처럼 보이게 가공
    
    #   A. 센서 값에 약간의 노이즈(Noise) 추가
    for key, value in record.items():
        if key.startswith("Sensor_") and value is not None:
            # 원본 값의 0.1% 수준에서 랜덤 노이즈 추가
            noise = np.random.normal(0, abs(value * 0.001) + 1e-6) 
            record[key] = value + noise
            
    #   B. 새로운 메타데이터 추가 (가장 중요)
    record['Wafer_ID'] = str(uuid.uuid4()) # 웨이퍼 고유 ID
    record['Line_ID'] = line_id            # 공정 라인 (A or B)
    record['Event_Time'] = datetime.utcnow().isoformat() # 이벤트 발생 시간 (UTC 표준)
    
    # 3. Python 딕셔너리를 JSON 문자열로 변환
    return json.dumps(record)

# --- 4. 메인 실행 로직 ---
def main():
    print("--- 반도체 공정 실시간 스트리밍 시뮬레이터 시작 ---")
    
    # 1. 환경변수 체크
    if not EVENTHUB_CONN_STR or not EVENTHUB_NAME:
        print("[오류] 환경변수 'EVENTHUB_CONNECTION_STRING'와 'EVENTHUB_NAME'이 설정되지 않았습니다.")
        print("스크립트 실행 전 환경변수를 설정해주세요.")
        return

    # 2. 기본 데이터 로드
    base_data = load_base_data()
    if base_data is None:
        return

    # 3. Event Hubs 프로듀서 클라이언트 생성
    try:
        producer_client = EventHubProducerClient.from_connection_string(
            conn_str=EVENTHUB_CONN_STR,
            eventhub_name=EVENTHUB_NAME
        )
    except Exception as e:
        print(f"Event Hubs 클라이언트 연결 실패: {e}")
        return

    print(f"Event Hubs 연결 성공. [{EVENTHUB_NAME}]")
    total_messages_per_second = MESSAGES_PER_SECOND_PER_LINE * len(SIMULATION_LINES)
    print(f"초당 약 {total_messages_per_second}건의 메시지 전송을 시작합니다. (Ctrl+C로 중지)")

    # 4. 무한 루프: 실시간 메시지 전송
    try:
        while True:
            start_time = time.time() # 1초 간격 조절용
            
            # [성능 최적화]
            # 메시지를 1건씩 보내는 것(send_event)은 매우 비효율적입니다.
            # '배치(Batch)'로 묶어서 한 번에 보내야 합니다.
            event_data_batch = producer_client.create_batch()
            
            msg_count = 0
            for _ in range(MESSAGES_PER_SECOND_PER_LINE):
                for line in SIMULATION_LINES:
                    # A. 가상 이벤트(JSON) 생성
                    event_json = create_simulated_event(base_data, line)
                    
                    # B. 배치에 추가
                    try:
                        event_data_batch.add(EventData(event_json))
                        msg_count += 1
                    except ValueError:
                        # 배치가 가득 차면(약 1MB), 일단 보내고 새 배치를 만듭니다.
                        producer_client.send_batch(event_data_batch)
                        print(f"  > 배치 Full. {msg_count}건 중간 전송.")
                        event_data_batch = producer_client.create_batch()
                        event_data_batch.add(EventData(event_json))
                        msg_count = 1 # 카운터 초기화

            # C. 1초 동안 모인 남은 배치를 전송합니다.
            if len(event_data_batch) > 0:
                producer_client.send_batch(event_data_batch)

            # D. 1초 간격 맞추기
            # 루프가 1초보다 빨리 끝났으면, 남은 시간만큼 대기합니다.
            time_to_sleep = 1.0 - (time.time() - start_time)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg_count}건 전송 완료.")
            
    except KeyboardInterrupt:
        print("\n전송 중지 요청. 스크립트를 종료합니다.")
    except Exception as e:
        print(f"\n[오류] 전송 중 예외 발생: {e}")
    finally:
        # 5. 클라이언트 종료 (필수)
        print("Event Hubs 프로듀서 클라이언트를 닫습니다.")
        producer_client.close()

if __name__ == "__main__":
    main()