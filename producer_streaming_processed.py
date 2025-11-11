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
SECOM_DATA_PATH = "data/processed/secom_named.csv"  # 전처리된 데이터 파일

# 1초당 생성할 총 메시지 수 (시뮬레이션 속도 조절)
MESSAGES_PER_SECOND_PER_LINE = 15 
SIMULATION_LINES = ['A-Line', 'B-Line']   # 가상 공정 라인

def load_base_data():
    """
    전처리된 CSV 파일을 로드하여 딕셔너리 리스트로 반환
    """
    print("전처리된 SECOM 데이터 로드 중...")
    try:
        # 전처리된 CSV 로드 (헤더 포함)
        df = pd.read_csv(SECOM_DATA_PATH)
        
        print(f"로드된 데이터 shape: {df.shape}")
        print(f"컬럼 목록 (처음 10개): {list(df.columns[:10])}")
        
        # NaN을 None으로 변환
        df = df.astype(object).where(pd.notnull(df), None)
        
        # 딕셔너리 리스트로 변환
        base_records = df.to_dict('records')
        
        print(f"데이터 로드 완료. (총 {len(base_records)}개 레코드)")
        print(f"샘플 레코드 키: {list(base_records[0].keys())[:10]}...")
        
        return base_records

    except FileNotFoundError:
        print(f"[오류] '{SECOM_DATA_PATH}' 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_simulated_event(base_records, line_id):
    """
    원본 레코드를 기반으로 실시간 이벤트 생성
    """
    # 무작위 샘플 선택
    record_index = np.random.randint(0, len(base_records))
    record = base_records[record_index].copy()
    
    # 센서 값에 노이즈 추가
    for key, value in record.items():
        if key.startswith("sensor_") and value is not None:
            try:
                noise = np.random.normal(0, abs(float(value) * 0.001) + 1e-6)
                record[key] = float(value) + noise
            except (ValueError, TypeError):
                pass  # 변환 불가능한 값은 그대로 유지
    
    # 메타데이터 추가
    record['Wafer_ID'] = str(uuid.uuid4())
    record['Line_ID'] = line_id
    record['Event_Time'] = datetime.utcnow().isoformat()
    
    return json.dumps(record, ensure_ascii=False)

# --- 4. 메인 실행 로직 ---
def main():
    print("=" * 70)
    print("  반도체 공정 실시간 스트리밍 시뮬레이터 (전처리 데이터 사용)")
    print("=" * 70)
    
    # 1. 환경변수 체크
    if not EVENTHUB_CONN_STR or not EVENTHUB_NAME:
        print("\n[오류] Event Hubs 환경변수가 설정되지 않았습니다.")
        print("  - EVENTHUB_CONNECTION_STRING")
        print("  - EVENTHUB_NAME")
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
        print(f" Event Hubs 연결 성공. [{EVENTHUB_NAME}]")
    except Exception as e:
        print(f" Event Hubs 클라이언트 연결 실패: {e}")
        return

    total_messages_per_second = MESSAGES_PER_SECOND_PER_LINE * len(SIMULATION_LINES)
    print(f" 초당 약 {total_messages_per_second}건의 메시지 전송을 시작합니다.")
    print(f"   ({MESSAGES_PER_SECOND_PER_LINE}건/초 × {len(SIMULATION_LINES)}개 라인)")
    print(" 중지: Ctrl+C\n")
    print("-" * 70)

    # 4. 무한 루프: 실시간 메시지 전송
    try:
        while True:
            start_time = time.time()
            
            # 배치로 묶어서 전송 (성능 최적화)
            event_data_batch = producer_client.create_batch()
            
            msg_count = 0
            for _ in range(MESSAGES_PER_SECOND_PER_LINE):
                for line in SIMULATION_LINES:
                    # 가상 이벤트(JSON) 생성
                    event_json = create_simulated_event(base_data, line)
                    
                    # 배치에 추가
                    try:
                        event_data_batch.add(EventData(event_json))
                        msg_count += 1
                    except ValueError:
                        # 배치가 가득 차면 전송 후 새 배치 생성
                        producer_client.send_batch(event_data_batch)
                        print(f"  📦 배치 Full. {msg_count}건 중간 전송.")
                        event_data_batch = producer_client.create_batch()
                        event_data_batch.add(EventData(event_json))
                        msg_count = 1

            # 남은 배치 전송
            if len(event_data_batch) > 0:
                producer_client.send_batch(event_data_batch)

            # 1초 간격 유지
            time_to_sleep = 1.0 - (time.time() - start_time)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg_count}건 전송 완료")
            
    except KeyboardInterrupt:
        print("\n  전송 중지 요청. 스크립트를 종료합니다.")
    except Exception as e:
        print(f" [오류] 전송 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 5. 클라이언트 종료 (필수)
        print("\n🔌 Event Hubs 프로듀서 클라이언트를 닫습니다.")
        producer_client.close()
        print("\n" + "=" * 70)
        print("  프로그램 종료")
        print("=" * 70)

if __name__ == "__main__":
    main()