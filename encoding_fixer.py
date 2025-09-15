import pandas as pd
import os

# 원본 CSV 파일이 있는 폴더 경로 (수정 필요)
# 예: r"C:\Users\YourName\Documents\Original_CSVs"
input_folder = r"C:/Users/asus/DMU/BigData_Campus/Sample_Data/csv/"

# 인코딩을 변경하여 새로 저장할 폴더 경로 (수정 필요)
# 예: r"C:\Users\YourName\Documents\UTF8_CSVs"
output_folder = r"C:/Users/asus/DMU/BigData_Campus/Sample_Data/csv_utf8/"

# 저장할 폴더가 없으면 자동으로 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"'{output_folder}' 폴더를 생성했습니다.")

# 입력 폴더 내 모든 파일에 대해 작업 시작
try:
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            try:
                # CSV 파일을 읽습니다.
                # 한글 깨짐의 주원인인 'cp949' 또는 'euc-kr' 인코딩으로 먼저 읽기를 시도합니다.
                # 만약 다른 인코딩이라면 이 부분을 수정해야 할 수 있습니다.
                df = pd.read_csv(input_path, encoding='cp949')

                # 'UTF-8-SIG' 인코딩으로 다시 저장합니다.
                # index=False는 불필요한 행 번호가 추가되는 것을 방지합니다.
                df.to_csv(output_path, index=False, encoding='utf-8-sig')

                print(f"✅ '{file_name}' 인코딩 변경 완료")

            except UnicodeDecodeError:
                # 만약 'cp949'로도 파일을 읽을 수 없다면, 다른 인코딩일 수 있습니다.
                # 'utf-8'로 다시 시도해봅니다.
                try:
                    df = pd.read_csv(input_path, encoding='utf-8')
                    df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    print(f"✅ '{file_name}' (UTF-8 to UTF-8-SIG) 변경 완료")
                except Exception as e:
                    print(f"❌ '{file_name}' 파일을 읽는 중 오류 발생: {e}")
            except Exception as e:
                print(f"❌ '{file_name}' 처리 중 오류 발생: {e}")

except FileNotFoundError:
    print(f"오류: '{input_folder}' 경로를 찾을 수 없습니다. 경로를 확인해주세요.")

print("\n모든 작업이 완료되었습니다.")