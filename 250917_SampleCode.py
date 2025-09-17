import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import glob

# polyCode.txt 파일 읽기 (행정동 코드/이름 정보)
code = pd.read_csv('polyCode.txt', sep="|")

# 컬럼 값에 포함된 ` 기호 제거
code['polycode'] = code['`polycode`'].str.replace('`', '', regex=False)
code['name'] = code['`name`'].str.replace('`', '', regex=False)
code['full_name'] = code['`full_name`'].str.replace('`', '', regex=False)

# polycode → full_name 매핑 딕셔너리 생성
codeDict = dict(zip(code['polycode'], code['full_name']))

# 입력/출력 폴더 지정
input_folder = 'datas/kt_move/202507'
output_folder = 'output_data_temp'
os.makedirs(output_folder, exist_ok=True)   # 출력 폴더 없으면 생성

# 입력 폴더 안의 모든 txt 파일 경로 가져오기
file_list = glob.glob(os.path.join(input_folder, "*.txt"))

for file_path in file_list:
    file_name = os.path.basename(file_path)   # 파일 이름만 추출

    # txt 파일 읽기
    df = pd.read_csv(file_path, sep='|')

    # 컬럼명 정리: 알파벳/언더바(_) 이외의 문자 제거
    df.columns = [re.sub(r'[^a-zA-Z_]', '', col) for col in df.columns]

    # 출발지/도착지 코드를 문자열로 변환
    df['start_place_cd'] = df['start_place_cd'].astype(str)
    df['arv_place_cd'] = df['arv_place_cd'].astype(str)

    # 코드값을 행정동 풀네임으로 매핑
    df['start_place_cd'] = df['start_place_cd'].map(codeDict)
    df['arv_place_cd'] = df['arv_place_cd'].map(codeDict)
    
    # 출발/도착 시간 datetime 변환
    df['start_dt'] = pd.to_datetime(df['start_dt'])
    df['move_st_hr'] = df['start_dt'].dt.floor("h")   # 출발 시간을 '시' 단위로 내림
    df['arv_dt'] = pd.to_datetime(df['arv_dt'])
    df['move_arv_hr'] = df['arv_dt'].dt.floor("h")   # 도착 시간을 '시' 단위로 내림
    
    # (필터링용 코드 — 현재는 주석 처리됨)
    # arange_datas = df[['move_st_hr', 'move_arv_hr', 'start_place_cd', 'arv_place_cd', 'sex_nm', 'age_grp','popl_cnt']]
    # mask = (arange_datas['move_st_hr'].dt.hour >= 11) & (arange_datas['move_st_hr'].dt.hour <= 16) \
    #      | (arange_datas['move_arv_hr'].dt.hour >= 11) & (arange_datas['move_arv_hr'].dt.hour <= 16)
    # arange_datas = arange_datas[mask]

    # 시간대·연령대별 출발 인구수 합계
    start_df_time = df.groupby(['move_st_hr', 'age_grp'])['popl_cnt'].sum().reset_index()
    # 시간대·연령대별 도착 인구수 합계
    arv_df_time = df.groupby(['move_arv_hr', 'age_grp'])['popl_cnt'].sum().reset_index()

    # 행정동 단위 집계 (출발 기준)
    start_df_time_dong = df.groupby(['start_dt', 'age_grp', '행정동 컬럼명'])['popl_cnt'].sum().reset_index()
    # 행정동 단위 집계 (도착 기준)
    arv_df_time_dong = df.groupby(['arv_dt', 'age_grp', '행정동 컬럼명'])['popl_cnt'].sum().reset_index()

    # 연령대(10살 단위) 변환
    df['age_grp'] = (df['agegrd_nm'] // 10) * 10
    
    # 저장할 파일명 정의
    file_name1 = str(file_name)+" start_time".replace('.txt', '.csv')
    file_name2 = str(file_name)+" arv_time".replace('.txt', '.csv')
    file_name3 = str(file_name)+" start_dong_time".replace('.txt', '.csv')
    file_name4 = str(file_name)+" arv_dong_time".replace('.txt', '.csv')

    # 저장 경로 설정
    output_path1 = os.path.join(output_folder, file_name1)
    output_path2 = os.path.join(output_folder, file_name2)
    output_path3 = os.path.join(output_folder, file_name3)
    output_path4 = os.path.join(output_folder, file_name4)

    # 각각 다른 집계 데이터를 CSV로 저장
    start_df_time.to_csv(output_path1, index=False, encoding='euc-kr')
    arv_df_time.to_csv(output_path2, index=False, encoding='euc-kr')
    start_df_time_dong.to_csv(output_path3, index=False, encoding='euc-kr')
    arv_df_time_dong.to_csv(output_path4, index=False, encoding='euc-kr')
