# 연금복권 역대 당첨번호 표 정리 사이트 크롤링 (걍 단순 실험)

import requests
from bs4 import BeautifulSoup
import pandas as pd

def save_tistory_table_to_excel(url, output_file):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"접속 에러: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table')
    
    if not table:
        print("페이지 내에서 표(table)를 찾을 수 없습니다.")
        return

    try:
        df_list = pd.read_html(str(table))
        if df_list:
            df = df_list[0]
            
            # 3. 엑셀 파일로 저장
            df.to_excel(output_file, index=False)
            print(f"성공! 표 데이터를 '{output_file}'에 저장했습니다.")
            print(df.head()) # 상위 5개 데이터 미리보기
        else:
            print("표 데이터를 파싱하는 데 실패했습니다.")
    except Exception as e:
        print(f"데이터 변환 중 오류 발생: {e}")

# 실행
target_url = "https://signalfire85.tistory.com/277"
output_filename = "pension_lotto.xlsx"

save_tistory_table_to_excel(target_url, output_filename)