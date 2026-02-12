import pandas as pd
import numpy as np
import os
import re

class DataLoader:
    def __init__(self):
        self.df = None
        self.mode = "lotto" # "lotto" 또는 "pension"
        self.cols = []

    def load_file(self, file_path, mode="lotto"):
        """파일 확장자와 내용을 분석하여 똑똑하게 데이터를 로드합니다."""
        if not os.path.exists(file_path):
            raise FileNotFoundError("파일을 찾을 수 없습니다.")
        
        self.mode = mode
        ext = os.path.splitext(file_path)[-1].lower()

        # 1. 파일 읽기 (헤더 위치 자동 탐색)
        try:
            if ext == '.csv':
                # 인코딩 문제 방지를 위해 여러 인코딩 시도
                try:
                    self.df = self._read_csv_auto_header(file_path, encoding='utf-8')
                except:
                    self.df = self._read_csv_auto_header(file_path, encoding='cp949')
            else:
                self.df = self._read_excel_auto_header(file_path)
        except Exception as e:
            raise Exception(f"파일을 읽는 중 오류가 발생했습니다: {e}")

        # 2. 모드별 데이터 처리
        if self.mode == "lotto":
            # [로또 6/45]
            self.cols = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
            # 컬럼명 매핑 (유연하게 대응)
            self._normalize_columns()
            self._analyze_lotto_features()
            
        else: 
            # [연금복권 720+]
            # 1등 당첨 내역만 필터링 ('구분' 컬럼이 있는 경우)
            if '구분' in self.df.columns:
                self.df = self.df[self.df['구분'].astype(str).str.contains('1등')].copy()
            
            # 컬럼명 매핑 (조단위 -> 조, 십만 -> 번호1 ...)
            rename_map = {
                '조단위': '조', '조': '조',
                '십만': '번호1', '만': '번호2', '천': '번호3', 
                '백': '번호4', '십': '번호5', '일': '번호6',
                '첫번째': '번호1', '두번째': '번호2', '세번째': '번호3',
                '네번째': '번호4', ' 다섯번째': '번호5', '여섯번째': '번호6'
            }
            self.df.rename(columns=rename_map, inplace=True)
            
            # 필요한 컬럼만 선택
            self.cols = ['조', '번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
            
            # [중요] 데이터 정제 (숫자가 아닌 문자 제거)
            for col in self.cols:
                if col in self.df.columns:
                    # '1조', '2회' 등에서 숫자만 추출
                    self.df[col] = self.df[col].astype(str).apply(self._extract_number)
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)
                else:
                    print(f"[경고] '{col}' 컬럼을 찾을 수 없습니다. (데이터 구조 확인 필요)")
                    self.df[col] = 0 # 에러 방지용 0 채움

            self._analyze_pension_features()
            
        return self.df

    def _read_csv_auto_header(self, path, encoding='utf-8'):
        """CSV 파일에서 '회차'나 '조단위'가 있는 행을 헤더로 찾습니다."""
        # 먼저 앞부분 5줄만 읽어봄
        preview = pd.read_csv(path, encoding=encoding, nrows=5, header=None)
        header_idx = 0
        
        for i, row in preview.iterrows():
            row_str = " ".join(row.astype(str).values)
            if '회차' in row_str or '조단위' in row_str or '당첨번호' in row_str:
                header_idx = i
                break
        
        return pd.read_csv(path, encoding=encoding, header=header_idx)

    def _read_excel_auto_header(self, path):
        """엑셀 파일에서 헤더 행을 찾습니다."""
        preview = pd.read_excel(path, nrows=5, header=None)
        header_idx = 0
        
        for i, row in preview.iterrows():
            row_str = " ".join(row.astype(str).values)
            if '회차' in row_str or '조단위' in row_str:
                header_idx = i
                break
                
        return pd.read_excel(path, header=header_idx)

    def _extract_number(self, val):
        """문자열에서 숫자만 추출합니다 (예: '4조' -> '4')"""
        numbers = re.findall(r'\d+', str(val))
        return numbers[0] if numbers else '0'

    def _normalize_columns(self):
        """컬럼명을 표준화합니다."""
        # 공백 제거
        self.df.columns = self.df.columns.str.replace(' ', '').str.strip()
        # 필요한 경우 추가 매핑 로직

    def _analyze_lotto_features(self):
        # 기존 로또 분석 코드 (동일)
        num_cols = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
        # 컬럼이 존재하는지 확인 후 계산
        valid_cols = [c for c in num_cols if c in self.df.columns]
        if len(valid_cols) < 6: return 

        self.df['총합'] = self.df[valid_cols].sum(axis=1)
        self.df['홀수개수'] = self.df[valid_cols].apply(lambda x: sum(n % 2 for n in x), axis=1)
        self.df['홀짝비율'] = self.df['홀수개수'].astype(str) + ":" + (6 - self.df['홀수개수']).astype(str)
        
        self.df['고숫자개수'] = self.df[valid_cols].apply(lambda x: sum(1 for n in x if n >= 23), axis=1)
        self.df['고저비율'] = (6 - self.df['고숫자개수']).astype(str) + ":" + self.df['고숫자개수'].astype(str)

        def get_ac(row):
            nums = sorted(list(row))
            diffs = set(nums[j] - nums[i] for i in range(6) for j in range(i+1, 6))
            return len(diffs) - 5
        self.df['AC값'] = self.df[valid_cols].apply(get_ac, axis=1)

    def _analyze_pension_features(self):
        # 연금복권 분석 코드 (동일)
        digit_cols = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
        valid_cols = [c for c in digit_cols if c in self.df.columns]
        if len(valid_cols) < 6: return

        self.df['숫자합'] = self.df[valid_cols].sum(axis=1)
        self.df['홀수개수'] = self.df[valid_cols].apply(lambda x: sum(n % 2 for n in x), axis=1)
        self.df['홀짝비율'] = self.df['홀수개수'].astype(str) + ":" + (6 - self.df['홀수개수']).astype(str)
        self.df['고숫자개수'] = self.df[valid_cols].apply(lambda x: sum(1 for n in x if n >= 5), axis=1)
        self.df['고저비율'] = (6 - self.df['고숫자개수']).astype(str) + ":" + self.df['고숫자개수'].astype(str)

    def preprocess(self):
        if self.df is None: return None
        
        if self.mode == "lotto":
            # 로또 처리
            cols = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
            if not all(col in self.df.columns for col in cols): return None
            
            numbers = self.df[cols].values
            oh_list = []
            for row in numbers:
                vec = np.zeros(45)
                for n in row: 
                    if 1 <= int(n) <= 45: vec[int(n)-1] = 1
                oh_list.append(vec)
            return np.array(oh_list)
            
        else: # pension
            # 연금복권 처리
            if not all(col in self.df.columns for col in self.cols): return None
            
            data = self.df[self.cols].values.astype(float)
            # 정규화
            data[:, 0] = data[:, 0] / 5.0      # 조
            data[:, 1:] = data[:, 1:] / 9.0    # 숫자
            return data

    def get_past_combinations(self):
        past = set()
        # 컬럼이 있는지 확인
        cols = self.cols if self.mode == "pension" else ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
        if all(c in self.df.columns for c in cols):
            for _, row in self.df[cols].iterrows():
                past.add(tuple(row.values))
        return past