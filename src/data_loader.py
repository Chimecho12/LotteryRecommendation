import pandas as pd
import numpy as np
import os
import re
import pickle

class DataLoader:
    def __init__(self):
        self.df = None
        self.mode = "lotto"
        self.cols = []
        self.file_path = None # 현재 파일 경로 저장

    def load_file(self, file_path, mode="lotto"):
        """캐시 기능이 추가된 파일 로드 함수"""
        if not os.path.exists(file_path):
            raise FileNotFoundError("파일을 찾을 수 없습니다.")
        
        self.mode = mode
        self.file_path = file_path
        
        # 캐시 파일 경로 생성 (예: data.xlsx -> data.xlsx.pkl)
        cache_path = file_path + f".{mode}.pkl"
        
        # 1. 캐시 유효성 검사
        if os.path.exists(cache_path):
            file_mtime = os.path.getmtime(file_path)
            cache_mtime = os.path.getmtime(cache_path)
            
            if cache_mtime > file_mtime:
                print(f"[시스템] 캐시된 데이터를 불러옵니다: {os.path.basename(cache_path)}")
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.df = cached_data['df']
                    self.cols = cached_data['cols']
                    return self.df

        # 2. 신규 분석 (캐시가 없거나 수정된 흔적이 있는 경우) 
        print("[시스템] 데이터를 새로 분석합니다...")
        ext = os.path.splitext(file_path)[-1].lower()

        try:
            if ext == '.csv':
                try:
                    self.df = self._read_csv_auto_header(file_path, encoding='utf-8')
                except:
                    self.df = self._read_csv_auto_header(file_path, encoding='cp949')
            else:
                self.df = self._read_excel_auto_header(file_path)
        except Exception as e:
            raise Exception(f"파일 읽기 오류: {e}")

        # 모드별 처리 (로또 / 연금복권)
        if self.mode == "lotto":
            self.cols = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
            self._normalize_columns()
            self._analyze_lotto_features()
        else: 
            if '구분' in self.df.columns:
                self.df = self.df[self.df['구분'].astype(str).str.contains('1등')].copy()
            
            rename_map = {
                '조단위': '조', '조': '조',
                '십만': '번호1', '만': '번호2', '천': '번호3', 
                '백': '번호4', '십': '번호5', '일': '번호6',
                '첫번째': '번호1', '두번째': '번호2', '세번째': '번호3',
                '네번째': '번호4', ' 다섯번째': '번호5', '여섯번째': '번호6'
            }
            self.df.rename(columns=rename_map, inplace=True)
            self.cols = ['조', '번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
            
            for col in self.cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype(str).apply(self._extract_number)
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0).astype(int)
                else:
                    self.df[col] = 0

            self._analyze_pension_features()
            
        # 분석 완료된 데이터 캐시 저장
        with open(cache_path, 'wb') as f:
            pickle.dump({'df': self.df, 'cols': self.cols}, f)
        print("[시스템] 분석 결과가 캐시로 저장되었습니다.")
            
        return self.df
    
    def _read_csv_auto_header(self, path, encoding='utf-8'):
        preview = pd.read_csv(path, encoding=encoding, nrows=5, header=None)
        header_idx = 0
        for i, row in preview.iterrows():
            row_str = " ".join(row.astype(str).values)
            if '회차' in row_str or '조단위' in row_str or '당첨번호' in row_str:
                header_idx = i; break
        return pd.read_csv(path, encoding=encoding, header=header_idx)

    def _read_excel_auto_header(self, path):
        preview = pd.read_excel(path, nrows=5, header=None)
        header_idx = 0
        for i, row in preview.iterrows():
            row_str = " ".join(row.astype(str).values)
            if '회차' in row_str or '조단위' in row_str:
                header_idx = i; break
        return pd.read_excel(path, header=header_idx)

    def _extract_number(self, val):
        numbers = re.findall(r'\d+', str(val))
        return numbers[0] if numbers else '0'

    def _normalize_columns(self):
        self.df.columns = self.df.columns.str.replace(' ', '').str.strip()

    def _analyze_lotto_features(self):
        num_cols = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
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
        else: 
            if not all(col in self.df.columns for col in self.cols): return None
            data = self.df[self.cols].values.astype(float)
            data[:, 0] = data[:, 0] / 5.0      
            data[:, 1:] = data[:, 1:] / 9.0    
            return data

    def get_past_combinations(self):
        past = set()
        cols = self.cols if self.mode == "pension" else ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
        if all(c in self.df.columns for c in cols):
            for _, row in self.df[cols].iterrows():
                past.add(tuple(row.values))
        return past