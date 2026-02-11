# src/gui.py
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os

# 위에서 만든 모듈 임포트
from src.data_loader import DataLoader
from src.predict_lotto import LottoAI

class LottoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI 로또 분석기 V2.0 (Modular)")
        self.root.geometry("600x700")

        # 로직 클래스 인스턴스화
        self.loader = DataLoader()
        self.ai = LottoAI()

        self._init_ui()

    def _init_ui(self):
        """UI 구성 요소 배치"""
        # 1. 상단 (파일)
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack()
        tk.Button(top_frame, text="📂 엑셀 파일 열기", command=self.load_file).pack(side="left", padx=5)
        self.lbl_status = tk.Label(top_frame, text="파일 없음", fg="gray")
        self.lbl_status.pack(side="left")

        # 2. 중단 (버튼)
        mid_frame = tk.Frame(self.root, pady=10)
        mid_frame.pack()
        self.btn_analyze = tk.Button(mid_frame, text="📊 시각화", command=self.show_analysis, state="disabled", bg="#e1f5fe")
        self.btn_analyze.pack(side="left", padx=5)
        self.btn_predict = tk.Button(mid_frame, text="🔮 AI 예측 시작", command=self.start_thread, state="disabled", bg="#e8f5e9")
        self.btn_predict.pack(side="left", padx=5)

        # 3. 하단 (로그)
        self.log_text = scrolledtext.ScrolledText(self.root, height=15, state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)

    def log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if path:
            try:
                self.loader.load_file(path)
                self.lbl_status.config(text="로드 완료", fg="green")
                self.btn_analyze.config(state="normal")
                self.btn_predict.config(state="normal")
                self.log(f"[파일] {os.path.basename(path)} 로드 성공")
            except Exception as e:
                messagebox.showerror("에러", str(e))

    def show_analysis(self):
        """시각화 창 띄우기"""
        win = tk.Toplevel(self.root)
        win.title("분석 결과")
        win.geometry("800x600")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        # 데이터 로더에서 데이터프레임 가져오기
        df = self.loader.df
        all_nums = df[['번호1','번호2','번호3','번호4','번호5','번호6']].values.flatten()
        
        # 그래프 그리기
        sns.histplot(all_nums, bins=45, ax=ax, color='skyblue')
        ax.set_title("번호별 빈도수")
        
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def start_thread(self):
        """스레딩 시작"""
        self.btn_predict.config(state="disabled", text="학습 중...")
        threading.Thread(target=self.run_ai).start()

    def run_ai(self):
        """AI 로직 실행"""
        try:
            self.log(">>> 데이터 전처리 중...")
            oh_data = self.loader.preprocess()
            
            self.log(">>> AI 학습 시작 (잠시 대기)...")
            self.ai.train_model(oh_data)
            self.log(">>> 학습 완료!")
            
            self.log(">>> 번호 생성 중 (필터링 적용)...")
            past_combos = self.loader.get_past_combinations()
            
            # 최근 5주 데이터 가져오기
            last_data = oh_data[-self.ai.window_size:]
            results = self.ai.predict_numbers(last_data, past_combos)
            
            self.log("\n[추천 번호]")
            for i, nums in enumerate(results):
                self.log(f"GAME {i+1}: {nums} (합계: {sum(nums)})")
                
        except Exception as e:
            self.log(f"[에러] {e}")
        finally:
            self.btn_predict.config(state="normal", text="🔮 AI 예측 시작")