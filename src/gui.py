# src/gui.py
import customtkinter as ctk  # tkinter 대신 이거 사용
import tkinter as tk         # 파일 다이얼로그 등 일부 기능용
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os

# 기존 로직 모듈 임포트 (수정 불필요)
from src.data_loader import DataLoader
from src.predict_lotto import LottoAI

# 기본 테마 설정 (시스템 설정 따라감, 혹은 "Dark", "Light")
ctk.set_appearance_mode("System") 
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

class LottoApp(ctk.CTk):  # tk.Tk 대신 ctk.CTk 상속
    def __init__(self):
        super().__init__()
        
        # 윈도우 설정
        self.title("Recommend Lottery")
        self.geometry("700x800")
        
        # 로직 클래스 연결
        self.loader = DataLoader()
        self.ai = LottoAI()

        self._init_ui()

    def _init_ui(self):
        # 그리드 레이아웃 설정 (반응형)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1) # 로그 창 부분이 늘어나도록

        # === 1. 상단 타이틀 및 파일 로드 ===
        self.header_frame = ctk.CTkFrame(self, corner_radius=10)
        self.header_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

        self.lbl_title = ctk.CTkLabel(self.header_frame, text="로또 추첨", font=ctk.CTkFont(size=24, weight="bold"))
        self.lbl_title.pack(pady=10)

        self.btn_file = ctk.CTkButton(self.header_frame, text="📂 엑셀 데이터 불러오기", command=self.load_file, height=40)
        self.btn_file.pack(padx=20, pady=(0, 10), fill="x")

        self.lbl_status = ctk.CTkLabel(self.header_frame, text="현재 로드된 파일 없음", text_color="gray")
        self.lbl_status.pack(pady=(0, 10))

        # === 2. 기능 버튼 영역 ===
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.grid(row=1, column=0, padx=20, pady=0, sticky="ew")
        self.btn_frame.grid_columnconfigure(0, weight=1)
        self.btn_frame.grid_columnconfigure(1, weight=1)

        # 분석 버튼 (색상: Teal)
        self.btn_analyze = ctk.CTkButton(self.btn_frame, text="📊 데이터 시각화", command=self.show_analysis, 
                                         state="disabled", fg_color="#00897B", hover_color="#00695C", height=50)
        self.btn_analyze.grid(row=0, column=0, padx=(0, 10), sticky="ew")

        # 예측 버튼 (색상: Indigo)
        self.btn_predict = ctk.CTkButton(self.btn_frame, text="🔮 AI 예측 시작", command=self.start_thread, 
                                         state="disabled", fg_color="#3949AB", hover_color="#283593", height=50)
        self.btn_predict.grid(row=0, column=1, padx=(10, 0), sticky="ew")

        # === 3. 로그 및 결과 창 ===
        self.log_textbox = ctk.CTkTextbox(self, corner_radius=10, font=("Consolas", 14))
        self.log_textbox.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")
        self.log_textbox.insert("0.0", "프로그램이 준비되었습니다.\n엑셀 파일을 로드해주세요.\n")
        self.log_textbox.configure(state="disabled")

    def log(self, msg):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", msg + "\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if path:
            try:
                self.loader.load_file(path)
                self.lbl_status.configure(text=f"로드 완료: {os.path.basename(path)}", text_color="#66BB6A") # 초록색
                self.btn_analyze.configure(state="normal")
                self.btn_predict.configure(state="normal")
                self.log(f"[시스템] 데이터 로드 성공!")
            except Exception as e:
                self.log(f"[에러] {e}")

    def show_analysis(self):
        # 분석 창도 CustomTkinter로 (CTkToplevel)
        win = ctk.CTkToplevel(self)
        win.title("분석 리포트")
        win.geometry("900x600")
        
        # Matplotlib 다크모드 대응
        plt.style.use('dark_background') # 차트도 어둡게 (원하면 'default'로 변경)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        df = self.loader.df
        all_nums = df[['번호1','번호2','번호3','번호4','번호5','번호6']].values.flatten()
        
        # 색상 커스텀
        ax.hist(all_nums, bins=45, color='#4FC3F7', edgecolor='black', alpha=0.7)
        ax.set_title("Lotto Number Frequency", color="white")
        ax.tick_params(colors='white')
        
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def start_thread(self):
        self.btn_predict.configure(state="disabled", text="AI 학습 중... (대기)")
        threading.Thread(target=self.run_ai).start()

    def run_ai(self):
        try:
            self.log("\n>>> 데이터 전처리 및 학습 시작...")
            oh_data = self.loader.preprocess()
            
            # 진행상황을 보여주기 위해 학습
            self.ai.train_model(oh_data)
            self.log(">>> 학습 완료! 번호 생성 중...")
            
            past_combos = self.loader.get_past_combinations()
            last_data = oh_data[-self.ai.window_size:]
            results = self.ai.predict_numbers(last_data, past_combos)
            
            self.log("\n============================")
            self.log("   추천 번호 (Top 5)   ")
            self.log("============================")
            for i, nums in enumerate(results):
                self.log(f" GAME {i+1}: {nums} (합계: {sum(nums)})")
            self.log("============================")
                
        except Exception as e:
            self.log(f"[치명적 오류] {e}")
        finally:
            self.btn_predict.configure(state="normal", text="🔮 AI 예측 시작")