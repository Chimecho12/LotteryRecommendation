# main.py
from src.gui import LottoApp

if __name__ == "__main__":
    # CustomTkinter는 root를 스스로 생성하므로 
    # tk.Tk()를 만들어서 넘겨줄 필요가 없습니다.
    app = LottoApp()
    app.mainloop()