# main.py
import tkinter as tk
from src.gui import LottoApp

if __name__ == "__main__":
    root = tk.Tk()
    app = LottoApp(root)
    root.mainloop()