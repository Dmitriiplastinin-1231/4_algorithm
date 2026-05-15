import tkinter as tk
from tkinter import ttk

from ui_hamilton import HamiltonTab
from ui_puzzle import PuzzleTab


class LabApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Лабораторная 4: Оптимизационные алгоритмы")
        self.geometry("1100x750")
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)
        notebook.add(HamiltonTab(notebook), text="Гамильтоновы пути")
        notebook.add(PuzzleTab(notebook), text="Пятнашки")
