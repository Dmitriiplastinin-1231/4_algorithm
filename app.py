import tkinter as tk
from tkinter import ttk

from ui_hamilton import HamiltonTab
from ui_puzzle import PuzzleTab


class LabApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lab 4: Optimization Algorithms")
        self.geometry("1100x750")
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)
        notebook.add(HamiltonTab(notebook), text="Hamilton Paths")
        notebook.add(PuzzleTab(notebook), text="15-Puzzle")
