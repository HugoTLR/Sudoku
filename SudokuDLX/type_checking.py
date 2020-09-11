from typing import List, TypeVar

# Solution List
Answer = TypeVar('A')

# ColumnNode Object
Column = TypeVar('C')

# DancingNode Object
DNode = TypeVar('D')

# SudokuBoard or sparse matrix for DLX
Grid = List[List[int]]

# DancingLinks object
Parent = TypeVar('P')