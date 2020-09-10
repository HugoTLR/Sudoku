from AbstractSudoku import *
from DancingLinks import *
from SolutionHandler import *

class SudokuDLX(AbstractSudokuSolver):
  def __init__(self):
    # super().__init__()
    pass

  def build_sudoku_exact_cover(self,sudoku):
    R = self.sudoku_exact_cover()
    for j in range(1, SudokuDLX.S + 1, 1):
      for i in range(1, SudokuDLX.S + 1, 1):
        n = sudoku[j - 1][i - 1]
        if n != 0:
          for num in range(1, SudokuDLX.S + 1, 1):
            if num != n:
              R[self.get_idx(j, i, num)] = [0 for _ in range(len(R[self.get_idx(j, i, num)]))]
    return R

  def sudoku_exact_cover(self):
    """Return the base exact cover grid for an empty puzzle
    e.g, only base constraint are 1
    """
    rows = 9**3
    cols = (9**2) * 4
    R = [[0 for _ in range(cols)] for _ in range(rows)]

    hBase = 0

    #row-column const
    for r in range(1, SudokuDLX.S + 1, 1):
      for c in range(1, SudokuDLX.S + 1, 1):
        for n in range(1, SudokuDLX.S + 1, 1):
          R[self.get_idx(r, c, n)][hBase] = 1
        hBase += 1

    #row-number const
    for r in range(1, SudokuDLX.S + 1, 1):
      for n in range(1, SudokuDLX.S + 1, 1):
        for c1 in range(1, SudokuDLX.S + 1, 1):
          R[self.get_idx(r, c1, n)][hBase] = 1
        hBase += 1

    #col-number const
    for c in range(1, SudokuDLX.S + 1, 1):
      for n in range(1, SudokuDLX.S + 1, 1):
        for r1 in range(1, SudokuDLX.S + 1, 1):
          R[self.get_idx(r1, c, n)][hBase] = 1
        hBase += 1

    #box-number const
    for br in range(1, SudokuDLX.S + 1, SudokuDLX.side):
      for bc in range(1, SudokuDLX.S + 1, SudokuDLX.side):
        for n in range(1, SudokuDLX.S + 1, 1):
          for rDelta in range(SudokuDLX.side):
            for cDelta in range(SudokuDLX.side):
              R[self.get_idx(br + rDelta, bc + cDelta, n)][hBase] = 1
          hBase += 1

    return R

  def get_idx(self,row,col,num):
    return (row - 1) * (SudokuDLX.S**2) + (col - 1) * SudokuDLX.S + (num - 1)

  def run(self,sudoku):
    cover = self.build_sudoku_exact_cover(sudoku)
    print(len(cover),len(cover[0]))
    dlx = DancingLinks(cover, SudokuHandler(SudokuDLX.S))
    dlx.run()