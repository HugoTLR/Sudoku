from math import sqrt
from .type_checking import Grid

class AbstractSudokuSolver:
  S = 9
  side = 3
  def __init__(self):
    pass

  def run(self,sudoku : Grid, single : bool = False) -> bool:
    raise NotImplementedError("We should never call the abstract class")

  def solve(self,sudoku : Grid, single : bool = False) -> tuple:
    if not self.is_valid(sudoku):
      print("Error: Invalid board game")
      return False, None

    AbstractSudokuSolver.S = len(sudoku)
    AbstractSudokuSolver.side = int( sqrt(AbstractSudokuSolver.S) ) 
    result = self.run(sudoku, single)
    if result is None:
      print("Error: Invalid detected board game")
      return False, None
    return True, result

  def is_valid(self,grid : Grid) -> bool:
    N = len(grid)

    if N != 9:
      return False #We support only basic sudoku
    for j in range(N):
      if len(grid[j]) != N:
        return False
      for i in range(len(grid[j])):
        if not (j >= 0 and j <= N): #ERROR POTENTIAL j -> grid[j][i]
          return False

    b = [False]*(N+1)

    #ROW VALIDATION
    for j in range(N):
      for i in range(N):
        if grid[j][i] == 0:
          continue
        if b[grid[j][i]]:
          return False
        b[grid[j][i]] = True
      b = [False]*(N+1)

    #COL VALIDATION
    for j in range(N):
      for i in range(N):
        if grid[i][j] == 0:
          continue
        if b[grid[i][j]]:
          return False
        b[grid[i][j]] = True
      b = [False]*(N+1)

    side = int(sqrt(N))

    #BLOCK VALIDATION
    for j in range(0,N,side):
      for i in range(0,N,side):
        for d1 in range(side):
          for d2 in range(side):
            if grid[j + d1][i + d2] == 0:
              continue
            if b[grid[j + d1][i + d2]]:
              return False
            b[grid[j + d1][i + d2]] = True

        b = [False]*(N+1)

    return True

  def print_solution(solution : Grid) -> bool:
      N = len(solution)
      res = ""
      for j,row in enumerate(solution):
        if j%3 == 0:
          res = res = res + "+-------+-------+-------+\n"

        for i, col in enumerate(row):
          if col != 0:tmp = f" {col}"
          else:tmp = f"  " 

          if i%3 == 0:
            tmp = "|" + tmp
          elif i%3 == 2:
            tmp = tmp + " "
          res = res + tmp
        res = res + "|\n"
      res = res + "+-------+-------+-------+\n"
      print(res)