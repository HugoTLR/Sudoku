from math import sqrt

class AbstractSudokuSolver:
  S = 9
  side = 3
  def __init__(self):
    pass

  def run(self,sudoku):
    raise NotImplementedError("We should never call the abstract class")

  def solve(self,sudoku):
    if not self.is_valid(sudoku):
      print("Error: Invalid board game")
      return False

    AbstractSudokuSolver.S = len(sudoku)
    AbstractSudokuSolver.side = int( sqrt(AbstractSudokuSolver.S) ) 
    self.run(sudoku)
    return True


  def print_solution(solution):
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

  # def print_solution(solution):
  #   N = len(solution)
  #   for j in range(N):
  #     ret = ""
  #     for i in range(N):
  #       ret = ret + f"{solution[j][i]} "
  #     print(ret)
  #   print()

  def is_valid(self,grid):
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

# #INVALID BOARD
# board = [[0,9,5,8,0,0,0,0,0],\
#         [8,0,0,3,0,0,6,0,0],\
#         [7,0,0,0,2,0,0,3,0],\
#         [6,8,0,0,0,0,0,0,0],\
#         [0,0,2,0,0,0,5,0,0],\
#         [0,0,0,0,0,0,0,1,6],\
#         [0,4,0,0,3,0,0,0,7],\
#         [0,0,6,0,0,4,4,0,5],\
#         [0,0,0,0,0,9,2,9,0]]

#VALID BOARD
# board = [[0,8,0,0,0,7,0,0,9],\
#     [1,0,0,0,0,0,6,0,0],\
#     [0,0,0,3,0,0,0,8,0],\
#     [0,0,2,0,3,0,0,0,7],\
#     [0,0,0,2,1,4,0,0,0],\
#     [5,0,0,0,9,0,4,0,0],\
#     [0,5,0,0,0,3,0,0,0],\
#     [0,0,4,0,0,0,0,0,3],\
#     [6,0,0,1,0,0,0,2,0]]
# print(AbstractSudokuSolver.is_valid(board))