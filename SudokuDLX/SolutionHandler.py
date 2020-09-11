from .AbstractSudoku import AbstractSudokuSolver
from .type_checking import Answer, Grid

class SudokuHandler:
  def __init__(self,board_size : int):
    self.size = board_size
    self.results = []

  def handle_solution(self,answer : Answer):
    result = self.parse_board(answer)
    AbstractSudokuSolver.print_solution(result)
    self.results.append(result)
    

  def parse_board(self,answer : Answer) -> Grid:
    result = [[0 for _ in range(self.size)] for _ in range(self.size)]

    for node in answer:
      rc_node = node
      mini = int( rc_node.column.name )

      tmp = node.right
      while tmp != node:
        val = int(tmp.column.name)
        if val < mini:
          mini = val
          rc_node = tmp
        tmp = tmp.right

      ans1 = int(rc_node.column.name)
      ans2 = int(rc_node.right.column.name)
      r = ans1 // self.size
      c = ans1 % self.size
      num = (ans2 % self.size) + 1
      result[r][c] = num
    return result

class DefaultHandler:
  def __init__(self):
    pass

  def handle_solution(self,answer : Answer):
    for node in answer:
      ret = f"{node.column.name} "
      tmp = node.right
      while tmp != node:
        ret = ret + f"{tmp.column.name} "
        tmp = tmp.right
      print(ret)

