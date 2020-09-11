from .SolutionHandler import DefaultHandler, SudokuHandler
from sys import maxsize
from .type_checking import Column, DNode, Grid, Parent


class DancingLinks:
  verbose = True
  def __init__(self,grid : Grid, handler=DefaultHandler()):
    self.header = self.build_DLX_board(grid)
    self.solutions = 0
    self.updates = 0
    self.handler = handler
    self.answer = []

  #Main alg
  def search(self, k : int, single : bool = False):
    if self.header.right == self.header:
      print(f"-----------------------------------------\nSolution # {self.solutions}\n") if DancingLinks.verbose else None
      self.handler.handle_solution(self.answer)
      print(f"-----------------------------------------") if DancingLinks.verbose else None
      self.solutions += 1
    else:
      c = self.select_column_node_heuristic()
      c.cover(self)

      r = c.down
      while r != c:
        self.answer.append(r)
        j = r.right
        while j != r:
          j.column.cover(self)
          j = j.right
        if single and self.solutions > 0:
          break
        self.search(k+1, single)
        r = self.answer.pop(-1)
        c = r.column

        j = r.left
        while j != r:
          j.column.uncover(self)
          j = j.left

        r = r.down
      c.uncover(self)

  def select_column_node_heuristic(self) -> Column:
    mini = maxsize
    ret = None
    c = self.header.right
    while c != self.header:
      if c.size < mini:
        mini = c.size
        ret = c
      c = c.right
    return ret

  def print_board(self):
    print("Board config: ")
    tmp = self.header.right
    while tmp != self.header:
      d = tmp.down
      while d != tmp:
        ret = f"{d.column.name} --> "

        i = d.right
        while i != d:
          ret = ret + f"{i.column.name} --> "
          i = i.right
        print(ret)
        d = d.down
      tmp = tmp.right

  def build_DLX_board(self, grid : Grid) -> Column:
    COLS = len(grid[0])
    ROWS = len(grid)

    header = ColumnNode("header")
    column_nodes = []
    for i in range(COLS):
      n = ColumnNode(str(i))
      column_nodes.append(n)
      header = header.hook_right(n)

    header = header.right.column

    for j in range(ROWS):
      prev = None
      for i in range(COLS):
        if grid[j][i] == 1:
          col = column_nodes[i]
          new_node = DancingNode(col)
          if prev is None:
            prev = new_node
          col.up.hook_down(new_node)
          prev = prev.hook_right(new_node)
          col.size += 1
    header.size = COLS
    return header

  def run(self,single : bool = False):
    self.solutions = 0
    self.updates = 0
    self.answer = []
    self.search(0, single)
    # print(f"{self.solutions=} {self.updates=}")
    # print(f"{self.handler.results[0]=}")


class DancingNode:
  def __init__(self,c : Column = None ):
    self.left = self
    self.right = self
    self.up = self
    self.down = self
    self.column = c

  def hook_down(self, n1 : DNode) -> DNode:
    assert self.column == n1.column, "WTF ?"
    n1.down = self.down
    n1.down.up = n1
    n1.up = self
    self.down = n1
    return n1

  def hook_right(self, n1 : DNode) -> DNode:
    n1.right = self.right
    n1.right.left = n1
    n1.left = self
    self.right = n1
    return n1

  def unlink_left_right(self,parent : Parent):
    self.left.right = self.right
    self.right.left = self.left
    parent.updates += 1

  def link_left_right(self,parent : Parent):
    self.left.right = self.right.left = self
    parent.updates += 1

  def unlink_up_down(self,parent : Parent):
    self.up.down = self.down
    self.down.up = self.up
    parent.updates += 1

  def link_up_down(self,parent : Parent):
    self.up.down = self.down.up = self
    parent.updates += 1


class ColumnNode(DancingNode):
  def __init__(self,n : str):
    super().__init__(self)
    self.size = 0
    self.name = n

  
  def cover(self,parent : Parent):
    self.unlink_left_right(parent)

    i = self.down
    while i != self:
      j = i.right
      while j != i:
        j.unlink_up_down(parent)
        j.column.size -= 1
        j = j.right
      i = i.down
    parent.header.size -= 1

  def uncover(self,parent : Parent):
    i = self.up
    while i != self:
      j = i.left
      while j != i:
        j.column.size += 1
        j.link_up_down(parent)
        j = j.left
      i = i.up
    self.link_left_right(parent)
    parent.header.size += 1