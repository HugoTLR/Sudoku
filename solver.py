import time

class Sudoku:
  def __init__(self):
    self.backtrack_index = 0 #Index for backtracking algorithm

  def __str__(self):
    res = ""
    for row in self.board:
      res = res + "".join([str(r) for r in row]) +"\n"
    return res[:-1] # We skip the last \n

  def getRow(self,pos):return [self.board[pos[0]][x] for x in range(9)]

  def getCol(self,pos):return [self.board[y][pos[1]] for y in range(9)]

  def getSquareBound(self,idx):
    if idx % 3 == 0: return idx, idx+2
    elif idx % 3 == 1: return idx-1, idx+1
    else: return idx-2, idx

  def getSquare(self,pos):
    lY,hY = self.getSquareBound(pos[0])
    lX,hX = self.getSquareBound(pos[1])    
    square = []
    for j in range(lY,hY+1,1):
      for i in range(lX,hX+1,1):
        square.append(self.board[j][i])
    return square

  def initialize_solved_board(self,board):

    self.board = board
    self.solve_cells = []
    for j,row in enumerate(self.board):
      for i,value in enumerate(row):
        self.solve_cells.append(Cell(j,i,value,self))

  def solve(self):
    start = time.time()
    while self.backtrack_index < len(self.solve_cells):
      self.move_forward()
      cell = self.set_cell_value()
      cell.increment()
      self.change_cells(cell)

    print(f"Solved Sudoku in {time.time()-start:.3f} seconds")
  def move_forward(self):
    while self.backtrack_index < len(self.solve_cells) -1 and self.solve_cells[self.backtrack_index].solved:
      self.backtrack_index += 1

  def set_cell_value(self):
    cell = self.solve_cells[self.backtrack_index]
    cell.set_value()
    return cell

  def change_cells(self,cell):
    if cell.is_valid_number():
      self.backtrack_index += 1
    else:
      self.decrement_cell(cell)

  def decrement_cell(self,cell):
    while cell.current_test_index == len(cell.potential_values) - 1:
      self.reset_cell(cell)
      self.backtrack()
      cell = self.solve_cells[self.backtrack_index]
    cell.current_test_index += 1

  def reset_cell(self,cell):
    cell.current_test_index = 0
    cell.value = 0
    self.board[cell.row][cell.col] = 0

  def backtrack(self):
    self.backtrack_index -= 1
    while self.solve_cells[self.backtrack_index].solved:
      self.backtrack_index -= 1

class Cell:
  def __init__(self,row,col,val,game):
    self.row = row
    self.col = col
    self.pos = (self.row,self.col)
    self.value = val
    self.solved = True if self.value > 0 else False           
    self.current_test_index = 0                     #Index of the value we testin'
    self.game = game                          #Parent link
    self.potential_values = set(range(1,10)) if not self.solved else [] #potential values for the cell

    if not self.solved:
      self.findPotentialValues()

  # Find all potential values for this cell
  def findPotentialValues(self):
    for item in self.game.getRow(self.pos) + self.game.getCol(self.pos) + self.game.getSquare(self.pos):
      if item in self.potential_values:
        self.potential_values.remove(item)
    self.potential_values = list(self.potential_values)
    self.handle_unique_value()


  # If only one value possible, set it, and mark the cell as solved
  def handle_unique_value(self):
    if len(self.potential_values) == 1:
      self.set_value()
      self.solved = True

  def set_value(self):
    if not self.solved:
      self.value = self.potential_values[self.current_test_index]
      self.game.board[self.row][self.col] = self.value


  def increment(self):
    while not self.is_valid_number() and self.current_test_index < len(self.potential_values) - 1:
      self.current_test_index += 1
      self.set_value()

  def is_valid_number(self):
    """checks to see if the current number is valid in its row, column, and box"""
    for condition in [self.game.getRow(self.pos), self.game.getCol(self.pos), self.game.getSquare(self.pos)]:
      if not self.check_alignement_condition(condition):
        return False
    return True

  def check_alignement_condition(self,condition):
    values = [value for value in condition if value != 0]
    return len(values) == len(set(values)) # Set doesn't take double into account, so if we have 2 '1' on the same row, comparison will be false

if __name__ == "__main__":
  sudoku = Sudoku()
  board = [[1,2,0,0,7,0,5,6,0],\
      [5,0,7,9,3,2,0,8,0],\
      [0,0,0,0,0,1,0,0,0],\
      [0,1,0,2,4,0,0,5,0],\
      [3,0,8,0,0,0,4,0,2],\
      [0,7,0,0,8,5,0,1,0],\
      [0,0,0,7,0,0,0,0,0],\
      [0,8,0,4,2,3,7,0,1],\
      [0,3,4,0,1,0,0,2,8]]
  sudoku.initialize_solved_board(board)

  sudoku.solve()

  print(sudoku.__str__())