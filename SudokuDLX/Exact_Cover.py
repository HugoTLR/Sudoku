import numpy as np 



class Sudoku:
  def __init__(self,SIZE,board):
    self.SIZE = SIZE
    self.N = self.SIZE ** 2
    self.board = board
    self.show()


  def show(self):
    res = ""
    for j,row in enumerate(self.board):
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

    


board = [[0,8,0,0,0,7,0,0,9],\
    [1,0,0,0,0,0,6,0,0],\
    [0,0,0,3,0,0,0,8,0],\
    [0,0,2,0,3,0,0,0,7],\
    [0,0,0,2,1,4,0,0,0],\
    [5,0,0,0,9,0,4,0,0],\
    [0,5,0,0,0,3,0,0,0],\
    [0,0,4,0,0,0,0,0,3],\
    [6,0,0,1,0,0,0,2,0]]

s = Sudoku(3,board)
s.run()