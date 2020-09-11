from SudokuDLX.SudokuDLX import SudokuDLX

#Easy
board = [[0,2,0,0,7,0,5,6,0],\
    [5,0,7,9,3,2,0,8,0],\
    [0,0,0,0,0,1,0,0,0],\
    [0,1,0,2,4,0,0,5,0],\
    [3,0,8,0,0,0,4,0,2],\
    [0,7,0,0,8,5,0,1,0],\
    [0,0,0,7,0,0,0,0,0],\
    [0,8,0,4,2,3,7,0,1],\
    [0,3,4,0,1,0,0,2,8]]

#HARDEST on earth ?
board = [[8,0,0,0,0,0,0,0,0],\
        [0,0,3,6,0,0,0,0,0],\
        [0,7,0,0,9,0,2,0,0],\
        [0,5,0,0,0,7,0,0,0],\
        [0,0,0,0,4,5,7,0,0],\
        [0,0,0,1,0,0,0,3,0],\
        [0,0,1,0,0,0,0,6,8],\
        [0,0,8,5,0,0,0,1,0],\
        [0,9,0,0,0,0,4,0,0]]

#449 Solutions Possible
#Simple = False to see
board = [[1,2,3,4,5,6,7,8,9],\
    [4,5,6,7,8,9,1,2,3],\
    [7,8,9,1,2,3,4,5,6],\
    [6,0,0,0,0,0,0,0,7],\
    [5,0,0,0,0,0,0,0,4],\
    [9,0,0,0,0,0,0,0,2],\
    [8,0,0,0,0,0,0,0,1],\
    [3,0,0,0,0,0,0,0,8],\
    [2,9,4,8,6,1,3,7,5]]
      
# # Long solve time
# board = [[0,8,0,0,0,7,0,0,9],\
#     [1,0,0,0,0,0,6,0,0],\
#     [0,0,0,3,0,0,0,8,0],\
#     [0,0,2,0,3,0,0,0,7],\
#     [0,0,0,2,1,4,0,0,0],\
#     [5,0,0,0,9,0,4,0,0],\
#     [0,5,0,0,0,3,0,0,0],\
#     [0,0,4,0,0,0,0,0,3],\
#     [6,0,0,1,0,0,0,2,0]]

# # Incorrect
# board = [[1,1,0,0,0,0,0,0,0],\
#     [0,0,0,0,0,0,0,0,0],\
#     [0,0,0,0,0,0,0,0,0],\
#     [0,0,0,0,0,0,0,0,0],\
#     [0,0,0,0,0,0,0,0,0],\
#     [0,0,0,0,0,0,0,0,0],\
#     [0,0,0,0,0,0,0,0,0],\
#     [0,0,0,0,0,0,0,0,0],\
#     [0,0,0,0,0,0,0,0,0]]
      



#If simple, return the first found solution,
#Otherwise print out all solution and return first found
simple = True

s = SudokuDLX()
SudokuDLX.print_solution(board)
solved = s.solve(board,simple)
print(solved)