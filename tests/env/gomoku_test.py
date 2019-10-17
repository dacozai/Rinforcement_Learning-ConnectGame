#################################################################
# Copyright (C)                                                 #
# 2018 - 2019 Li-Han, Chen(eden.chen@bath.edu)                  #
# Permission given to modify the code as long as you keep this  #
# declaration at the top                                        #
#################################################################

import numpy as np
import sys
sys.path.append('../..')
from src.env.Gomoku import Gomoku
from lib.Config import *

class Test_g(Gomoku):
  def __init__(self, test_row, test_col, t_rule=0):
    super().__init__(bd_sz=[test_row, test_col], rule=t_rule)

sz = 6
gomoku = Test_g(sz, sz)
gomoku.now_player = 'O'
gomoku.is_win()
"""
Vertical Test

1 0 0 0 0 0   1 0 0 0 0 0   1 0 0 0 0 0 
1 0 0 0 0 0   1 0 0 0 0 0   1 0 0 0 0 1
1 0 0 0 0 0   1 0 0 0 0 0   1 0 0 0 0 1
1 0 0 0 0 0   1 0 0 0 0 0   1 0 0 0 0 1
1 0 0 0 0 0   0 0 0 0 0 0   0 0 0 0 0 1
0 0 0 0 0 0   1 0 0 0 0 0   1 0 0 0 0 1
"""
for i in range(sz -1):  gomoku.place_stone((i, 0),'O')
assert(gomoku.straight_exam([1, 0]))
gomoku._board[4][0] = 0
assert(not gomoku.straight_exam([1, 0]))
gomoku.place_stone((5,0), 'O')
assert(not gomoku.straight_exam([1, 0]))
for i in range(1, sz):  gomoku.place_stone((i, 5),'O')
assert(gomoku.straight_exam([1, 0]))
gomoku.board = "RESET"

"""
Horizontal Test

1 1 1 1 0 0   1 1 1 1 0 1   1 1 1 1 0 1 
1 0 0 0 0 0   1 0 0 0 0 0   1 0 0 0 0 0
1 0 0 0 0 0   1 0 0 0 0 0   1 0 0 0 0 0
1 0 0 0 0 0   1 0 0 0 0 0   1 1 1 1 1 1
1 0 0 0 0 0   1 0 0 0 0 0   1 0 0 0 0 0
0 0 0 0 0 0   0 0 0 0 0 0   0 0 0 0 0 0
"""
for i in range(sz -1):  gomoku.place_stone((i, 0),'O')
for i in range(sz -2):  gomoku.place_stone((0, i),'O')
assert(not gomoku.straight_exam([0, 1]))
gomoku.place_stone((0,5), 'O')
assert(not gomoku.straight_exam([0, 1]))
for i in range(sz):  gomoku.place_stone((3, i),'O')
assert(gomoku.straight_exam([0, 1]))

del gomoku
sz = 7
gomoku = Test_g(sz, sz, 0)
gomoku.now_player = 'O'
"""
Diagonal Test

0 0 0 0 0 0 0   0 0 0 0 0 0 0
0 1 0 0 0 0 0   0 0 0 0 0 0 0
0 0 1 0 0 0 0   0 0 0 0 0 0 0
0 0 0 1 0 0 0   1 0 0 0 0 0 0
0 0 0 0 1 0 0   0 1 0 0 0 0 0
0 0 0 0 0 1 0   0 0 1 0 0 0 0
0 0 0 0 0 0 0   0 0 0 1 1 0 0
"""
# Case 1
for i in range(sz-2):  gomoku.place_stone((1+i, 1+i),'O')
assert(gomoku.diagonal_exam([1, 1]))
gomoku._board[5][5] = 0
gomoku.place_stone((6,6),'O')
assert(not gomoku.diagonal_exam([1, 1]))
gomoku.place_stone((5,5),'O')
assert(gomoku.diagonal_exam([1, 1]))

# Case 2 (include of testing that conect to the next diagonal line)
gomoku.board = "RESET"
for i in range(sz-2):  gomoku.place_stone((3+i, 0+i),'O')
gomoku.place_stone((6,4),'O')
assert(not gomoku.diagonal_exam([1, 1]))
gomoku.place_stone((6,2),'O')
assert(not gomoku.diagonal_exam([1, 1]))
gomoku.place_stone((4,0),'O')
gomoku.place_stone((5,1),'O')
assert(not gomoku.diagonal_exam([1, 1]))
gomoku.place_stone((2,0),'O')
assert(not gomoku.diagonal_exam([1, 1]))

# Case Study

gomoku.print_board()
