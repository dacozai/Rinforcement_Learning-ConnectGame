#################################################################
# Copyright (C)                                                 #
# 2018 - 2019 Li-Han, Chen(eden.chen@bath.edu)                  #
# Permission given to modify the code as long as you keep this  #
# declaration at the top                                        #
#################################################################

import numpy as np
import sys
sys.path.append('../..')
from src.env.Board import Board
from lib.Config import *

class Test(Board):
  def __init__(self, t_row, t_col):
    super().__init__(row=t_row, col=t_col)

square_test = Test(t_row=6, t_col=6)
square_test.place_stone((3,2), 'X')
square_test.place_stone((0,2), 'O')
assert(square_test.cell_is_empty((0,2)) == FAIL)
square_test.place_stone((1,3), 'X')
square_test.place_stone((3,5), 'O')
assert(square_test.cell_is_empty((3,5)) == FAIL)

square_test.board = "RESET"
for i in range(square_test._row):
  for j in range(square_test._col):
    assert(square_test._board[i][j] == 0)
    assert(square_test.cell_is_empty((i,j)) == PASS)

square_test.place_stone((0,0), 'O')
assert(square_test.cell_is_empty((0,0)) == FAIL)
assert(square_test.cell_is_empty((3,8)) == ERR_OUT_OF_BOUND)
assert(square_test.cell_is_empty((11111113,8)) == ERR_OUT_OF_BOUND)
assert(square_test.cell_is_empty((-13,-8)) == ERR_OUT_OF_BOUND)
assert(square_test.cell_is_empty((-13,3)) == ERR_OUT_OF_BOUND)

for i in range(square_test._row):
  for j in range(square_test._col):
    assert(square_test.is_full() == FAIL)
    square_test._board[i][j] = np.random.choice([1,-1])
assert(square_test.is_full() == PASS)
# square_test.print_board()
assert(square_test.place_stone((3,6), 'X') == ERR_OUT_OF_BOUND)
assert(square_test.place_stone((3,3), 'W') == ERR_WRONG_SYMBOL)
square_test.board = "RESET"
# square_test.print_board()
assert(square_test.place_stone((0,3), 'O') == PASS)
assert(square_test._board[0][3] == 1)
assert(square_test.place_stone((1,3), 'X') == PASS)
assert(square_test._board[1][3] == -1)

#assert(square_test.cell_is_empty(()))