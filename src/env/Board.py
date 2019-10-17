#################################################################
# Copyright (C)                                                 #
# 2018 - 2019 Li-Han, Chen(eden.chen@bath.edu)                 #
# Permission given to modify the code as long as you keep this  #
# declaration at the top                                        #
#################################################################

import numpy as np
from typing import Dict, Tuple, Sequence
from lib.Config import *
from src.env.BoardSetting import BoardSet

class Board:
  """
  Board with associated board games(Gomoku)

  ...

  Attributes
  ----------
  row
      The number of row in the board
  col
      The number of row in the board
  num_cells
      The total number of cells of the board
  board
      The board for a board game.
  _black/ white_stone
      Black stone is -1; White stone is 1 on the board.
  _empty
      Empty cell is 0.

  Methods
  -------
  board()
      self._board: getter and setter
  cell_is_empty(pos=(0,0))
      Check the cell in the board is empty or not.
  in_the_bound(pos=(-1, 2))
      Check the position is within the bound of the board
  is_full()
      Check the board is full or not.
  place_stone(pos=(0,0), 'X)
      Place stone onto the position.
  print_board()
      Print out the board.
  """
  def __init__(self, row:int, col:int) -> None:
    self._row:int = row
    self._col:int = col
    self._num_cells:int = self._row * self._col
    self._board: Sequence[Sequence[int]] = np.zeros((self._row,self._col), dtype=int)
    self._black_stone = -1
    self._white_stone = 1

    self._empty = 0

  @property
  def board(self) -> Sequence[Sequence[int]]:
    return self._board
  @board.setter
  def board(self, choice:str) -> None:
    bd_creator = BoardSet()
    self._board = None
    bd = np.zeros((self._row,self._col), dtype=int) 

    self._board = bd_creator.reset_choice(choice, bd)

  @property
  def row(self):
    return self._row
  @property
  def col(self):
    return self._col
  @property
  def num_cells(self):
    return self._num_cells

  def cell_is_empty (self, pos:Tuple[int, int]) -> int:
    if (self.in_the_bound(pos) == FAIL):  return ERR_OUT_OF_BOUND
    if self._board[pos] == self._empty: return PASS 
    # This guard might be redundant
    if self._board[pos] != self._black_stone and \
        self._board[pos] != self._white_stone: return ERR_CONTAIN_WEIRD
    return FAIL

  def in_the_bound(self, pos:int) -> int:
    (var_x, var_y) = pos
    if var_x < 0 or var_x >= self._row: return FAIL
    if var_y < 0 or var_y >= self._col: return FAIL
    return PASS

  def is_full(self) -> int:
    Total_number:int = abs(self._board).sum()
    if Total_number == self._num_cells:  return PASS
    return FAIL
  
  def place_stone(self, pos:Tuple[int, int], symbol:str) -> int:
    flag:int = self._white_stone
    if symbol != 'X' and symbol != 'O': return ERR_WRONG_SYMBOL
    if (self.in_the_bound(pos) == FAIL):  return ERR_OUT_OF_BOUND
    
    if symbol == 'X': flag = self._black_stone
    if self.cell_is_empty(pos):
      self._board[pos] = flag
    else:
      return ERR_CONTAIN_WEIRD

    return PASS

  def print_board(self) -> None:
    print(' ',end='  |  ')
    for i in range(self._col):  print(i, end='  |  ')
    print()
    for j in range(self._col+1):
      if j ==self._col: print('----', end='')
      else: print('------', end='')
    print()
    for i in range(self._row):
      print(i, end='  |  ')
      for j in range(self._col):
        if self._board[i][j] != self._black_stone and \
            self._board[i][j] != self._white_stone and \
            self._board[i][j] != 0: assert(False) 
        if self._board[i][j] == 1:  print('O', end='  |  ')
        elif self._board[i][j] == -1:  print('X', end='  |  ')
        else: print(' ', end='  |  ')

      print()
      for j in range(self._col+1):
        if j ==self._col: print('----', end='')
        else: print('------', end='')
      print()