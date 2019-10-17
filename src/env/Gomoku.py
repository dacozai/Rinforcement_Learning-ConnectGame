#################################################################
# Copyright (C)                                                 #
# 2018 - 2019 Li-Han, Chen(eden.chen@bath.edu)                  #
# Permission given to modify the code as long as you keep this  #
# declaration at the top                                        #
#################################################################

import numpy as np
from typing import Sequence, Tuple, TypeVar, List
from src.env.Board import Board
from lib.Config import *

class Gomoku(Board):
  """
  Gomoku Board Game Environment
  ...

  Attributes
  ----------
  __win_cond
      The winning condition. This is a winning condition is set as 5 because this is Gomoku.
  __rule
      Decide which rule that players are going to play
      0: Free-style (Black stone always win)
      1: Renju (Japan Style) 
  __reward
    1 : win.
    0 : draw
    -1: loss 
  _now_player
    This attribute is for the game to verify the current player who is going to place a stone.

  Methods
  -------
  now_player()
      self._now_player: getter and setter
  is_win()
      To determine that the player is win or not
  free_cond()
    This method is to verify the winning of Free-Style Gomoku. 
  straight_exam(movement=[1,0])
    This method check x in a row in a straight line(Vertical/ Horizontal line.) 
    The idea here is that passing movement variable to control the moving direction.
    (0,1) is Horizontal; (1,0) is Vertical.
  diagonal_exam(movement=[1,1])
    This method check x in a row in a diagonal line(Forward/ Backward). 
    The idea here is that passing movement variable to control the moving direction.
    (1,1) is Backward; (1,-1) is Forward.
  check_connect(start_point=[0,0],movement=[0,1]) 
    This method is fed by the start_point and the moving directio(movement)
    to check whether a player connect N or not.
  gomoku_respond(move=(0,0), symbol='O')
    This method is design to be a respond to a reinforcement learning agetn.

    Return
      Env: self.board.
      Reward: the reward after placing the stone.
      Terminate flag: to indicate that the game is over or not.
      Winner: to show who wins the game.
  decide_side()
    This method helps to decide the side at the beginning 

  """
  def __init__(self, bd_sz:List[int]=[15,15], win_cond = 5,rule:int = 0, reward:int = 1) -> None:
    super().__init__(row=bd_sz[0], col=bd_sz[1])
    self.__win_cond = win_cond
    self.__rule = rule
    self.__reward = reward
    # Black stone plays first
    self._now_player = -1

  @property 
  def now_player(self) -> int:
    return self._now_player 

  @now_player.setter
  def now_player(self, player_name:str) -> None:
    if player_name == 'X':
      self._now_player = self._black_stone
    else:
      self._now_player = self._white_stone

  # 這邊需要做修改, 未來如果要加入"連珠"的話
  def is_win(self):
    if self.free_cond():  return True
    return False

  def free_cond(self) -> int:
    return self.straight_exam([0, 1]) or \
            self.straight_exam([1, 0]) or \
            self.diagonal_exam([1, 1]) or \
            self.diagonal_exam([1, -1])

  def straight_exam(self, movement: List[int]) -> bool:
    susp_axis_increment = np.array([1, 1]) - np.array(movement)
    init = np.array([0, 0])
    while init.max() < self._row:
      if self.check_connect(init, movement) is PASS:  
        return True
      init += susp_axis_increment

    return False

  def diagonal_exam(self, movement: List[int]) -> bool:
    susp_axis_sets: Sequence[Sequence[int]] = []
    for i in range(self._row):
      susp_axis_sets.append([0, i])

    flag:int = int((1 - movement[1]) / 2)
    for i in range(1, self._row):
      susp_axis_sets.append([i, flag*(self._row - 1)])

    for tup in susp_axis_sets:
      if self.check_connect(tup, movement) is PASS:
        return True
    
    return False
    
  def check_connect(self, start_point: List[int], movement: List[int]) -> int:
    num_in_row:int = 0
    susp_row:int = start_point[0]
    susp_col:int = start_point[1]
    while self.in_the_bound( (susp_row, susp_col) ) == PASS:
      assert(susp_row < self._row)
      assert(susp_col < self._col)
      if self._board[susp_row][susp_col] == self._now_player: 
        num_in_row += 1
      else: 
        num_in_row = 0

      if num_in_row > self.__win_cond - 1:
        return PASS

      susp_row += movement[0]
      susp_col += movement[1]
    
    return FAIL

  def gomoku_respond(self, move: Tuple[int, int], symbol: str) -> [Sequence[Sequence[int]], int, bool,str]: 
    if not self.cell_is_empty(move):  return self._board, -1, True, 'Your opponent'
    self.place_stone(move, symbol)
    self._now_player = 1
    if symbol == 'X':
      self._now_player = -1
    
    if self.is_win(): return self._board, self.__reward, True, symbol
    if self.is_full(): return self._board, 0, True,'nobody'
    return self._board, 0, False, None
  
  def decide_side(self):
    if np.random.randint(100) > 49:
      return 'X'
    return 'O'
    











