#################################################################
# Copyright (C)                                                 #
# 2018 - 2019 Li-Han, Chen(eden.chen@bath.edu)                  #
# Permission given to modify the code as long as you keep this  #
# declaration at the top                                        #
#################################################################

import numpy as np
from typing import Set, Dict, Tuple, Sequence, List, Any

from src.Player import Player
from src.env.Gomoku import Gomoku
from lib.General import *

class Q_learn(Player):
  """
  Q-learning algorithm
  This algorithm eliminates the symmtery of states and the order of plays
  ...
  Attributes
  ----------
  _row/ _col
      The number of row/ col of a board

  Methods
  -------
  q_table()
      The getter of _q_table
  __init_q_table()
      To initialise the q_table for the learning agent
  create_act_row(board: n-by-n numpy 2D array)
      *** Complexity: O(n^2)

      This method is to search available actions for the specific state.

      Return
        act_row: it is a list of index for the empty locations on the board
        ie. utilising int list is because of that it costs less memory than tuple list.
  convert_board_to_state(board: n-by-n numpy 2D array) 
      *** Complexity: O(n^2)

      The method converts a n by n np array into two arrays of tuples.
      And then, utilising frozenset to build keys for a_table. 
      By doing so, the sequence can be eliminated.
      @ The method could be extended to remove a symmetry. However, it is not finished yet.
      Return
        me_ky(outside): it is the agent set of plays.
        op_ky(inner)  : it is the opponent set of plays.
  """

  def __init__(self, symbol:str, bd_sz:List[int], args:Dict[str, Any]) -> None:
    super().__init__(symbol=symbol)
    self.name    = "Q-learn"
    self.epsilon = args["epsilon"]
    self.gamma   = args["gamma"]
    self.alpha   = args["alpha"]
    self._row    = bd_sz[0]
    self._col    = bd_sz[1]

    self._q_table = dict()
    self._q_table = self.__init_q_table()
    self._prev_state = None
    self._prev_action = None
    self._init = False

  @property
  def q_table(self):
    return self._q_table

  def __init_q_table(self) -> Dict[Any, Dict[Any, List[int]]]:
    board = np.zeros((self._row, self._col))
    me_ky, op_ky = convertBoardToStateKey(board, self.symbol)
    print(me_ky, op_ky)
    return {me_ky:{op_ky:self.create_act_row(board)}}

  # Should be get the available actions
  def create_act_row(self, board: List[ List[int] ]) -> List[int] :
    act_row = {}
    for i in range(self._row):
      for j in range(self._col):
        if board[i][j] == 0:  act_row[i*self._col+j] = 0 
    return act_row

  def check_symmetry(self, me_ky, op_ky):
    # 1. 取得外層的 Dict
    inspect_dict = self._q_table
    # 2. 將me_ky跟op_ky (set)都轉成state ([Tuple[int.int]])
    me_st = list(me_ky)
    op_st = list(op_ky)

    # 3. 確認是否能轉向 
    # Flipping the board through below three corners of the board  
    flip = [(0, self._col-1), (self._row-1, 0), (self._row-1, self._col-1)]
    for flip_elem in flip:
      me_tmp = [ (abs(flip_elem[0]-elem[0]), abs(flip_elem[1]-elem[1])) for elem in me_st ]
      me_ky_tmp = frozenset(me_tmp)
      # 4. 成功確認轉向之後, 內層也要確定存在方能轉向
      if me_ky_tmp in inspect_dict:
        op_tmp = [ (abs(flip_elem[0]-elem[0]), abs(flip_elem[1]-elem[1])) for elem in op_st ]
        op_ky_tmp = frozenset(op_tmp)
        # 5. 如果內層也可以轉向, 則一併回傳 (一定要同時能轉! 不然, 並不是同一張表格)
        if op_ky_tmp in inspect_dict[me_ky_tmp]:  return me_ky_tmp, op_ky_tmp, flip_elem
    
    return me_ky, op_ky, (0,0)
  
  def search_move_w_max_value(self, q_row: Dict[int, float]) -> List[int]:
    init:bool = False
    for indx in q_row:
      if not init:
        actions = [indx]
        max_val = q_row[indx]
        init = True
      else:
        if q_row[indx] > max_val:
          actions = [indx]
          max_val = q_row[indx]
        elif q_row[indx] == max_val:
          actions.append(indx)
      
    
    return actions

  def get_action(self, env: Gomoku) -> Tuple[int, int]:
    me_ky, op_ky = convertBoardToStateKey(env.board, self.symbol)
    me_ky, op_ky, turn_flag = self.check_symmetry(me_ky, op_ky) 
    self._prev_state = [me_ky, op_ky]

    if me_ky not in self._q_table:
      self._q_table[me_ky] = {op_ky:self.create_act_row(env.board)}
    else:
      if op_ky not in self._q_table[me_ky]:
        self._q_table[me_ky][op_ky] = self.create_act_row(env.board)

    intended_move = self.epsilon_greedy([me_ky, op_ky])
    self._prev_action = convertIndexToPosTuple(intended_move, self._row, self._col)
    # return self._prev_action
    return (abs(self._prev_action[0]-turn_flag[0]), abs(self._prev_action[1]-turn_flag[1]))
  
  def epsilon_greedy(self, keys) -> int:
    out_ky, in_ky = keys
    actions = [ ky for ky in self._q_table[out_ky][in_ky] ]
    if np.random.uniform(0, 1) > self.epsilon:
      actions = self.search_move_w_max_value(self._q_table[out_ky][in_ky])

    return np.random.choice(actions)
  
  def learn_from_transition(self, state:Sequence[Sequence[int]], reward:int, terminate:bool) -> None:
    if not self._init:
      self._init = True
      return

    out_ky, in_ky = self._prev_state 
    me_ky, op_ky = convertBoardToStateKey(state, self.symbol)
    # 這裡, 理論上不需要轉向 action. 因為, 這邊只拿最大值的Q
    me_ky, op_ky, _ = self.check_symmetry(me_ky, op_ky) 

    action = self._prev_action[0] * self._row + self._prev_action[1]
    act_row = self.create_act_row(state)

    if len(act_row) == 0:
      self._q_table[out_ky][in_ky][action] = (1-self.alpha) * \
            self._q_table[out_ky][in_ky][action] + self.alpha * (reward)
      return
    else:
      if not terminate and me_ky not in self._q_table:
        self._q_table[me_ky] = {}
      if not terminate and op_ky not in self._q_table[me_ky]:
        self._q_table[me_ky][op_ky] = self.create_act_row(state)

    if terminate:
      self._prev_action = None
      self._prev_state = None
      self._init = False
      # If it is the end, there is no q value from the next state
      max_q_val = 0
    else:
      max_ky = self.search_move_w_max_value(self._q_table[me_ky][op_ky]) 
      max_ky = np.random.choice(max_ky)
      max_q_val = self._q_table[me_ky][op_ky][max_ky]
    # print("Before update ",self._q_table[out_ky][in_ky])
    self._q_table[out_ky][in_ky][action] = (1-self.alpha) * \
          self._q_table[out_ky][in_ky][action] + self.alpha * (reward + self.gamma * max_q_val)
    # print("After update ",self._q_table[out_ky][in_ky])
  
  def print_qTable(self):
    for key in self._q_table:
      print(key)
      print(self._q_table[key])
      print("---")
  def print_qRow(self, me_ky, op_ky):
    print(self._q_table[me_ky])