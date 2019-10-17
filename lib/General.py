#################################################################
# Copyright (C)                                                 #
# 2018 - 2019 Li-Han, Chen(eden.chen@bath.edu)                  #
# Permission given to modify the code as long as you keep this  #
# declaration at the top                                        #
#################################################################

import numpy as np
import copy
import sys
import copy
import matplotlib.pyplot as plt
from collections import namedtuple
from typing import Set, Dict, Tuple, Sequence, List, Any


def convertBoardToStateKey(board, symbol):
  rowNum, colNum = board.shape

  symbolNum = 1
  me_state = []
  op_state = []
  if symbol is 'X':  symbolNum = -1
  
  for i in range(rowNum):
    for j in range(colNum):
      if board[i][j] != 0: 
        if board[i][j] == symbolNum: me_state.append((i,j))
        else: op_state.append((i,j))
  
  return frozenset(me_state), frozenset(op_state)    

def convertIndexToPosTuple(actionIndx:int, row:int, col:int) -> Tuple[int,int]:
  row_num = int(actionIndx / row)
  col_num = actionIndx % col
  return (row_num, col_num)

def draw_learn_curve(epoch, sim_num, data_series, agent_type, agent_num, title_name=""):
  xAxis = [ (num+1) * epoch for num in range(sim_num)]
  _, ag = plt.subplots()
  ag.plot(xAxis, data_series, color = 'blue', label=agent_type)
  plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  ag.set(xlabel='episodes', ylabel='average return('+str(agent_num)+' agents)',
        title=title_name+'Learning curve')
  ag.grid()
  plt.show()
  return





