#################################################################
# Copyright (C)                                                 #
# 2018 - 2019 Li-Han, Chen(eden.chen@bath.edu)                  #
# Permission given to modify the code as long as you keep this  #
# declaration at the top                                        #
#################################################################

import numpy as np
from src.Player import Player
from src.env.Gomoku import Gomoku
from lib.Config import *

from typing import Dict, Tuple, Sequence

class Rand_play(Player):
  def __init__(self, symbol):
    super().__init__(symbol)

  @staticmethod 
  def get_action(env: Gomoku) -> None:
    rand_row = np.random.randint(env.row)
    rand_col = np.random.randint(env.col)
    while env.cell_is_empty((rand_row, rand_col)) is FAIL:
      rand_row = np.random.randint(env.row)
      rand_col = np.random.randint(env.col)
    
    return (rand_row, rand_col)
  
  def learn_from_transition(self, state:Sequence[Sequence[int]], reward:int, terminate:bool) -> None:
    pass



  