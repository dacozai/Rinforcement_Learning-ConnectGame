#################################################################
# Copyright (C)                                                 #
# 2018 - 2019 Li-Han, Chen(eden.chen@bath.edu)                  #
# Permission given to modify the code as long as you keep this  #
# declaration at the top                                        #
#################################################################

import numpy as np
from typing import Dict, Tuple, Sequence

from src.Player import Player
from src.env.Gomoku import Gomoku

class Human(Player):
  def __init__(self, symbol):
    super().__init__(symbol)

  # Format row col
  def get_action(self, env:Gomoku) -> Tuple[int, int]:
    env.print_board()
    flag = True
    while flag:
      pos = input("input the position: ")
      pos = list(pos)
      
      try:
        flag = not env.cell_is_empty((int(pos[0]), int(pos[1])))
      except:
        print("Canot place it!")

    return (int(pos[0]), int(pos[1]))

  def learn_from_transition(self, state:Sequence[Sequence[int]], reward:int, Terminate:bool) -> None:
    pass