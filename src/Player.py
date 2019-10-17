#################################################################
# Copyright (C)                                                 #
# 2018 - 2019 Li-Han, Chen(eden.chen@bath.edu)                  #
# Permission given to modify the code as long as you keep this  #
# declaration at the top                                        #
#################################################################

from typing import Set, Dict, Tuple, Sequence

class Player:
  def __init__(self, symbol:str) -> None:
    self.__symbol = symbol
    self.name = None
    self.code = 1
    if self.__symbol == 'X':
      self.code = -1

  @property
  def symbol(self):
    return self.__symbol
  @symbol.setter
  def symbol(self, symbol:str) -> None:
    self.__symbol = symbol
  
  def get_action(self):
    pass
  
  def learn_from_transition(self, state:Sequence[Sequence[int]], reward:int, Terminate:bool) -> None:
    pass