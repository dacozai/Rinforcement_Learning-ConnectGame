#################################################################
# Copyright (C)                                                 #
# 2018 - 2019 Li-Han, Chen(eden.chen@bath.edu)                  #
# Permission given to modify the code as long as you keep this  #
# declaration at the top                                        #
#################################################################

import numpy as np
from src.env.Gomoku import Gomoku
from src.Human import Human

from typing import Any, Sequence
from collections import deque

class Game():
  """
  Game class to run a board game for two agents including training and playing.

  ...

  Attributes
  ----------
  env
      The environment of a board game(Connect-N).
  agents
      The array of agents including reinforcement learning agent and o.w.
  _simNum
      The number of simulations.
  _epiNum
      The training number under a simulation
  _reward_set
      This is a reward set for the history of a training.

  Methods
  -------
  learn()
      The method to train the robot.
  play(agent_symbol=(0,0))
      The mehotd for human player to play with the agent.
  """

  def __init__(self, env, agents: Sequence[Any], simNum:int=1, epiNum:int=1000) -> None:
    self.env         = env
    self.agents      = agents 
    self._simNum     = simNum
    self._epiNum     = epiNum
    self._reward_set = []

  def learn(self, who_first=0,reset_tactic='RESET'):
    for sim_times in range(self._simNum):
      print(sim_times)
      total_reward = 0
      for _ in range(self._epiNum):
        # print(epoch_times)
        # print(str(sim_times) + '\t' + str(epoch_times))
        # print(str(sim_times) + '\t' + str(epoch_times), end='\t')
        game_over = False
        itr = who_first
        self.env.board = reset_tactic
        state = self.env.board
        reward = 0
        while not game_over:
          # self.env.print_board()
          # self.env.now_player = self.agents[itr%2].symbol
          self.agents[itr%2].learn_from_transition(state, reward, False)
          pos = self.agents[itr%2].get_action(self.env)
          # print(pos)
          # self.env.print_board()
          state, reward, game_over, winner = self.env.gomoku_respond(pos, self.agents[itr%2].symbol)
          itr += 1
        # self.env.print_board()

        # Set position 1 as studying target
        if winner ==  self.agents[1].symbol:
          reward = 1
        elif winner == 'nobody':  reward = 0
        else: reward = -1
        
        # print('Winner is '+winner)
        # print(win)
        self.agents[1].learn_from_transition(self.env.board, reward, game_over)
        # self.env.board = reset_tactic
        
        # if reward == -1:
        #  total_reward += reward
        total_reward += reward
        """ 
        self.env.place_stone((0,0),'X')
        self.env.place_stone((2,1),'X')
        self.env.place_stone((0,1),'O')
        self.env.place_stone((1,1),'O')
        """
        # print('\n\n\n\n')
      self._reward_set.append(total_reward/self._epiNum)

    # self.agents[1].print_qTable()
    # print(self._win_set)
    return np.array(self._reward_set)

  def play(self, agent_symbol:str= 'X', who_first=0, preset_board='RESET'):
    self.agents[1].symbol = agent_symbol
    if agent_symbol is 'X':
      human = Human('O')
    else:
      human = Human('X')

    agents = [self.agents[1], human]
    itr = who_first
    self.env.board = preset_board
    state = self.env.board
    reward = 0
    game_over = False
    while not game_over:
      self.env.now_player = agents[itr%2].symbol
      agents[itr%2].learn_from_transition(state, reward, False)
      pos = agents[itr%2].get_action(self.env)
      state, reward, game_over, winner = self.env.gomoku_respond(pos, agents[itr%2].symbol)
      itr += 1

    self.env.print_board()
    print('Winner is '+winner+'!')
