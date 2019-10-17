#################################################################
# Copyright (C)                                                 #
# 2018 - 2019 Li-Han, Chen(eden.chen@bath.edu)                    #
# Permission given to modify the code as long as you keep this  #
# declaration at the top                                        #
#################################################################

import numpy as np
import sys
sys.path.append('..')
from typing import Set, Dict, Tuple, Sequence, List, Any
import copy

from src.env.Gomoku import Gomoku
from src.env.Game import Game 
from src.q_learning import Q_learn 
from src.Rand_play import Rand_play 
from lib.General import *

class Q_test(Q_learn):
  def __init__(self, bd_sz, args):
    super().__init__('X', bd_sz, args )
  
  def pick_action(self, env:Gomoku, intended_move:Tuple[int, int]) -> None:
    me_ky, op_ky = convertBoardToStateKey(env.board, self.symbol)
    me_ky, op_ky, turn_flag = self.check_symmetry(me_ky, op_ky) 
    self._prev_state = [me_ky, op_ky]
    if me_ky not in self._q_table:
      self._q_table[me_ky] = {op_ky:self.create_act_row(env.board)}
    else:
      if op_ky not in self._q_table[me_ky]:
        self._q_table[me_ky][op_ky] = self.create_act_row(env.board)  
    
    self._prev_action = convertIndexToPosTuple(intended_move, self._row, self._col)
    return (abs(self._prev_action[0]-turn_flag[0]), abs(self._prev_action[1]-turn_flag[1]))
  
  def epsilon_greedy(self, keys):
    out_ky, in_ky = keys
    actions = [ ky for ky in self._q_table[out_ky][in_ky] ]
    flag = False
    if np.random.uniform(0, 1) > self.epsilon:
      flag = True
      actions = self.search_move_w_max_value(self._q_table[out_ky][in_ky])

    return np.random.choice(actions), flag

  def get_action(self, env):
    me_ky, op_ky = convertBoardToStateKey(env.board, self.symbol)
    me_ky, op_ky, turn_flag = self.check_symmetry(me_ky, op_ky) 
    self._prev_state = [me_ky, op_ky]

    if me_ky not in self._q_table:
      self._q_table[me_ky] = {op_ky:self.create_act_row(env.board)}
    else:
      if op_ky not in self._q_table[me_ky]:
        self._q_table[me_ky][op_ky] = self.create_act_row(env.board)

    intended_move, f_bool = self.epsilon_greedy([me_ky, op_ky])
    self._prev_action = convertIndexToPosTuple(intended_move, self._row, self._col)
    # return self._prev_action
    return (abs(self._prev_action[0]-turn_flag[0]), abs(self._prev_action[1]-turn_flag[1])), f_bool

def run_a_trail(env, q_agent, me_indx, op_indx, pos_indx, op_pos):
  act_pos = q_agent.pick_action(env, pos_indx)
  # op_pos = (0,0)
  state, reward, game_over, _ = env.gomoku_respond(act_pos, 'X')
  state, reward, game_over, _ = env.gomoku_respond(op_pos,'O')
  me_indx.append(act_pos)
  op_indx.append(op_pos)
  return env, q_agent, state, reward, me_indx, op_indx, game_over

sz            = 3
win_condition = 3
bd_sz = [sz, sz]
argsSet = {'epsilon': 0.05, 'gamma': 1, 'alpha': 0.9}
rand_agent = Rand_play('O')
q_agent = Q_test(bd_sz, argsSet)
env = Gomoku(bd_sz,win_condition)

# 1 ........................................................................
env.board = 'RESET'
state = copy.deepcopy(env.board)
reward = 0

"""
TEST FOR THE FIRST ROUND
1. initial state : learn nothing [0,0,0,0,0,0,0,0,0]
"""
q_agent.learn_from_transition(state, reward, False)
me_indx = []
op_indx = []
test_case = {frozenset(me_indx):{frozenset(op_indx): {indx:0 for indx in range(np.prod(bd_sz))}}}
assert(q_agent.q_table == test_case)
env, q_agent, state, reward, me_indx, op_indx, game_over = \
        run_a_trail(env, q_agent, me_indx, op_indx, 4, (0,0))

"""
2. move to the next state [1,0,0,0,-1,0,0,0,0]
"""
q_agent.learn_from_transition(state, reward, False)
assert(frozenset(me_indx) in q_agent.q_table)
assert(frozenset(op_indx) in q_agent.q_table[frozenset(me_indx)])
test_case = {indx:0 for indx in range(np.prod(bd_sz))}
del test_case[4]
del test_case[0]
assert(q_agent.q_table[frozenset(me_indx)][frozenset(op_indx)] == test_case)
del test_case[1]
env, q_agent, state, reward, me_indx, op_indx, game_over = \
        run_a_trail(env, q_agent, me_indx, op_indx, 1, (2,1))

"""
3. next state is [1,-1,0,0,-1,0,0,1,0]
"""
q_agent.learn_from_transition(state, reward, False)
assert(frozenset(me_indx) in q_agent.q_table)
assert(frozenset(op_indx) in q_agent.q_table[frozenset(me_indx)])
del test_case[7]
assert(q_agent.q_table[frozenset(me_indx)][frozenset(op_indx)] == test_case)
env, q_agent, state, reward, me_indx, op_indx, game_over = \
        run_a_trail(env, q_agent, me_indx, op_indx, 2, (2,0))
del test_case[2]
del test_case[6]
assert(test_case == {3:0, 5:0, 8:0})

"""
4. next state is [1,-1,-1,0,-1,0,1,1,0]
"""
q_agent.learn_from_transition(state, reward, False)
assert(frozenset(me_indx) in q_agent.q_table)
assert(frozenset(op_indx) in q_agent.q_table[frozenset(me_indx)])
assert(q_agent.q_table[frozenset(me_indx)][frozenset(op_indx)] == test_case)
env, q_agent, state, reward, me_indx, op_indx, game_over = \
        run_a_trail(env, q_agent, me_indx, op_indx, 5, (1,0))
del test_case[5]
del test_case[3]

"""
5. Game_over dealing [1,-1,-1,1,-1,-1,1,1,0] 
"""
assert(game_over)
q_agent.learn_from_transition(env.board, -1, game_over)
assert(frozenset(me_indx) not in q_agent.q_table)
del me_indx[-1]
del op_indx[-1]
assert(q_agent.q_table[frozenset(me_indx)][frozenset(op_indx)][5] == -0.9)
test_q_table = copy.deepcopy(q_agent.q_table)


# 2 ........................................................................


"""
1. Reset the game, now, it is not possible to learn from the last move
"""
env.board = "RESET"
state = copy.deepcopy(env.board)
reward = 0
me_indx = []
op_indx = []

q_agent.learn_from_transition(state, reward, False)
assert(q_agent.q_table == test_q_table)
env, q_agent, state, reward, me_indx, op_indx, game_over = \
        run_a_trail(env, q_agent, me_indx, op_indx, 4, (0,0))

"""
2. Keep placing stones (should be the same as well)
"""
q_agent.learn_from_transition(state, reward, False)
assert(q_agent.q_table == test_q_table)
env, q_agent, state, reward, me_indx, op_indx, game_over = \
        run_a_trail(env, q_agent, me_indx, op_indx, 1, (2,1))

q_agent.learn_from_transition(state, reward, False)
assert(q_agent.q_table == test_q_table)
env, q_agent, state, reward, me_indx, op_indx, game_over = \
        run_a_trail(env, q_agent, me_indx, op_indx, 2, (2,0))

q_agent.learn_from_transition(state, reward, False)
assert(q_agent.q_table == test_q_table)

# verify it 
for indx in range(100):
  pos, f_bool = q_agent.get_action(env)
  pos_indx = pos[0] * bd_sz[0] + pos[1]
  if f_bool:  assert( f_bool and pos_indx != 5)
  else: assert(pos_indx in [3,5,8])

env, q_agent, state, reward, me_indx, op_indx, game_over = \
        run_a_trail(env, q_agent, me_indx, op_indx, 8, (1,0))
del me_indx[-1]
del op_indx[-1]
assert(game_over)
q_agent.learn_from_transition(env.board, -1, game_over)
assert(q_agent.q_table[frozenset(me_indx)][frozenset(op_indx)][5] == -0.9)
assert(q_agent.q_table[frozenset(me_indx)][frozenset(op_indx)][8] == -0.9)
test_q_table = copy.deepcopy(q_agent.q_table)


# 3 ........................................................................


"""
1. Reset the game, now, it is not possible to learn from the last move
"""
env.board = "RESET"
state = copy.deepcopy(env.board)
reward = 0
me_indx = []
op_indx = []

q_agent.learn_from_transition(state, reward, False)
assert(q_agent.q_table == test_q_table)
env, q_agent, state, reward, me_indx, op_indx, game_over = \
        run_a_trail(env, q_agent, me_indx, op_indx, 4, (0,0))

"""
2. Keep placing stones (should be the same as well)
"""
q_agent.learn_from_transition(state, reward, False)
assert(q_agent.q_table == test_q_table)
env, q_agent, state, reward, me_indx, op_indx, game_over = \
        run_a_trail(env, q_agent, me_indx, op_indx, 1, (2,1))

q_agent.learn_from_transition(state, reward, False)
assert(q_agent.q_table == test_q_table)
env, q_agent, state, reward, me_indx, op_indx, game_over = \
        run_a_trail(env, q_agent, me_indx, op_indx, 2, (2,0))

q_agent.learn_from_transition(state, reward, False)
assert(q_agent.q_table == test_q_table)

# verify it 
for indx in range(100):
  pos,f_bool = q_agent.get_action(env)
  pos_indx = pos[0] * bd_sz[0] + pos[1]
  if f_bool:  assert( f_bool and pos_indx == 3)
  else: assert(pos_indx in [3,5,8])

env, q_agent, state, reward, me_indx, op_indx, game_over = \
        run_a_trail(env, q_agent, me_indx, op_indx, 3, (2,2))
del me_indx[-1]
del op_indx[-1]
assert(game_over)
q_agent.learn_from_transition(env.board, -1, game_over)
assert(q_agent.q_table[frozenset(me_indx)][frozenset(op_indx)][3] == -0.9)
assert(q_agent.q_table[frozenset(me_indx)][frozenset(op_indx)][5] == -0.9)
assert(q_agent.q_table[frozenset(me_indx)][frozenset(op_indx)][8] == -0.9)
test_q_table = copy.deepcopy(q_agent.q_table)


# 4 ........................................................................


"""
1. Reset the game, now, it is not possible to learn from the last move
"""
env.board = "RESET"
state = copy.deepcopy(env.board)
reward = 0
me_indx = []
op_indx = []

q_agent.learn_from_transition(state, reward, False)
assert(q_agent.q_table == test_q_table)
env, q_agent, state, reward, me_indx, op_indx, game_over = \
        run_a_trail(env, q_agent, me_indx, op_indx, 4, (0,0))

"""
2. Keep placing stones (should be the same as well)
"""
q_agent.learn_from_transition(state, reward, False)
assert(q_agent.q_table == test_q_table)
env, q_agent, state, reward, me_indx, op_indx, game_over = \
        run_a_trail(env, q_agent, me_indx, op_indx, 1, (2,1))

q_agent.learn_from_transition(state, reward, False)
assert(q_agent.q_table == test_q_table)
env, q_agent, state, reward, me_indx, op_indx, game_over = \
        run_a_trail(env, q_agent, me_indx, op_indx, 2, (2,0))

################################################################
###  In here, it should learn action_index 2 is a bad idea.  ###
################################################################
q_agent.learn_from_transition(state, reward, False)
assert(q_agent.q_table != test_q_table)
assert(q_agent.q_table[frozenset(me_indx[:-1])][frozenset(op_indx[:-1])][2] == -0.81)
env, q_agent, state, reward, me_indx, op_indx, game_over = \
        run_a_trail(env, q_agent, me_indx, op_indx, 5, (2,2))
del me_indx[-1]
del op_indx[-1]
assert(game_over)
q_agent.learn_from_transition(env.board, -1, game_over)
assert(q_agent.q_table[frozenset(me_indx)][frozenset(op_indx)][5] == -0.99)
test_q_table = copy.deepcopy(q_agent.q_table)


# 5 ........................................................................


"""
1. Reset the game, now, it is not possible to learn from the last move
"""
env.board = "RESET"
state = copy.deepcopy(env.board)
reward = 0
me_indx = []
op_indx = []

q_agent.learn_from_transition(state, reward, False)
assert(q_agent.q_table == test_q_table)
env, q_agent, state, reward, me_indx, op_indx, game_over = \
        run_a_trail(env, q_agent, me_indx, op_indx, 4, (0,0))

"""
2. Keep placing stones (should be the same as well)
"""
q_agent.learn_from_transition(state, reward, False)
assert(q_agent.q_table == test_q_table)
env, q_agent, state, reward, me_indx, op_indx, game_over = \
        run_a_trail(env, q_agent, me_indx, op_indx, 1, (2,1))

#####################################
# Divergent point                   #
# never pick 2 unless it is epsilon #
#####################################

for i in range(100):
  pos,f_bool = q_agent.get_action(env)
  pos_indx = pos[0] * bd_sz[0] + pos[1]
  if f_bool:  assert(pos_indx != 2)
  else: assert(pos_indx in [2,3,5,6,8])

# TEST .....................................................................
env.board = 'RESET'
state = copy.deepcopy(env.board)
num           = 1
simNum        = 20
epiNum        = 500
sz            = 3
win_condition = 3
bd_sz = [sz, sz]

del env
env = Gomoku(bd_sz,win_condition)
gg = np.zeros(simNum)
gm = None
q_agent = Q_learn('X', bd_sz, argsSet)
for ag_times in range(num):
  print("This is "+str(ag_times))
  agents = [copy.deepcopy(rand_agent), copy.deepcopy(q_agent)]
  gm = None
  gm = Game(env=env, agents=agents, simNum=simNum, epiNum=epiNum)
  result = gm.learn(who_first=1)
  # print(result)
  gg = gg + result/num

draw_learn_curve(epiNum, simNum, gg, q_agent.name, num)
# gm.test(agents[1])
gm.play()













