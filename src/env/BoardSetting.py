#################################################################
# Copyright (C)                                                 #
# 2018 - 2019 Li-Han, Chen(eden.chen@bath.edu)                  #
# Permission given to modify the code as long as you keep this  #
# declaration at the top                                        #
#################################################################

class BoardSet():

  def reset_choice(self, choice, board):
    if choice == 'RESET': return board
    elif choice == '3t1': return self.bd3_tactic_1(board)
    elif choice == '3t2': return self.bd3_tactic_2(board)
    elif choice == '3t3': return self.bd3_tactic_3(board)
    elif choice == '3t4': return self.bd3_tactic_4(board)
    elif choice == '3t5': return self.bd3_tactic_5(board)
    elif choice == '3n1': return self.bd3_normal_1(board)
    elif choice == '3n2': return self.bd3_normal_2(board)
    elif choice == '3n3': return self.bd3_normal_3(board)
    elif choice == '3n4': return self.bd3_normal_4(board)
    elif choice == '3n5': return self.bd3_normal_5(board)

  """
    | X |   
    | X | 
  O | O |  
  """
  def bd3_normal_1(self, board):
    board[0][1] = -1
    board[1][1] = -1
    board[2][0] =  1
    board[2][1] =  1
    return board

  """
  X |   |   
  O | O |  
  X |   |  
  """
  def bd3_normal_2(self, board):
    board[1][0] =  1
    board[1][1] =  1
    board[0][0] = -1
    board[2][0] = -1
    return board
  
  """
  X | X | 
  O | O | 
    |   |  
  """
  def bd3_normal_3(self, board):
    board[1][0] =  1
    board[1][1] =  1
    board[0][0] = -1
    board[0][1] = -1
    return board
  
  """
  O | X | 
    | X | 
    |   | O 
  """
  def bd3_normal_4(self, board):
    board[0][0] =  1
    board[2][2] =  1
    board[0][1] = -1
    board[1][1] = -1
    return board
  
  """
    |   | O
  X | X | O
  X | O |  
  """
  def bd3_normal_5(self, board):
    board[0][2] =  1
    board[1][2] =  1
    board[2][1] =  1
    board[1][0] = -1
    board[1][1] = -1
    board[2][0] = -1
    return board

  """
  O |   | X
    |   | O
    |   | X 
  """
  def bd3_tactic_1(self, board):
    board[0][2] = -1
    board[2][2] = -1
    board[0][0] =  1
    board[1][2] =  1
    return board 
  
  """
  X |   | O
    |   | X
    |   | O
  """
  def bd3_tactic_2(self, board):
    board[0][2] =  1
    board[2][2] =  1
    board[0][0] = -1
    board[1][2] = -1
    return board 

  """
  X | O | 
    | O | 
    | X |  
  """
  def bd3_tactic_3(self, board):
    board[0][1] =  1
    board[1][1] =  1
    board[0][0] = -1
    board[2][1] = -1
    return board 
  
  """
  O | X | 
    | X | 
    | O |  
  """
  def bd3_tactic_4(self, board):
    board[0][1] = -1
    board[1][1] = -1
    board[0][0] =  1
    board[2][1] =  1
    return board

  """
  X |   | 
  O |   | 
    |   |  
  """
  def bd3_tactic_5(self, board):
    board[0][0] = -1
    board[1][0] =  1
    return board 


