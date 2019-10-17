# Reinforcement Learning in a Board Game
This project offers tabular methods and deep learning methods to solve Connect Games such as Noughts-and-Crosses or Gomoku. However, a bigger board will lead to a lower performance. 

## Tutorial
```
How to use this?
You could verify the methods that I posted in `tests` folder.
I put them in `tests` because people could examine algorithms through either my code or my testing logic.
```

## Folder Structure
    .
    ├── ..
    ├── lib                        [GENERAL METHODS]
    │   ├── config.py              # Debugging file containing assertion code. (In the beginning, many codes are asserted, 
    │   │                            but when the agents fail too many times, I only focus on agents code and networks architecture.) 
    │   └── General.py             # This file contains general methods 
    │ 
    ├── src                        [SOURCE FILES]
    │   ├── env                    # General library for agents 
    │   │   ├── Board.py           # Class for a board game
    │   │   ├── BoardSetting.py    # Class for setting a specific game start 
    │   │   ├── Game.py            # Class to play connect N (Noughts-and-Crosses)
    │   │   └── Gomoku.py          # Class of Connect including Noughts-and-Crosses/ Gomoku/ Connect N 
    │   │
    │   ├── Human.py               # Class for Human player to play with any agents 
    │   ├── Player.py              # Class for player in a board game 
    │   ├── q_learning.py          # Temporal-Difference Learning proposed methods by Watkins(1992). There are two major features in this algorithm design.
    │   │                            1. The play order is removed in this algorithm. By doing so, the complexity of computing will be minimise to merely around 10k.
    │   │                            2. A symmetry is eliminated as well. Hence, the complexity could reach O(765).
    │   └── Rand_play.py           # Class for a random player which pick a move based on the size of a board
    │
    ├── tests                      [UNIT TESTS]
    │   ├── env                    # This folder tests environments
    │   │   ├── board_test.py      # Unit-test of Board Class including verifying winning condition 
    │   │   └── gomoku_test.py     # Unit-test of Connect Game. I leave the name Gomoku but it could be Noughts-and-Crosses/ connect Four/ Gomoku etc, depending on the need.
    │   └── q_test.py              # This is a famous Temporal-Difference Learning proposed by Watkins(1992)
    │
    └── README.md                  # MArkdown file to explain this project


# References
[1] Watkins, C.J. and Dayan, P., 1992. Q-learning. Machine learning, 8(3-4), pp.279–292.
