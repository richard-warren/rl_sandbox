import numpy as np

mazes = [
    {'walls': np.array([[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype='bool'),
     'terminal_states': [[4,14,1], [4,19,99]],
     'start_coords': [0,0]},

    {'walls': np.zeros((10,10), dtype='bool'),
     'terminal_states': [[9,9,99], [0,9,5]],
     'start_coords': [0,0]},

    {'walls': np.zeros((5,5), dtype='bool'),
     'terminal_states': [[4,4,1]],
     'start_coords': [0,0]}
]
