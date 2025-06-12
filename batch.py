import numpy as np
from conbandit_combi1_0 import run

seed = np.random.randint(0, 10000)

for i in range(20):
    run(seed + i)