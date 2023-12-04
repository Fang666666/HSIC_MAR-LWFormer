# -*- coding: utf-8 -*-
"""
@author: Yu Fang
"""
from Loop_SA_train import SA_loop_train_test
import warnings
import time

# remove abundant output
warnings.filterwarnings('ignore')

## global constant value
run_times = 2  # 20
class_num = 16
patch_size = 13


def Run_experiment():
    hp = {
        'run_times': run_times,
        'class_num': class_num,
        'patch_size': patch_size,
    }
    SA_loop_train_test(hp)


if __name__ == '__main__':
    Run_experiment()
    print(time.asctime(time.localtime()))
