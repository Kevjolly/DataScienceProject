import numpy as np
from downsample import *
import sys
sys.path.append('../')    # import from folder one directory out
import util
import pandas as pd
import data_handler
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=12)


df = util.csv_to_df('../Datasets/star_light/train.csv')
reductions = np.linspace(1, 200, 100) 
shifting = [0, 20, 40, 80, 200]

test_classification(df, reductions, shifting)