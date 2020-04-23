import os
import sys
import copy
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from sklearn.model_selection import train_test_split

from debug import ipsh


class SCM(object):

  def __init__(self, exogenous_nodes, exogenous_nodes, exogenous_probs, structural_equations):

