import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from typing import Callable
from dataclasses import dataclass
import math

# random mapping so grok doesn't learn based off token patterns
mod_p = 97
np.random.seed(42)
mapping = np.arange(mod_p)
np.random.shuffle(mapping)
# np.array as is didnt work with np.vectorize(randomize_vals)?
mapping = {i:v for i,v in enumerate(mapping)}
mapping_inv = {v:i for i,v in mapping.items()}

# operations to generate data
@dataclass
class Op:
    name: str
    fn: Callable[[int,int], int]
    def gen_data(self):
        ret = np.zeros((mod_p,mod_p, 3))
        for x in range(mod_p):
            for y in range(mod_p):
                ret[x,y] = x,y, self.fn(x,y)
        return ret.reshape(-1, 3)
ops = [
    Op(
        "+",
        lambda x,y: (x+y)%mod_p
    ),
    Op(
        "-",
        lambda x,y: (x-y)%mod_p
    ),
    Op(
        "x^2+y^2",
        lambda x,y: (x**2+y**2)%mod_p
    ),
    Op(
        "x^2+xy+y^2",
        lambda x,y: (x**2+x*y+y**2)%mod_p
    ),
    Op(
        "x^2+xy+y^2+x",
        lambda x,y: (x**2+x*y+y**2+x)%mod_p
    ),
    Op(
        "x^3+xy",
        lambda x,y: (x**3+x*y)%mod_p
    ),
    Op(
        "x^3+xy^2+y",
        lambda x,y: (x**3+x*y**2+y)%mod_p
    ),
]

def randomize_vals(v):
    return mapping[v]
randomize_vals = np.vectorize(randomize_vals)
def unrandomize_vals(v):
    return mapping_inv[v]
unrandomize_vals = np.vectorize(unrandomize_vals)



def generate_all_data(random=True):
    data = {}
    for op in ops:
        data[op.name] = {}
        for p in np.arange(0.2, 0.85, 0.05):
            vals = op.gen_data()
            if random:
                vals = randomize_vals(vals)
            np.random.seed(42)
            np.random.shuffle(vals)
            split_i = int(len(vals)*p)
            train = torch.tensor(vals[:split_i], dtype=torch.long)
            val = torch.tensor(vals[split_i:split_i+512], dtype=torch.long)
            # TODO: kinda of jank, we're technically also predicting y. maybe bad?
            data[op.name][f"{p:.2f}"] = {
                "train": (train[:,:2].contiguous(), train[:,1:].contiguous()),
                "val": (val[:,:2].contiguous(), val[:,1:].contiguous()),
            }
    return data

#import code; code.interact(local=locals())