import numpy as np
import pandas as pd
import readline
import rpy2.robjects
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.packages import importr
pandas2ri.activate()

def glmer(data, formula):
    base = importr('base')
    lme4 = importr('lme4')
    rs = lme4.glmer(Formula(formula), data, family = 'binomial')
    
    fixed_effects = lme4.fixed_effects(rs)
    fixed_effects = {k:v for k,v in zip(fixed_effects.names, list(fixed_effects))}
                                  
    random_effects = lme4.random_effects(rs)[0]
    random_effects = pd.DataFrame([list(lst) for lst in random_effects], index = list(random_effects.colnames)).T
    print(base.summary(rs))
    return fixed_effects, random_effects

