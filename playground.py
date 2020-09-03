#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 21:37:17 2020

@author: daniyalusmani1
"""


import numpy as np
import pandas as pd

x = np.array([
    [np.nan,2,3,4,7,6,5,5,4,4,np.nan,np.nan,np.nan], [1,2,np.nan,4,7,6,5,5,np.nan,4,3,2,1], [1.5,2.123,3.432,4.432,7.123,6.234,np.nan,5.234,np.nan,4,3,2,1], [1,2,3,4,7,6,5,5,4,4,3,2,1], [1,2,3,4,np.nan,np.nan,np.nan,5,4,4,3,2,1]])

y = np.array([1,2,3,4,5])
print(x.shape)
print(y.shape)

y = np.expand_dims(y, axis=1)
print(y.shape)

print(np.concatenate([x,y], axis= 1))


x_df = pd.DataFrame(x)
print(x_df)

x_df_ff = x_df.ffill()
print(x_df_ff)

x_df_bf = x_df.T.bfill().T
print(x_df_bf)

x_df_interpolate = x_df.T.interpolate(method='slinear').interpolate().bfill().T
print(x_df_interpolate)


