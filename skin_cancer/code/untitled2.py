# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:18:26 2019

@author: lauri
"""

import matplotlib.pyplot as plt
from colour import Color
red = Color("red")
colors = list(red.range_to(Color("blue"),10))
colors = [x.get_rgb() for x in colors]