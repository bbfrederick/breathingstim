#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from psychopy import visual, event, core

draw_expanding_indicator(value, diameter=1.0, lobes=6, minval=0.5, maxval=1.0):
    currentsize = (value * (maxval - minval) + minval) * diameter
    for i in range(lobes):
        angle=i * 360.0/lobes
        xloc = 0.5 * diameter * np.sin(angle)
        yloc = 0.5 * diameter * np.cos(angle)