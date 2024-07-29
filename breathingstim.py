#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo for psychopy.visual.ShapeStim.contains() and .overlaps()

Also inherited by various other stimulus types.
"""

from psychopy import visual, event, core
import numpy as np


def draw_expanding_indicator(win, value, diameter=1.0, lobes=6, minval=0.5, maxval=1.0):
    currentsize = (value * (maxval - minval) + minval) * diameter
    acircle = visual.Circle(win, radius=currentsize, edges=2, units='pix')
    for i in range(lobes):
        angle=i * 360.0/lobes
        xloc = 0.5 * currentsize * np.sin(angle)
        yloc = 0.5 * currentsize * np.cos(angle)
        print(xloc, yloc, currentsize)
        acircle.pos = (xloc, yloc)
        acircle.size = currentsize
        acircle.opacity = 0.5
        acircle.draw()
        
        
win = visual.Window(size=(500, 500), monitor='testMonitor', units='norm')
mouse = event.Mouse()
txt = 'click the shape to quit\nscroll to adjust circle'
instr = visual.TextStim(win, text=txt, pos=(0, -.7), opacity=0.5)
msg = visual.TextStim(win, text=' ', pos=(0, -.4))


# define a buffer zone around the mouse for proximity detection:
# use pix units just to show that it works to mix (shape and mouse use norm units)
bufzone = visual.Circle(win, radius=30, edges=13, units='pix')

# loop until detect a click inside the shape:
while not False:
    instr.draw()
    # dynamic buffer zone around mouse pointer:
    bufzone.pos = mouse.getPos() * win.size / 2  # follow the mouse
    bufzone.size += mouse.getWheelRel()[1] / 20.0  # vert scroll adjusts radius

    draw_expanding_indicator(win, 1.0, diameter=30, lobes=6, minval=0.5, maxval=1.0)
    msg.draw()
    win.flip()

win.close()
core.quit()

# The contents of this file are in the public domain.
