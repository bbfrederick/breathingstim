#!/usr/bin/env python
#
#       $Author: frederic $
#       $Date: 2013/01/10 22:16:36 $
#       $Id: fMRI_checkerflash.py,v 1.3 2013/01/10 22:16:36 frederic Exp $
#

"""This demo illustrates using hardware.emulator.launchScan() to either start a real scan, 
or emulate sync pulses and user responses. Emulation is to allow debugging script timing
offline, without requiring either a scanner or a hardware sync pulse emulator.
"""

__author__ = "Blaise Frederick"

import numpy as np
from psychopy import core, event, gui, plugins, visual
import json
import sys

plugins.activatePlugins()

import matplotlib.pyplot as plt
from psychopy.hardware.emulator import SyncGenerator, launchScan

################################################
MAXLINES = 10000

# How to define a breathing stimulus:
# First you decide what the respiratory type will be (boxbreathe, inhaleexhale, etc.)
#   This defines how you breathe across the respiration cycle (when you inhale, hold, and exhale)
# 0.0 <= respphase < 1.0 is the position within the respiration waveform for one full cycle
# 0.0 <= respval <= 1.0 is the depth of inspiration, from full exhale to full inhale
# respfunction is the mapping from respphase to respval

class Boxbreathe:

    def __init__(self, xoffset=0.0, xscale=0.25, yoffset=0.0, yscale=0.25):
        self.xoffset = xoffset
        self.xscale = xscale
        self.yoffset = yoffset
        self.yscale = yscale
        self.inhaleend = 0.25
        self.inhaleholdend = 0.5
        self.exhaleend = 0.75
        initstim()

    def initstim(self):
        # make the frame
        self.frame = visual.Rect(
            win,
            width=targetscale,
            height=targetscale,
            lineWidth=6.0,
            lineColor=(0, 0.5, 0),
            fillColor=None,
            units="height",
        )

        # make the target
        self.target = visual.Circle(win, radius=0.03, units="height")
        self.target.setFillColor((0, 0.5, 0))
        self.target.setLineColor((0, 0.5, 0))

    def phasetopos(self, thephase):
        if 0.0 <= thephase < self.inhaleend:
            winstart = 0.0
            winwidth = self.inhaleend - winstart
            xval = -0.5
            yval = -0.5 + (thephase - winstart) / winwidth
        elif self.inhaleend <= thephase < selfinhaleholdend:
            winstart = self.inhaleend
            winwidth = self.selfinhaleholdend - winstart
            xval = -0.5 + (thephase - winstart) / winwidth
            yval = 0.5
        elif self.inhaleholdend <= thephase < self.exhaleend:
            winstart = self.inhaleholdend
            winwidth = self.exhaleend - winstart
            xval = 0.5
            yval = 0.5 - (thephase - winstart) / winwidth
        else:
            winstart = self.exhaleend
            winwidth = 1.0 - winstart
            xval = 0.5 - (thephase - winstart) / winwidth
            yval = 0.5 
        xpos = xval * self.xscale + self.xoffset
        ypos = yval * self.yscale + self.yoffset
        return [xpos, ypos]

    def draw(self, thephase):
        self.target.setPos(self.phasetopos(thephase))
        self.frame.draw()
        self.target.draw()

    def getvalue(self):
        return self.phasetopos(thephase)[1]


class BreathingPattern:
    thetype = None
    thephase = None

    def __init__(self, thetype="boxbreathe", thephase=0.0):
        self.settype(thetype)
        self.setphase(thephase)

    def settype(self, thetype):
        self.thetype = thetype
        if self.thetype == "boxbreathe":
            self.stimulus = Boxbreathe()

    def gettype(self):
        return self.thetype

    def setphase(self, thephase):
        self.thephase = thephase

    def getphase(self):
        return self.thephase

    def draw(self):
        self.stimulus.draw(self.thephase)

    def getval(self):
        self.stimulus.getval(self.thephase)


def readexpfile(filename):
    with open(filename, "r") as json_data:
        d = json.load(json_data)
        for token in [
            "preambletime",
            "warntime",
            "BPM_start",
            "BPM_end",
            "fmritime",
            "TR"]:
            try:
                inval = float(d[token])
            except KeyError:
                print(f"{token} is not defined in input file - quitting")
                sys.exit()
        try:
            stimtype = d["stimtype"]
        except KeyError:
            print(f"{token} is not defined in input file - quitting")
            sys.exit()                
        return float(d["preambletime"]), float(d["warntime"]), float(d["BPM_start"]), float(d["BPM_end"]), float(d["fmritime"]), float(d["TR"]), d["stimtype"]
    print(f"could not open input file {filename}")
    sys.exit()


def set_expanding_indicator(
    respval, stim, diameter=1.0, lobes=6, minval=0.25, maxval=0.5
):
    currentsize = (respval * (maxval - minval) + minval) * diameter
    for i in range(lobes):
        angle = i * 360.0 / lobes
        xloc = 0.5 * diameter * np.sin(angle)
        yloc = 0.5 * diameter * np.cos(angle)


def valtopos(xval, yval, xoffset=0.0, xscale=0.25, yoffset=0.0, yscale=0.25):
    xpos = xval * xscale + xoffset
    ypos = yval * yscale + yoffset
    return [xpos, ypos]


def readvecs(inputfilename):
    file = open(inputfilename)
    lines = file.readlines(MAXLINES)
    numvecs = len(lines[0].split())
    inputvec = np.zeros((numvecs, MAXLINES), dtype="float")
    numvals = 0
    for line in lines:
        numvals = numvals + 1
        thetokens = line.split()
        for vecnum in range(0, numvecs):
            inputvec[vecnum, numvals - 1] = float(thetokens[vecnum])
    return np.transpose(1.0 * inputvec[:, 0:numvals])


def readandprocessstims(
    thefilename,
    numtrs,
    tr,
    timestep,
    debug=False,
):
    valarray = readvecs(thefilename)
    if debug:
        print(valarray)
    starttime, startx, starty = valarray[0, :]
    finishtime = valarray[-1, 0]
    if debug:
        print(f"{starttime=}, {finishtime=}, {timestep=}")
    numsteps = int(finishtime / timestep) + 1
    specifiedvals = np.zeros((numsteps, 3), dtype=float)
    if debug:
        print(f"{numsteps=}, {specifiedvals.shape=}")
    whichstep = 0
    for i in range(1, valarray.shape[0]):
        endtime, endx, endy = valarray[i]
        if debug:
            print(
                f"{i}: {starttime=}, {endtime=}, {startx=}, {endx=}, {starty=}, {endy=}"
            )
        startindex = int(starttime / timestep)
        endindex = int(endtime / timestep)
        segsize = endindex - startindex
        segtime = (segsize + 1) * timestep
        if debug:
            print(f"step {i}: {startindex=}, {endindex=}, {segsize=}")
        if startx == endx:
            # y is changing
            specifiedvals[startindex:endindex, 0] = startx
            specifiedvals[startindex:endindex, 1] = np.linspace(
                starty, endy, num=segsize, endpoint=False, dtype=float
            )
        else:
            # x is changing
            specifiedvals[startindex:endindex, 0] = np.linspace(
                startx, endx, num=segsize, endpoint=False, dtype=float
            )
            specifiedvals[startindex:endindex, 1] = starty
        specifiedvals[startindex:endindex, 2] = segtime
        starttime = endtime
        startx = endx
        starty = endy
    numoutputvals = int(numtrs * (tr / timestep))
    outputvals = np.zeros((numoutputvals, 3), dtype=float)
    if numoutputvals > numsteps:
        outputvals[:numsteps, :] = specifiedvals
        outputvals[numsteps:, :] = specifiedvals[-1]
    elif numoutputvals == numsteps:
        outputvals = specifiedvals
    else:
        outputvals = specifiedvals[:numoutputvals, :]
    return outputvals


# execution starts here
# Configurable parameters
initpath = "/Users/frederic/code/breathingstim"
initfile = "risingtime.json"
debug = False  # turn on counter in upper righthand corner
targetscale = 0.25
fullscreen = True
preamblelength = 10.0
warninglength = 3.0

theexpfile = gui.fileOpenDlg(
    tryFilePath=initpath, tryFileName=initfile, allowed="*.json"
)
if theexpfile:
    preambletime, warntime, BPM_start, BPM_end, fmritime, TR, stimtype = readexpfile(theexpfile[0])
    print(preambletime, warntime, BPM_start, BPM_end, fmritime, TR, stimtype)
    sys.exit()

# settings for launchScan:
MR_settings = {
    "TR": 1.33,  # duration (sec) per volume
    "volumes": 286,  # number of whole-brain 3D volumes / frames
    "sync": "t",  # character to use as the sync timing event; assumed to come at start of a volume
    "skip": 0,  # number of volumes lacking a sync pulse at start of scan (for T1 stabilization)
    "sound": False,  # in test mode only, play a tone as a reminder of scanner noise
}

# set a timestep that is ~0.1 seconds but is an integral divisor of TR
stepspertr = int(np.round(MR_settings["TR"] / 0.01, 0))
timestep = MR_settings["TR"] / stepspertr
print("MR_settings initialization done")

print("Stimulus initialization started")
infoDlg = gui.DlgFromDict(MR_settings, title="fMRI parameters", order=["TR", "volumes"])
if not infoDlg.OK:
    core.quit()

filename = gui.fileOpenDlg(
    tryFilePath=initpath, tryFileName=initfile, allowed="*.bstim"
)
if filename:
    # generate a list of values for each TR
    outputvals = readandprocessstims(
        filename[0],
        MR_settings["TR"],
        MR_settings["volumes"],
        timestep,
        debug=False,
    )
else:
    core.quit()
print("Stimulus initialization done")

win = visual.Window(
    [800, 600], fullscr=fullscreen, checkTiming=False
)  # this has been moved up to the stimulus definition
globalClock = core.Clock()

# summary of run timing, for each key press:
output = "vol    onset key\n"
for i in range(-1 * MR_settings["skip"], 0):
    output += "%d prescan skip (no sync)\n" % i

key_code = MR_settings["sync"]
pause_during_delay = MR_settings["TR"] > 0.3
sync_now = False

# can simulate user responses, here 3 key presses in order 'a', 'b', 'c' (they get sorted by time):
simResponses = []

infer_missed_sync = False  # best if your script timing works without this, but this might be useful sometimes
max_slippage = 0.02  # how long to allow before treating a "slow" sync as missed
# any slippage is almost certainly due to timing issues with your script or PC, and not MR scanner

# make the frame
frame = visual.Rect(
    win,
    width=targetscale,
    height=targetscale,
    lineWidth=6.0,
    lineColor=(0, 0.5, 0),
    fillColor=None,
    units="height",
)

# make the target
target = visual.Circle(win, radius=0.03, units="height")
target.setFillColor((0, 0.5, 0))
target.setLineColor((0, 0.5, 0))

# make the preamble
preamble = visual.TextStim(
    win, height=0.05, pos=(0.0, 0.0), color=win.rgb + 0.5, units="height"
)
preamble.setText("Breathe normally")

# make the warning
warning = visual.TextStim(
    win, height=0.05, pos=(0.0, 0.0), color=win.rgb + 0.5, units="height"
)
warning.setText("Exhale")

# make the counter
counter = visual.TextStim(
    win, height=0.05, pos=(0.0, -0.4), color=win.rgb + 0.5, units="height"
)
counter.setText("")

# plot the waveforms
if not fullscreen:
    plt.plot(outputvals[:, 0])
    plt.plot(outputvals[:, 1])
    plt.show()

duration = MR_settings["volumes"] * MR_settings["TR"]
# note: globalClock has been reset to 0.0 by launchScan()
vol = 0
onset = 0.0

# launch: operator selects Scan or Test (emulate); see API documentation
vol = launchScan(win, MR_settings, globalClock=globalClock, wait_msg="Breathe normally")
sync_now = "Experiment start"

# draw the preamble text so if will be on the screen while waiting
preamble.draw()
win.flip()

currentindex = -1
preambleendindex = int((preamblelength - warninglength) / timestep)
warningendindex = int(preamblelength / timestep)
print("here we go!")
while globalClock.getTime() < duration:
    allKeys = event.getKeys()
    if "escape" in allKeys:
        output += "user cancel, "
        break
    # detect sync or infer it should have happened:
    if MR_settings["sync"] in allKeys:
        sync_now = key_code  # flag
        onset = globalClock.getTime()
    if infer_missed_sync:
        expected_onset = vol * MR_settings["TR"]
        now = globalClock.getTime()
        if now > expected_onset + max_slippage:
            sync_now = "(inferred onset)"  # flag
            onset = expected_onset
    if sync_now:
        # do your experiment code at this point; for demo, just shows a counter & time
        output += "%3d  %7.3f %s\n" % (vol, onset, sync_now)
        vol += 1
        sync_now = False
    # now draw whatever we are drawing
    now = globalClock.getTime()
    thisindex = int(now / timestep)
    if thisindex > currentindex and thisindex < (warningendindex + outputvals.shape[0]):
        # time to update the display
        if thisindex >= warningendindex:
            outputvalue = outputvals[thisindex - warningendindex, :]
            newpos = valtopos(
                outputvalue[0], outputvalue[1], xscale=targetscale, yscale=targetscale
            )
            target.setPos(newpos)
            frame.draw()
            target.draw()
            counter.setText(f"Index: {thisindex:5d}, segtime: {outputvalue[2]:.3f}")
        elif thisindex >= preambleendindex:
            warning.draw()
            counter.setText(f"Index: {thisindex:5d}")
        else:
            preamble.draw()
            counter.setText(f"Index: {thisindex:5d}")
        if debug:
            counter.draw()
        win.flip()
        currentindex = thisindex
print("now we're done!")
output += "end of scan (vol 0..%d = %d of %s). duration = %7.3f" % (
    vol,
    MR_settings["volumes"],
    MR_settings["sync"],
    globalClock.getTime(),
)
print(output)
core.quit()
