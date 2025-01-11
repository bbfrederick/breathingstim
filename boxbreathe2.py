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

import json
import sys

import numpy as np
import pandas as pd
from psychopy import core, event, gui, plugins, visual

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


class Mindful:
    def __init__(self, scale=0.125, lobes=9, minrad=0.25, maxrad=1.0, debug=False):
        self.scale = scale
        self.lobes = lobes
        self.minrad = minrad
        self.maxrad = maxrad
        self.inhaleend = 0.25
        self.inhaleholdend = 0.5
        self.exhaleend = 0.75
        self.thecircles = []
        self.debug = debug
        self.initstim()

    def initstim(self):
        for i in range(self.lobes):
            self.thecircles.append(visual.Circle(win, radius=1.0, units="height"))
            self.thecircles[-1].setFillColor(None)
            self.thecircles[-1].setLineColor((0, 0.5, 0))
            self.thecircles[-1].setLineWidth(6.0)
        if self.debug:
            print(f"Expanding indicator initialized with {len(self.thecircles)} lobes")

    def phasetorad(self, thephase):
        radslope = self.maxrad - self.minrad
        if 0.0 <= thephase < self.inhaleend:
            winstart = 0.0
            winwidth = self.inhaleend - winstart
            radval = self.minrad + radslope * (thephase - winstart) / winwidth
        elif self.inhaleend <= thephase < self.inhaleholdend:
            radval = self.maxrad
        elif self.inhaleholdend <= thephase < self.exhaleend:
            winstart = self.inhaleholdend
            winwidth = self.exhaleend - winstart
            radval = self.maxrad - radslope * (thephase - winstart) / winwidth
        elif self.exhaleend <= thephase < 1.0:
            radval = self.minrad
        else:
            print(f"{thephase} is not a legal phase value")
            sys.exit()
        therad = radval * self.scale
        if self.debug:
            print(f"Ph: {thephase}, Rad: {therad}")
        return therad

    def draw(self, thephase):
        therad = self.phasetorad(thephase)
        for i in range(self.lobes):
            angle = i * 2.0 * np.pi / self.lobes
            xloc = therad * np.sin(angle)
            yloc = therad * np.cos(angle)
            self.thecircles[i].setRadius(therad)
            self.thecircles[i].setPos([xloc, yloc])
            self.thecircles[i].draw()

    def getrespvalue(self, thephase):
        return (self.phasetorad(thephase) - self.minrad) / (
            self.maxrad - self.minrad
        ) - 0.5


class Boxbreathe:
    def __init__(self, xoffset=0.0, xscale=0.25, yoffset=0.0, yscale=0.25, debug=False):
        self.xoffset = xoffset
        self.xscale = xscale
        self.yoffset = yoffset
        self.yscale = yscale
        self.inhaleend = 0.25
        self.inhaleholdend = 0.5
        self.exhaleend = 0.75
        self.debug = debug
        self.initstim()

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
        elif self.inhaleend <= thephase < self.inhaleholdend:
            winstart = self.inhaleend
            winwidth = self.inhaleholdend - winstart
            xval = -0.5 + (thephase - winstart) / winwidth
            yval = 0.5
        elif self.inhaleholdend <= thephase < self.exhaleend:
            winstart = self.inhaleholdend
            winwidth = self.exhaleend - winstart
            xval = 0.5
            yval = 0.5 - (thephase - winstart) / winwidth
        elif self.exhaleend <= thephase < 1.0:
            winstart = self.exhaleend
            winwidth = 1.0 - winstart
            xval = 0.5 - (thephase - winstart) / winwidth
            yval = -0.5
        else:
            print(f"{thephase} is not a legal phase value")
            sys.exit()
        xpos = xval * self.xscale + self.xoffset
        ypos = yval * self.yscale + self.yoffset
        if self.debug:
            print(
                f"{thephase=}, {winstart=}, {winwidth=}, {xval=}, {yval=}, {xpos=}, {ypos=}"
            )
        return [xpos, ypos]

    def draw(self, thephase):
        self.target.setPos(self.phasetopos(thephase))
        self.frame.draw()
        self.target.draw()

    def getrespvalue(self, thephase):
        return self.phasetopos(thephase)[1]


class BreathingPattern:
    thetype = None
    thephase = None

    def __init__(self, thetype="boxbreathe_square", thephase=0.0, debug=False):
        self.debug = debug
        self.settype(thetype)
        self.setphase(thephase)

    def settype(self, thetype):
        self.thetype = thetype
        if self.thetype == "boxbreathe_square":
            self.stimulus = Boxbreathe(debug=self.debug)
        elif self.thetype == "boxbreathe_mindful":
            self.stimulus = Mindful(debug=self.debug)
        else:
            print("illegal stimulus type")
            sys.exit()

    def gettype(self):
        return self.thetype

    def setphase(self, thephase):
        self.thephase = thephase

    def getphase(self):
        return self.thephase

    def draw(self):
        self.stimulus.draw(self.thephase)

    def getrespvalue(self):
        return self.stimulus.getrespvalue(self.thephase)


def readexpfile(filename):
    with open(filename, "r") as json_data:
        d = json.load(json_data)
        for token in ["warntime", "fmritime", "TR"]:
            try:
                inval = float(d[token])
            except KeyError:
                print(f"{token} is not defined in input file - quitting")
                sys.exit()
        try:
            stimtype = d["stimtype"]
        except KeyError:
            print("stimtype is not defined in input file - quitting")
            sys.exit()
        try:
            waypoints = d["waypoints"]
        except KeyError:
            print("waypoints array is not defined in input file - quitting")
            sys.exit()
        else:
            if len(waypoints) < 1:
                print(
                    "waypoints array needs to be a list with at least one element - quitting"
                )
                sys.exit()
            for thewaypoint in waypoints:
                if len(thewaypoint) != 3:
                    print("each entry in waypoints array needs 3 values - quitting")
                    sys.exit()
            return (
                d["waypoints"],
                float(d["warntime"]),
                float(d["fmritime"]),
                float(d["TR"]),
                d["stimtype"],
            )
    print("file not found")
    sys.exit()


def makerespphasewaveform(waypointlist, timestep, expendtime):
    # waypointlist is assumed to be clean
    numpts = int(expendtime / timestep)
    phasevals = np.zeros((numpts), dtype=float)
    currentphase = 0.0
    startindex = int(waypointlist[0][0] / timestep)
    startcycletime = waypointlist[0][1]
    if waypointlist[0][2] >= 0.0:
        currentphase = waypointlist[0][2]
    for [endtime, endcycletime, endval] in waypointlist[1:]:
        endindex = np.min([int(endtime / timestep), numpts])
        if endval >= 0.0:
            currentphase = endval % 1.0
        phaseincrements = timestep * np.linspace(
            startcycletime / 60.0,
            endcycletime / 60.0,
            num=(endindex - startindex + 1),
            endpoint=False,
        )
        for i in range(startindex, endindex):
            phasevals[i] = (currentphase + phaseincrements[i - startindex]) % 1.0
            currentphase = phasevals[i]
    if endindex < numpts - 1:
        phasevals[endindex:-1] = phasevals[endindex - 1]
    return phasevals


def writebidstsv(indata, nameroot, colname, samplerate):
    columns = [colname]
    df = pd.DataFrame(data=np.transpose(indata), columns=columns)
    df.to_csv(
        nameroot + ".tsv.gz",
        sep="\t",
        compression="gzip",
        header=False,
        index=False,
    )
    headerdict = {}
    headerdict["SamplingFrequency"] = float(samplerate)
    headerdict["StartTime"] = float(0.0)
    headerdict["Columns"] = columns

    with open(nameroot + ".json", "wb") as fp:
        fp.write(
            json.dumps(
                headerdict, sort_keys=True, indent=4, separators=(",", ":")
            ).encode("utf-8")
        )


# execution starts here
# Configurable parameters
initpath = "/Users/frederic/code/breathingstim"
initfile = "risingtime.json"
debug = False  # turn on counter in upper righthand corner
targetscale = 0.25
fullscreen = False
warninglength = 3.0
nominaltimestep = 0.01

theexpfile = gui.fileOpenDlg(
    tryFilePath=initpath, tryFileName=initfile, allowed="*.json"
)
if theexpfile:
    waypoints, warntime, fmritime, TR, stimtype = readexpfile(theexpfile[0])
    print(waypoints, warntime, fmritime, TR, stimtype)
preamblelength = waypoints[0][0]

# settings for launchScan:
MR_settings = {
    "TR": TR,  # duration (sec) per volume
    "volumes": int(fmritime / TR),  # number of whole-brain 3D volumes / frames
    "sync": "t",  # character to use as the sync timing event; assumed to come at start of a volume
    "skip": 0,  # number of volumes lacking a sync pulse at start of scan (for T1 stabilization)
    "sound": False,  # in test mode only, play a tone as a reminder of scanner noise
}

# set a timestep that is ~0.1 seconds but is an integral divisor of TR
stepspertr = int(np.round(MR_settings["TR"] / nominaltimestep, 0))
timestep = MR_settings["TR"] / stepspertr
print("MR_settings initialization done")

print("Stimulus initialization started")
infoDlg = gui.DlgFromDict(MR_settings, title="fMRI parameters", order=["TR", "volumes"])
if not infoDlg.OK:
    core.quit()

phasevals = makerespphasewaveform(
    waypoints, timestep, MR_settings["TR"] * MR_settings["volumes"]
)
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

# make the breathing stim
print("initializing the class")
thebreathingstim = BreathingPattern(thetype=stimtype, debug=False)
print("done initializing the class")

# save the target respiratory waveform
respwaveroot = theexpfile[0].replace(".json", "_respwave")
respvals = phasevals * 0.0
for thisindex in range(phasevals.shape[0]):
    thebreathingstim.setphase(phasevals[thisindex])
    respvals[thisindex] = thebreathingstim.getrespvalue()
writebidstsv(respvals, respwaveroot, "respvals", 1.0 / timestep)

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
    plt.plot(phasevals)
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


# here's the main execution loop
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
    if thisindex > currentindex and thisindex < phasevals.shape[0]:
        # time to update the display
        if thisindex >= warningendindex:
            thebreathingstim.setphase(phasevals[thisindex])
            thebreathingstim.draw()
            thephase = thebreathingstim.getphase()
            counter.setText(f"Index: {thisindex:5d}, segtime: {thephase:.3f}")
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
