# -*- coding: utf-8 -*-

import json
import math
import os
import platform
import random
import re
import sys
import time


from collections import OrderedDict
from io import StringIO
import requests

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import matplotlib
#matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg', etc.
matplotlib.use('Agg')

__author__ = "Romain Gaillac and François-Xavier Coudert"
__version__ = "2019.01.09"
__license__ = "MIT"

###############################################################################
# 3D plotting functions 
###############################################################################

def make3DPlot(func, legend="", npoints=100):
    """
        Generates meshgrids to be plotted.
        
        Parameters
        ----------
        func : callable
            The is should be an attribute of the Elasitc class
        legend : str, optional
            string to be used in the legend. The default is "".
        npoints : int, optional
            The number of points in the meshgrids. The default is 100.
            
        Returns
        -------
        Returns : A tuple of list of lists of coordinate points. Simialr to numpy meshgrid.

    """

    str1 = legend.split("'")[0]
    str2 = legend.split("'")[1]

    u = np.linspace(0, np.pi, npoints)
    v = np.linspace(0, 2 * np.pi, 2 * npoints)
    r = np.zeros(len(u) * len(v))

    dataX = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataY = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataZ = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataR = [["0.0" for i in range(len(v))] for j in range(len(u))]

    count = 0
    for cu in range(len(u)):
        for cv in range(len(v)):

            r_tmp = func(u[cu], v[cv])
            z = r_tmp * np.cos(u[cu])
            x = r_tmp * np.sin(u[cu]) * np.cos(v[cv])
            y = r_tmp * np.sin(u[cu]) * np.sin(v[cv])
            dataX[cu][cv] = x
            dataY[cu][cv] = y
            dataZ[cu][cv] = z
            dataR[cu][cv] = (
                "'E = "
                + str(float(int(10 * r_tmp)) / 10.0)
                + " GPa, "
                + "\u03B8 = "
                + str(float(int(10 * u[cu] * 180 / np.pi)) / 10.0)
                + "\u00B0, "
                + "\u03c6 = "
                + str(float(int(10 * v[cv] * 180 / np.pi)) / 10.0)
                + "\u00B0'"
            )
            count = count + 1
    return (dataX, dataY, dataZ, dataR)

def make3DPlotPosNeg(func, legend="", npoints=100):
    """
        Generates positive and negative meshgrids to be plotted.
        
        Parameters
        ----------
        func : callable
            The is should be an attribute of the Elasitc class
        legend : str, optional
            string to be used in the legend. The default is "".
        npoints : int, optional
            The number of points in the meshgrids. The default is 100.
            
        Returns
        -------
        Returns : A size 2 tuple of tuples with list of lists of coordinate points. Simialr to numpy meshgrid. 
        The first index is for the positive value mesh. 
        The second index is for the negative value mesh

    """
    u = np.linspace(0, np.pi, npoints)
    v = np.linspace(0, 2 * np.pi, 2 * npoints)

    dataX1 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataY1 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataZ1 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataR1 = [["0.0" for i in range(len(v))] for j in range(len(u))]

    count = 0
    for cu in range(len(u)):
        for cv in range(len(v)):
            r_tmp = max(0, func(u[cu], v[cv]))
            z = r_tmp * np.cos(u[cu])
            x = r_tmp * np.sin(u[cu]) * np.cos(v[cv])
            y = r_tmp * np.sin(u[cu]) * np.sin(v[cv])
            dataX1[cu][cv] = x
            dataY1[cu][cv] = y
            dataZ1[cu][cv] = z
            dataR1[cu][cv] = (
                "'"
                + "\u03B2 = "
                + str(float(int(10 * r_tmp)) / 10.0)
                + " TPa'"
                + "+'-1'.sup()+"
                + "', \u03B8 = "
                + str(float(int(10 * u[cu] * 180 / np.pi)) / 10.0)
                + "\u00B0, "
                + "\u03c6 = "
                + str(float(int(10 * v[cv] * 180 / np.pi)) / 10.0)
                + "\u00B0'"
            )
            count = count + 1

    dataX2 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataY2 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataZ2 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataR2 = [["0.0" for i in range(len(v))] for j in range(len(u))]

    count = 0
    for cu in range(len(u)):
        for cv in range(len(v)):
            r_tmp = max(0, -func(u[cu], v[cv]))
            z = r_tmp * np.cos(u[cu])
            x = r_tmp * np.sin(u[cu]) * np.cos(v[cv])
            y = r_tmp * np.sin(u[cu]) * np.sin(v[cv])
            dataX2[cu][cv] = x
            dataY2[cu][cv] = y
            dataZ2[cu][cv] = z
            dataR2[cu][cv] = (
                "'"
                + "\u03B2 = -"
                + str(float(int(10 * r_tmp)) / 10.0)
                + " TPa'"
                + "+'-1'.sup()+"
                + "', \u03B8 = "
                + str(float(int(10 * u[cu] * 180 / np.pi)) / 10.0)
                + "\u00B0, "
                + "\u03c6 = "
                + str(float(int(10 * v[cv] * 180 / np.pi)) / 10.0)
                + "\u00B0'"
            )
            count = count + 1

    return ((dataX1, dataY1, dataZ1, dataR1), (dataX2, dataY2, dataZ2, dataR2))

def make3DPlot2(func, legend="", npoints=200):

    """
        Generates meshgrids for positive value and maximum values to be plotted.
        
        Parameters
        ----------
        func : callable
            The is should be an attribute of the Elasitc class
        legend : str, optional
            string to be used in the legend. The default is "".
        npoints : int, optional
            The number of points in the meshgrids. The default is 100.
            
        Returns
        -------
        Returns : A size 2 tuple of tuples with list of lists of coordinate points. Simialr to numpy meshgrid. 
        The first index is for the positive value mesh. 
        The second index is for the maximum value mesh.

    """
    u = np.linspace(0, np.pi, npoints)
    v = np.linspace(0, np.pi, npoints)
    w = [v[i] + np.pi for i in range(1, len(v))]
    v = np.append(v, w)

    dataX1 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataY1 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataZ1 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataR1 = [["0.0" for i in range(len(v))] for j in range(len(u))]

    dataX2 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataY2 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataZ2 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataR2 = [["0.0" for i in range(len(v))] for j in range(len(u))]

    count = 0
    r = [0.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    for cu in range(len(u)):
        for cv in range(len(v)):

            r = func(u[cu], v[cv], r[2], r[3])
            z = np.cos(u[cu])
            x = np.sin(u[cu]) * np.cos(v[cv])
            y = np.sin(u[cu]) * np.sin(v[cv])

            r1_tmp = r[0]
            z1 = r1_tmp * z
            x1 = r1_tmp * x
            y1 = r1_tmp * y
            dataX1[cu][cv] = x1
            dataY1[cu][cv] = y1
            dataZ1[cu][cv] = z1
            dataR1[cu][cv] = (
                "'"
                + "G'"
                + "+'min'.sub()+"
                + "' = "
                + str(float(int(10 * r1_tmp)) / 10.0)
                + "GPa, "
                + "\u03B8 = "
                + str(float(int(10 * u[cu] * 180 / np.pi)) / 10.0)
                + "\u00B0, "
                + "\u03c6 = "
                + str(float(int(10 * v[cv] * 180 / np.pi)) / 10.0)
                + "\u00B0'"
            )

            r2_tmp = r[1]
            z2 = r2_tmp * z
            x2 = r2_tmp * x
            y2 = r2_tmp * y
            dataX2[cu][cv] = x2
            dataY2[cu][cv] = y2
            dataZ2[cu][cv] = z2
            dataR2[cu][cv] = (
                "'"
                + "G'"
                + "+'max'.sub()+"
                + "' = "
                + str(float(int(10 * r1_tmp)) / 10.0)
                + "GPa, "
                + "\u03B8 = "
                + str(float(int(10 * u[cu] * 180 / np.pi)) / 10.0)
                + "\u00B0, "
                + "\u03c6 = "
                + str(float(int(10 * v[cv] * 180 / np.pi)) / 10.0)
                + "\u00B0'"
            )
            count = count + 1

    i = random.randint(0, 100000)
    return ((dataX1, dataY1, dataZ1, dataR1), (dataX2, dataY2, dataZ2, dataR2))

def make3DPlot3(func, legend="", width=600, height=600, npoints=100):
    """
        Generates positive, negative, and maximum vlaue meshgrids to be plotted.
        
        Parameters
        ----------
        func : callable
            The is should be an attribute of the Elasitc class
        legend : str, optional
            string to be used in the legend. The default is "".
        npoints : int, optional
            The number of points in the meshgrids. The default is 100.
            
        Returns
        -------
        Retruns : A size 3 tuple of tuples with list of lists of coordinate points. Simialr to numpy meshgrid. 
        The first index is for the negative value mesh. 
        The second index is for the positive value mesh.
        The thrid index is for the maximum value mesh.

    """
    
    str1 = legend.split("'")[0]
    str2 = legend.split("'")[1]

    u = np.linspace(0, np.pi, npoints)
    v = np.linspace(0, np.pi, npoints)
    w = [v[i] + np.pi for i in range(1, len(v))]
    v = np.append(v, w)

    dataX1 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataY1 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataZ1 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataR1 = [["0.0" for i in range(len(v))] for j in range(len(u))]

    dataX2 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataY2 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataZ2 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataR2 = [["0.0" for i in range(len(v))] for j in range(len(u))]

    dataX3 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataY3 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataZ3 = [[0.0 for i in range(len(v))] for j in range(len(u))]
    dataR3 = [["0.0" for i in range(len(v))] for j in range(len(u))]

    count = 0
    r = [0.0, 0.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    ruv = [[r for i in range(len(u))] for j in range(len(v))]
    for cu in range(len(u)):
        for cv in range(len(v)):
            ruv[cv][cu] = func(u[cu], v[cv], r[3], r[4])

    for cu in range(len(u)):
        for cv in range(len(v)):

            z = np.cos(u[cu])
            x = np.sin(u[cu]) * np.cos(v[cv])
            y = np.sin(u[cu]) * np.sin(v[cv])

            r = ruv[cv][cu]
            r1_tmp = r[0]
            dataX1[cu][cv] = r1_tmp * x
            dataY1[cu][cv] = r1_tmp * y
            dataZ1[cu][cv] = r1_tmp * z
            dataR1[cu][cv] = (
                "'"
                + "\u03BD'"
                + "+'min'.sub()+"
                + "' = "
                + str(float(int(100 * r1_tmp)) / 100.0)
                + ", "
                + "\u03B8 = "
                + str(float(int(100 * u[cu] * 180 / np.pi)) / 100.0)
                + "\u00B0, "
                + "\u03c6 = "
                + str(float(int(100 * v[cv] * 180 / np.pi)) / 100.0)
                + "\u00B0'"
            )

            r2_tmp = r[1]
            dataX2[cu][cv] = r2_tmp * x
            dataY2[cu][cv] = r2_tmp * y
            dataZ2[cu][cv] = r2_tmp * z
            dataR2[cu][cv] = float(int(100 * r2_tmp)) / 100.0

            r3_tmp = r[2]
            dataX3[cu][cv] = r3_tmp * x
            dataY3[cu][cv] = r3_tmp * y
            dataZ3[cu][cv] = r3_tmp * z
            dataR3[cu][cv] = (
                "'"
                + "\u03BD'"
                + "+'max'.sub()+"
                + "' = "
                + str(float(int(100 * r3_tmp)) / 100.0)
                + ", "
                + "\u03B8 = "
                + str(float(int(100 * u[cu] * 180 / np.pi)) / 100.0)
                + "\u00B0, "
                + "\u03c6 = "
                + str(float(int(100 * v[cv] * 180 / np.pi)) / 100.0)
                + "\u00B0'"
            )
            count = count + 1

    return (
        (dataX1, dataY1, dataZ1, dataR1),
        (dataX2, dataY2, dataZ2, dataR2),
        (dataX3, dataY3, dataZ3, dataR3),
    )

###############################################################################
# Polar plot functions
###############################################################################

def makePolarPlot(func, legend="", p="xy", npoints=200):
    """
        Generates coordinates points to be plotted.
        
        Parameters
        ----------
        func : callable
            The is should be an attribute of the Elasitc class
        legend : str, optional
            string to be used in the legend. The default is "".
        p : str, optional
            String to indicate what cross section to calculate. The default is "xy".
        npoints : int, optional
            The number of coordinate points. The default is 100.
            
        Returns
        -------
        Returns : A size 2 tuple of array of coordinate points

    """
    
    
    u = np.linspace(0, 2 * np.pi, npoints)
    r = list(map(func, u))
    if p == "xy":
        x = r * np.cos(u)
        y = r * np.sin(u)
    else:
        y = r * np.cos(u)
        x = r * np.sin(u)

    return (x, y)

def makePolarPlotPosNeg(func, legend="", p="xy", npoints=200):
    """
    Generates coordinates points for positive and negative values to be plotted.
    
    Parameters
    ----------
    func : callable
        The is should be an attribute of the Elasitc class
    legend : str, optional
        string to be used in the legend. The default is "".
    p : str, optional
        String to indicate what cross section to calculate. The default is "xy".
    npoints : int, optional
        The number of coordinate points. The default is 100.
        
    Returns
    -------
    Returns : A size 2 tuple of tuples containing coordinate point arrays.
    The first index cooresponds to the positive value.
    The second index cooresponds to the negative value.

    """
    
    
    u = np.linspace(0, 2 * np.pi, npoints)
    r = list(map(lambda x: max(0, func(x)), u))
    if p == "xy":
        x1 = r * np.cos(u)
        y1 = r * np.sin(u)
    else:
        y1 = r * np.cos(u)
        x1 = r * np.sin(u)
    r = list(map(lambda x: max(0, -func(x)), u))
    if p == "xy":
        x2 = r * np.cos(u)
        y2 = r * np.sin(u)
    else:
        y2 = r * np.cos(u)
        x2 = r * np.sin(u)

    return ((x1, y1), (x2, y2))

def makePolarPlot2(func, legend="", p="xy", npoints=200):
    """
    Generates coordinates points for positive and maximum values to be plotted.
    
    Parameters
    ----------
    func : callable
        The is should be an attribute of the Elasitc class
    legend : str, optional
        string to be used in the legend. The default is "".
    p : str, optional
        String to indicate what cross section to calculate. The default is "xy".
    npoints : int, optional
        The number of coordinate points. The default is 100.
        
    Returns
    -------
    Returns : A size 2 tuple of tuples containing coordinate point arrays
    The first index cooresponds to the positive value.
    The second index cooresponds to the maximum value.

    """
    
    
    u = np.linspace(0, 2 * np.pi, npoints)
    r = list(map(func, u))

    if p == "xy":
        x1 = np.array([ir[0] * np.cos(iu) for ir, iu in zip(r, u)])
        y1 = np.array([ir[0] * np.sin(iu) for ir, iu in zip(r, u)])
        x2 = np.array([ir[1] * np.cos(iu) for ir, iu in zip(r, u)])
        y2 = np.array([ir[1] * np.sin(iu) for ir, iu in zip(r, u)])
    else:
        y1 = np.array([ir[0] * np.cos(iu) for ir, iu in zip(r, u)])
        x1 = np.array([ir[0] * np.sin(iu) for ir, iu in zip(r, u)])
        y2 = np.array([ir[1] * np.cos(iu) for ir, iu in zip(r, u)])
        x2 = np.array([ir[1] * np.sin(iu) for ir, iu in zip(r, u)])

    return ((x1, y1), (x2, y2))

def makePolarPlot3(func, legend="", p="xy", npoints=200):
    """
    Generates coordinates points for positive, negative, and maximum values to be plotted.
    
    Parameters
    ----------
    func : callable
        The is should be an attribute of the Elasitc class
    legend : str, optional
        string to be used in the legend. The default is "".
    p : str, optional
        String to indicate what cross section to calculate. The default is "xy".
    npoints : int, optional
        The number of coordinate points. The default is 100.
        
    Returns
    -------
    Returns : A size 3 tuple of tuples containing coordinate point arrays
    The first index cooresponds to the negative value.
    The second index cooresponds to the positive value.
    The third index cooresponds to the maximum value.

    """
    u = np.linspace(0, 2 * np.pi, npoints)
    r = list(map(func, u))

    if p == "xy":
        x1 = np.array([ir[0] * np.cos(iu) for ir, iu in zip(r, u)])
        y1 = np.array([ir[0] * np.sin(iu) for ir, iu in zip(r, u)])
        x2 = np.array([ir[1] * np.cos(iu) for ir, iu in zip(r, u)])
        y2 = np.array([ir[1] * np.sin(iu) for ir, iu in zip(r, u)])
        x3 = np.array([ir[2] * np.cos(iu) for ir, iu in zip(r, u)])
        y3 = np.array([ir[2] * np.sin(iu) for ir, iu in zip(r, u)])
    else:
        y1 = np.array([ir[0] * np.cos(iu) for ir, iu in zip(r, u)])
        x1 = np.array([ir[0] * np.sin(iu) for ir, iu in zip(r, u)])
        y2 = np.array([ir[1] * np.cos(iu) for ir, iu in zip(r, u)])
        x2 = np.array([ir[1] * np.sin(iu) for ir, iu in zip(r, u)])
        y3 = np.array([ir[2] * np.cos(iu) for ir, iu in zip(r, u)])
        x3 = np.array([ir[2] * np.sin(iu) for ir, iu in zip(r, u)])

    return ((x1, y1), (x2, y2), (x3, y3))


###############################################################################


def dirVec(theta, phi):
    """
        Generates a directional vector given spherical angles
        
        Parameters
        ----------
        theta : float
            Angle with the z axis
        phi : float, optional
            Angle with the x axis axis

        Returns
        -------
        Returns : Size 3 list representing direction vector

    """
    return [
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    ]


def dirVec1(theta, phi, chi):
    """
        Generates a directional vector given spherical angles
        
        Parameters
        ----------
        theta : float
            Angle with the z axis
        phi : float, optional
            Angle with the x axis

        Returns
        -------
        Returns : Size 3 list representing direction vector

    """
    return [
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    ]


def dirVec2(theta, phi, chi):
    """
        Generates a directional vector given euler rotation angles
        
        Parameters
        ----------
        theta : float
            Angle with the z axis
        phi : float, optional
            Angle with the x axis
        chi : float, optional
            Angle with the y axis

        Returns
        -------
        Returns : Size 3 list representing direction vector

    """
    return [
        math.cos(theta) * math.cos(phi) * math.cos(chi) - math.sin(phi) * math.sin(chi),
        math.cos(theta) * math.sin(phi) * math.cos(chi) + math.cos(phi) * math.sin(chi),
        -math.sin(theta) * math.cos(chi),
    ]


# Functions to minimize/maximize
def minimize(func, dim):
    """
        Finds the spherical angles corresponding to a minimum value
        
        Parameters
        ----------
        func : callable
            The is should be an attribute of the Elasitc class
        dim : int
            The dimension of the property

        Returns
        -------
        Returns : Tuple of spherical angles of the minimal direction

    """
    if dim == 2:
        r = ((0, np.pi), (0, np.pi))
        n = 25
    elif dim == 3:
        r = ((0, np.pi), (0, np.pi), (0, np.pi))
        n = 10
    
    # TODO -- try basin hopping or annealing
    return optimize.brute(func, r, Ns=n, full_output=True, finish=optimize.fmin)[0:2]


def maximize(func, dim):
    """
        Finds the spherical angles corresponding to a maximum value
        
        Parameters
        ----------
        func : callable
            The is should be an attribute of the Elasitc class
        dim : int
            The dimension of the property

        Returns
        -------
        Returns : Tuple of spherical angles of the minimal direction

    """
    res = minimize(lambda x: -func(x), dim)
    return (res[0], -res[1])


class ELATE:
    def __init__(self, s,density = None):

        self.elas = Elastic(s, density = density)
        self.elasList = s
        self.density = density
        minE = minimize(self.elas.Young, 2)
        maxE = maximize(self.elas.Young, 2)
        minLC = minimize(self.elas.LC, 2)
        maxLC = maximize(self.elas.LC, 2)
        minG = minimize(self.elas.shear, 3)
        maxG = maximize(self.elas.shear, 3)
        minNu = minimize(self.elas.Poisson, 3)
        maxNu = maximize(self.elas.Poisson, 3)
        minPugh = minimize(self.elas.Pugh_ratio, 3)
        maxPugh = maximize(self.elas.Pugh_ratio, 3)
        minK = minimize(self.elas.Bulk, 3)
        maxK = maximize(self.elas.Bulk, 3)
    

        
        self.voigtE = self.elas.averages()[0][1]
        self.reussE = self.elas.averages()[1][1]
        self.hillE = self.elas.averages()[2][1]
        self.max_E = maxE[1]
        self.min_E = minE[1]
        self.min_axis_E = tuple(dirVec(*minE[0]))
        self.max_axis_E = tuple(dirVec(*maxE[0]))
        self.anis_E = maxE[1] / minE[1]

        self.max_LC = maxLC[1]
        self.min_LC = minLC[1]
        self.min_axis_LC = tuple(dirVec(*minLC[0]))
        self.max_axis_LC = tuple(dirVec(*maxLC[0]))
        if minLC[1] > 0:
            self.anis_LC = maxLC[1] / minLC[1]
        else:
            self.anis_LC = "&infin;"

        self.voigtShear = self.elas.averages()[0][2]
        self.reussShear = self.elas.averages()[1][2]
        self.hillShear = self.elas.averages()[2][2]
        self.max_Shear = maxG[1]
        self.min_Shear = minG[1]
        self.min_axis_Shear = tuple(dirVec1(*minG[0]))
        self.max_axis_Shear = tuple(dirVec1(*maxG[0]))
        self.mix_2nd_axis_Shear = tuple(dirVec2(*minG[0]))
        self.max_2nd_axis_Shear = tuple(dirVec2(*maxG[0]))
        self.anis_Shear = maxG[1] / minG[1]

        self.voigtPoisson = self.elas.averages()[0][3]
        self.reussPoisson = self.elas.averages()[1][3]
        self.hillPoisson = self.elas.averages()[2][3]
        self.max_Poisson = maxNu[1]
        self.min_Poisson = minNu[1]
        self.min_axis_Poisson = tuple(dirVec1(*minNu[0]))
        self.max_axis_Poisson = tuple(dirVec1(*maxNu[0]))
        self.min_2nd_axis_Poisson = tuple(dirVec2(*minNu[0]))
        self.max_2nd_axis_Poisson = tuple(dirVec2(*maxNu[0]))
        if minNu[1] * maxNu[1] > 0:
            self.anis_Poisson = maxNu[1] / minNu[1]
        else:
            self.anis_Poisson = "&infin;"
            
        self.voigtPugh = self.elas.averages()[0][0]/self.elas.averages()[0][2]
        self.reussPugh = self.elas.averages()[1][0]/self.elas.averages()[1][2]
        self.hillPugh = self.elas.averages()[2][0]/self.elas.averages()[2][2]
        self.max_Pugh = maxPugh[1]
        self.min_Pugh = minPugh[1]
        self.min_axis_Pugh = tuple(dirVec1(*minPugh[0]))
        self.max_axis_Pugh = tuple(dirVec1(*maxPugh[0]))
        self.mix_2nd_axis_Pugh = tuple(dirVec2(*minPugh[0]))
        self.max_2nd_axis_Pugh = tuple(dirVec2(*maxPugh[0]))
        self.anis_Pugh = maxPugh[1] / minPugh[1]
        
        
        self.voigtK = self.elas.averages()[0][0]
        self.reussK = self.elas.averages()[1][0]
        self.hillK = self.elas.averages()[2][0]
        self.max_K = maxK[1]
        self.min_K = minK[1]
        self.min_axis_K = tuple(dirVec1(*minK[0]))
        self.max_axis_K = tuple(dirVec1(*maxK[0]))
        self.mix_2nd_axis_K = tuple(dirVec2(*minK[0]))
        self.max_2nd_axis_K = tuple(dirVec2(*maxK[0]))
        self.anis_K = maxK[1] / minK[1]
            

        self.title_width = len("Young's Modulus")  # initializing with expected longest title
        self.unit_width = len("(N/m,Experimental)")  # initializing with expected longest unit
        self.value_width = len("Anisotropy")  # generic starting width
        self.anisotropy_width = len("Anisotropy")  # as per your sample
            
        if self.density != None:
            # print(type(self.elas.Compression_Speed()))
            minVc = minimize(self.elas.Compression_Speed, 3)
            maxVc = maximize(self.elas.Compression_Speed, 3)
            
            minVs = minimize(self.elas.Shear_Speed, 3)
            maxVs = maximize(self.elas.Shear_Speed, 3)
            
            minRatio = minimize(self.elas.Ratio_Compression_Shear, 3)
            maxRatio = maximize(self.elas.Ratio_Compression_Shear , 3)
            
            minDebyeSpeed = minimize(self.elas.Debye_Speed, 3)
            maxDebyeSpeed = maximize(self.elas.Debye_Speed, 3) 
           
            self.voigtCompressionSpeed = ((10**9)*(self.elas.averages()[0][0]+(4/3)*self.elas.averages()[0][2])/self.density)**0.5
            self.reussCompressionSpeed = ((10**9)*(self.elas.averages()[1][0]+(4/3)*self.elas.averages()[1][2])/self.density)**0.5
            self.hillCompressionSpeed  = ((10**9)*(self.elas.averages()[2][0]+(4/3)*self.elas.averages()[2][2])/self.density)**0.5
            self.max_CompressionSpeed  = maxVc[1]
            self.min_CompressionSpeed  = minVc[1]
            self.min_axis_CompressionSpeed = tuple(dirVec1(*minVc[0]))
            self.max_axis_CompressionSpeed = tuple(dirVec1(*maxVc[0]))
            self.anis_CompressionSpeed = maxVc[1] / minVc[1]
            
            self.voigtShearSpeed = ((self.elas.averages()[0][2]*(10**9))/self.density)**0.5
            self.reussShearSpeed = ((self.elas.averages()[1][2]*(10**9))/self.density)**0.5
            self.hillShearSpeed =  ((self.elas.averages()[2][2]*(10**9))/self.density)**0.5
            self.max_ShearSpeed = maxVs[1]
            self.min_ShearSpeed = minVs[1]
            self.min_axis_ShearSpeed = tuple(dirVec1(*minVs[0]))
            self.max_axis_ShearSpeed = tuple(dirVec1(*maxVs[0]))
            self.anis_ShearSpeed = maxVs[1] / minVs[1]
            
            self.voigtRatioCompressionShearSpeed = (self.voigtCompressionSpeed /self.voigtShearSpeed )**2
            self.reussRatioCompressionShearSpeed = (self.reussCompressionSpeed /self.reussShearSpeed )**2
            self.hillRatioCompressionShearSpeed =  (self.hillCompressionSpeed /self.hillShearSpeed )**2
            self.max_RatioCompressionShearSpeed = maxRatio[1]
            self.min_RatioCompressionShearSpeed = minRatio[1]
            self.min_axis_RatioCompressionShearSpeed = tuple(dirVec1(*minRatio[0]))
            self.max_axis_RatioCompressionShearSpeed = tuple(dirVec1(*maxRatio[0]))
            self.anis_RatioCompressionShearSpeed = maxRatio[1] / minRatio[1]
            
            self.voigtDebyeSpeed = self.voigtCompressionSpeed* self.voigtShearSpeed/(2* self.voigtShearSpeed**3 + self.voigtCompressionSpeed**3)**(1/3)
            self.reussDebyeSpeed = self.reussCompressionSpeed* self.reussShearSpeed/(2* self.reussShearSpeed**3 + self.reussCompressionSpeed**3)**(1/3)
            self.hillDebyeSpeed =  self.hillCompressionSpeed* self.hillShearSpeed/(2* self.hillShearSpeed**3 + self.hillCompressionSpeed**3)**(1/3)
            self.max_DebyeSpeed = maxDebyeSpeed[1]
            self.min_DebyeSpeed = minDebyeSpeed[1]
            self.min_axis_DebyeSpeed = tuple(dirVec1(*minDebyeSpeed[0]))
            self.max_axis_DebyeSpeed = tuple(dirVec1(*maxDebyeSpeed[0]))
            self.anis_DebyeSpeed = maxDebyeSpeed[1] / minDebyeSpeed[1]
            
            # Set it to print in km/s
            self.voigtCompressionSpeed /= 1000  # Convert from m/s to km/s
            self.reussCompressionSpeed /= 1000
            self.hillCompressionSpeed /= 1000
            self.voigtShearSpeed /= 1000
            self.reussShearSpeed /= 1000
            self.hillShearSpeed /= 1000

            self.voigtDebyeSpeed /= 1000
            self.reussDebyeSpeed /= 1000
            self.hillDebyeSpeed /= 1000




    """
        The container class object of the ELATE Analysis.
        
        Parameters
        ----------
        s : List of lists
            This is a list of rows of the elastic tensor
        density : float, optional
            The density of a material in units of kg/m^3

    """



    def YOUNG2D(self, npoints):
        """
        Generates a cross sectional data for the young's modulus
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns : A size 3 tuple containing the xy,xz,yz cross sections. 
            Each cross section is a size 2 tuple of array of coordinate points.
            The first index is the xy plane.
            The second index is the xz plane.
            The third index is the yz plane.

        """
        data1 = makePolarPlot(
            lambda x: self.elas.Young([np.pi / 2, x]),
            "Young's modulus in (xy) plane",
            "xy",
            npoints=npoints,
        )
        data2 = makePolarPlot(
            lambda x: self.elas.Young([x, 0]),
            "Young's modulus in (xz) plane",
            "xz",
            npoints=npoints,
        )
        data3 = makePolarPlot(
            lambda x: self.elas.Young([x, np.pi / 2]),
            "Young's modulus in (yz) plane",
            "yz",
            npoints=npoints,
        )
        return (data1, data2, data3)

    def YOUNG3D(self, npoints):
        """
        Generates a meshgrid for the young's modulus
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns : A tuple of list of lists of coordinate points. Simialr to numpy meshgrid.

        """
        if self.elas.isOrthorhombic():
            self.elas = ElasticOrtho(self.elas)

        data = make3DPlot(
            lambda x, y: self.elas.Young_2(x, y), "Young's modulus", npoints=npoints
        )

        return data

    def LC2D(self, npoints):
        """
        Generates a cross sectional data for the Linear Compression
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns : A size 3 tuple containing the xy,xz,yz cross sections
            The first index is the xy plane.
            The second index is the xz plane.
            The third index is the yz plane.
            
            Each cross section is a size 2 tuple of tuples containing coordinate point arrays.
            The first index cooresponds to the positive value.
            The second index cooresponds to the negative value.


        """
        data1 = makePolarPlotPosNeg(
            lambda x: self.elas.LC([np.pi / 2, x]),
            "linear compressibility in (xy) plane",
            "xy",
            npoints=npoints,
        )
        data2 = makePolarPlotPosNeg(
            lambda x: self.elas.LC([x, 0]),
            "linear compressibility in (xz) plane",
            "xz",
            npoints=npoints,
        )
        data3 = makePolarPlotPosNeg(
            lambda x: self.elas.LC([x, np.pi / 2]),
            "linear compressibility in (yz) plane",
            "yz",
            npoints=npoints,
        )
        return (data1, data2, data3)

    def LC3D(self, npoints):
        """
        Generates a meshgrid for the Linear Compression
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns : A size 2 tuple of tuples with list of lists of coordinate points. Simialr to numpy meshgrid. 
            The first index is for the positive value mesh. 
            The second index is for the negative value mesh

        """

        if self.elas.isOrthorhombic():
            self.elas = ElasticOrtho(self.elas)

        data = make3DPlotPosNeg(
            lambda x, y: self.elas.LC_2(x, y), "Linear compressiblity", npoints=npoints
        )

        return data

    def BULK3D(self, npoints):
        """
        Generates a meshgrid for the Bulk Modulus
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns :  A size 2 tuple of tuples with list of lists of coordinate points. Simialr to numpy meshgrid. 
            The first index is for the positive value mesh. 
            The second index is for the maximum value mesh.

        """

        if self.elas.isOrthorhombic():
            self.elas = ElasticOrtho(self.elas)

        data = make3DPlot2(
            lambda x, y, g1, g2: self.elas.bulk3D(x, y, g1, g2),
            "Shear modulus",
            npoints=npoints,
        )

        return data

    def BULK2D(self, npoints):
        """
        Generates a cross sectional data for the Bulk modulus
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns : A size 3 tuple containing the xy,xz,yz cross sections
            The first index is the xy plane.
            The second index is the xz plane.
            The third index is the yz plane.
            
            Each cross section is a size 2 tuple of tuples containing coordinate point arrays
            The first index cooresponds to the positive value.
            The second index cooresponds to the maximum value.
        """
        data1 = makePolarPlot2(
            lambda x: self.elas.bulk2D([np.pi / 2, x]),
            "Shear modulus in (xy) plane",
            "xy",
            npoints=npoints,
        )
        data2 = makePolarPlot2(
            lambda x: self.elas.bulk2D([x, 0]), "Shear modulus in (xz) plane", "xz"
        )
        data3 = makePolarPlot2(
            lambda x: self.elas.bulk2D([x, np.pi / 2]),
            "Shear modulus in (yz) plane",
            "yz",
            npoints=npoints,
        )
        return (data1, data2, data3)

    def SHEAR2D(self, npoints):
        """
        Generates a cross sectional data for the Shear modulus
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns : A size 3 tuple containing the xy,xz,yz cross sections
            The first index is the xy plane.
            The second index is the xz plane.
            The third index is the yz plane.
            
            Each cross section is a size 2 tuple of tuples containing coordinate point arrays
            The first index cooresponds to the positive value.
            The second index cooresponds to the maximum value.
        """
        data1 = makePolarPlot2(
            lambda x: self.elas.shear2D([np.pi / 2, x]),
            "Shear modulus in (xy) plane",
            "xy",
            npoints=npoints,
        )
        data2 = makePolarPlot2(
            lambda x: self.elas.shear2D([x, 0]), "Shear modulus in (xz) plane", "xz"
        )
        data3 = makePolarPlot2(
            lambda x: self.elas.shear2D([x, np.pi / 2]),
            "Shear modulus in (yz) plane",
            "yz",
            npoints=npoints,
        )
        return (data1, data2, data3)

    def SHEAR3D(self, npoints):
        """
        Generates a meshgrid for the Shear Modulus
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns :  A size 2 tuple of tuples with list of lists of coordinate points. Simialr to numpy meshgrid. 
            The first index is for the positive value mesh. 
            The second index is for the maximum value mesh.

        """

        if self.elas.isOrthorhombic():
            self.elas = ElasticOrtho(self.elas)

        data = make3DPlot2(
            lambda x, y, g1, g2: self.elas.shear3D(x, y, g1, g2),
            "Shear modulus",
            npoints=npoints,
        )

        return data

    def POISSON2D(self, npoints):
        """
        Generates a cross sectional data for the Poisson's Ratio
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns : A size 3 tuple containing the xy,xz,yz cross sections.
            The first index is the xy plane.
            The second index is the xz plane.
            The third index is the yz plane.
        
            Each cross section is a size 3 tuple of tuples containing coordinate point arrays
            The first index cooresponds to the negative value.
            The second index cooresponds to the positive value.
            The third index cooresponds to the maximum value.

        """
        data1 = makePolarPlot3(
            lambda x: self.elas.Poisson2D([np.pi / 2, x]),
            "Poisson's ratio in (xy) plane",
            "xy",
            npoints=npoints,
        )
        data2 = makePolarPlot3(
            lambda x: self.elas.Poisson2D([x, 0]),
            "Poisson's ratio in (xz) plane",
            "xz",
            npoints=npoints,
        )
        data3 = makePolarPlot3(
            lambda x: self.elas.Poisson2D([x, np.pi / 2]),
            "Poisson's ratio in (yz) plane",
            "yz",
            npoints=npoints,
        )
        return (data1, data2, data3)

    def POISSON3D(self, npoints):
        """
        Generates a meshgrid for the Poisson's Ratio
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns :  A size 3 tuple of tuples with list of lists of coordinate points. Simialr to numpy meshgrid. 
            The first index is for the negative value mesh. 
            The second index is for the positive value mesh.
            The thrid index is for the maximum value mesh.
        """

        if self.elas.isOrthorhombic():
            self.elas = ElasticOrtho(self.elas)

        data = make3DPlot3(
            lambda x, y, g1, g2: self.elas.poisson3D(x, y, g1, g2),
            "Poisson's ratio",
            npoints=npoints,
        )

        return data
    
    def PUGH_RATIO2D(self, npoints):
        """
        Generates a cross sectional data for the Pugh's Ratio
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns : A size 3 tuple containing the xy,xz,yz cross sections
            The first index is the xy plane.
            The second index is the xz plane.
            The third index is the yz plane.
            
            Each cross section is a size 2 tuple of tuples containing coordinate point arrays
            The first index cooresponds to the positive value.
            The second index cooresponds to the maximum value.
        """
        data1 = makePolarPlot2(
            lambda x: self.elas.pugh_ratio2D([np.pi / 2, x]),
            "Pugh's Ratio in (xy) plane",
            "xy",
            npoints=npoints,
        )
        data2 = makePolarPlot2(
            lambda x: self.elas.pugh_ratio2D([x, 0]), "Pugh's Ratio in (xz) plane", "xz"
        )
        data3 = makePolarPlot2(
            lambda x: self.elas.pugh_ratio2D([x, np.pi / 2]),
            "Pugh's Ratio in (yz) plane",
            "yz",
            npoints=npoints,
        )
        return (data1, data2, data3)
    
    def PUGH_RATIO3D(self, npoints):
        """
        Generates a meshgrid for the Pugh Ratio
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns :  A size 2 tuple of tuples with list of lists of coordinate points. Simialr to numpy meshgrid. 
            The first index is for the positive value mesh. 
            The second index is for the maximum value mesh.

        """

        if self.elas.isOrthorhombic():
            self.elas = ElasticOrtho(self.elas)

        data = make3DPlot2(
            lambda x, y, g1, g2: self.elas.pugh_ratio3D(x, y, g1, g2),
            "Pugh's Ratio",
            npoints=npoints,
        )
        return data
    
    def COMPRESSION_SPEED2D(self, npoints):
        """
        Generates a cross sectional data for the Compression Speed
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns : A size 3 tuple containing the xy,xz,yz cross sections
            The first index is the xy plane.
            The second index is the xz plane.
            The third index is the yz plane.
            
            Each cross section is a size 2 tuple of tuples containing coordinate point arrays
            The first index cooresponds to the positive value.
            The second index cooresponds to the maximum value.

        """
        data1 = makePolarPlot2(
            lambda x: self.elas.compressionSpeed2D([np.pi / 2, x],density = self.density),
            "Compression Speed in (xy) plane",
            "xy",
            npoints=npoints,
        )
        data2 = makePolarPlot2(
            lambda x: self.elas.compressionSpeed2D([x, 0],density = self.density), "Compression Speed in (xz) plane", "xz"
        )
        data3 = makePolarPlot2(
            lambda x: self.elas.compressionSpeed2D([x, np.pi / 2], density = self.density),
            "Compression Speed in (yz) plane",
            "yz",
            npoints=npoints,
        )
        return (data1, data2, data3)
    
    def COMPRESSION_SPEED3D(self, npoints):
        """
        Generates a meshgrid for the Compression speed
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns :  A size 2 tuple of tuples with list of lists of coordinate points. Simialr to numpy meshgrid. 
            The first index is for the positive value mesh. 
            The second index is for the maximum value mesh.

        """
        if self.elas.isOrthorhombic():
            self.elas = ElasticOrtho(self.elas)

        data = make3DPlot2(
            lambda x, y, g1, g2: self.elas.compressionSpeed3D(x, y, g1, g2, density = self.density),
            "Compression speed",
            npoints=npoints,
        )

        return data
    
    def SHEAR_SPEED2D(self, npoints):
        """
        Generates a cross sectional data for the Shear Speed
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns : A size 3 tuple containing the xy,xz,yz cross sections
            The first index is the xy plane.
            The second index is the xz plane.
            The third index is the yz plane.
            
            Each cross section is a size 2 tuple of tuples containing coordinate point arrays
            The first index cooresponds to the positive value.
            The second index cooresponds to the maximum value.

        """
        data1 = makePolarPlot2(
            lambda x: self.elas.shearSpeed2D([np.pi / 2, x],density = self.density),
            "Shear speed in (xy) plane",
            "xy",
            npoints=npoints,
        )
        data2 = makePolarPlot2(
            lambda x: self.elas.shearSpeed2D([x, 0],density = self.density), "Shear speed in (xz) plane", "xz"
        )
        data3 = makePolarPlot2(
            lambda x: self.elas.shearSpeed2D([x, np.pi / 2],density = self.density),
            "Shear speed in (yz) plane",
            "yz",
            npoints=npoints,
        )
        return (data1, data2, data3)
    
    def SHEAR_SPEED3D(self, npoints):
        """
        Generates a meshgrid for the Shear Speed
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns :  A size 2 tuple of tuples with list of lists of coordinate points. Simialr to numpy meshgrid. 
            The first index is for the positive value mesh. 
            The second index is for the maximum value mesh.

        """
        if self.elas.isOrthorhombic():
            self.elas = ElasticOrtho(self.elas)

        data = make3DPlot2(
            lambda x, y, g1, g2: self.elas.shearSpeed3D(x, y, g1, g2, density = self.density),
            "Shear speed",
            npoints=npoints,
        )
        return data
    
    def RATIO_COMPRESSIONAL_SHEAR2D(self, npoints):
        """
        Generates a cross sectional data for the Ratio of the Compression Speed 
        to the Shear Speed
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns : A size 3 tuple containing the xy,xz,yz cross sections
            The first index is the xy plane.
            The second index is the xz plane.
            The third index is the yz plane.
            
            Each cross section is a size 2 tuple of tuples containing coordinate point arrays
            The first index cooresponds to the positive value.
            The second index cooresponds to the maximum value.

        """
        data1 = makePolarPlot2(
            lambda x: self.elas. ratio_compressional_shear2D([np.pi / 2, x],density = self.density),
            "Ratio compressional/shear in (xy) plane",
            "xy",
            npoints=npoints,
        )
        data2 = makePolarPlot2(
            lambda x: self.elas. ratio_compressional_shear2D([x, 0],density = self.density), "Ratio compressional/shear in (xz) plane", "xz"
        )
        data3 = makePolarPlot2(
            lambda x: self.elas. ratio_compressional_shear2D([x, np.pi / 2],density = self.density),
            "Ratio compressional/shear in (yz) plane",
            "yz",
            npoints=npoints,
        )
        return (data1, data2, data3)
        
    def RATIO_COMPRESSIONAL_SHEAR3D(self, npoints):
        """
        Generates a meshgrid for the Ratio of the compressional to shear speed
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns :  A size 2 tuple of tuples with list of lists of coordinate points. Simialr to numpy meshgrid. 
            The first index is for the positive value mesh. 
            The second index is for the maximum value mesh.

        """
        if self.elas.isOrthorhombic():
            self.elas = ElasticOrtho(self.elas)

        data = make3DPlot2(
            lambda x, y, g1, g2: self.elas.ratio_compressional_shear3D(x, y, g1, g2,density = self.density),
            "Ratio of compression/shear",
            npoints=npoints,
        )
        return data
    
    def DEBYE_SPEED2D(self, npoints):
        """
        Generates a cross sectional data for the Debye Speed
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns : A size 3 tuple containing the xy,xz,yz cross sections
            The first index is the xy plane.
            The second index is the xz plane.
            The third index is the yz plane.
            
            Each cross section is a size 2 tuple of tuples containing coordinate point arrays
            The first index cooresponds to the positive value.
            The second index cooresponds to the maximum value.

        """
        data1 = makePolarPlot2(
            lambda x: self.elas.debyeSpeed2D([np.pi / 2, x],density = self.density),
            "Debye speed in (xy) plane",
            "xy",
            npoints=npoints,
        )
        data2 = makePolarPlot2(
            lambda x: self.elas.debyeSpeed2D([x, 0], density = self.density), "Debye speed in (xz) plane", "xz"
        )
        data3 = makePolarPlot2(
            lambda x: self.elas.debyeSpeed2D([x, np.pi / 2],density = self.density),
            "Debye speed in (yz) plane",
            "yz",
            npoints=npoints,
        )
        return (data1, data2, data3)
    
    def DEBYE_SPEED3D(self, npoints):
        """
        Generates a meshgrid for the Debye Speed
        
        Parameters
        ----------
        npoints : int
            The number of coordinate points. The default is 100.

        Returns
        -------
        Returns :  A size 2 tuple of tuples with list of lists of coordinate points. Simialr to numpy meshgrid. 
            The first index is for the positive value mesh. 
            The second index is for the maximum value mesh.

        """

        if self.elas.isOrthorhombic():
            self.elas = ElasticOrtho(self.elas)

        data = make3DPlot2(
            lambda x, y, g1, g2: self.elas.debyeSpeed3D(x, y, g1, g2,density = self.density),
            "Debye Speed",
            npoints=npoints,
        )

        return data
    
    def to_dict(self):
        """
        Generates a dictionary of all the ELATE analysis values
        

        Returns
        -------
        Returns :  dict

        """
        
        tmp_dict =  {
                   'bulk_modulus_voigt':self.voigtK,
                   'bulk_modulus_reuss':self.reussK,
                   'bulk_modulus_hill':self.hillK,
                   'bulk_max':self.max_K,
                   'bulk_min':self.min_K,
                   'bulk_min_axis':self.min_axis_K,
                   'bulk_max_axis':self.max_axis_K,
                   'bulk_min_axis_2':self.mix_2nd_axis_K,
                   'bulk_max_axis_2':self.max_2nd_axis_K,
                   'bulk_anisotropy':self.anis_K,
    
                   'youngs_modulus_voigt':self.voigtE,
                   'youngs_modulus_reuss':self.reussE,
                   'youngs_modulus_hill':self.hillE,
                   'youngs_max':self.max_E,
                   'youngs_min':self.min_E,
                   'youngs_min_axis':self.min_axis_E, 
                   'youngs_max_axis':self.max_axis_E,
                   'youngs_anisotropy':self.anis_E,


                   'linearCompression_max':self.max_LC,
                   'linearCompression_min':self.min_LC,
                   'linearCompression_min_axis':self.min_axis_LC,
                   'linearCompression_max_axis':self.max_axis_LC,
                   'linearCompression_anisotropy':self.anis_LC,

            
                   'shear_modulus_voigt':self.voigtShear,
                   'shear_modulus_reuss':self.reussShear,
                   'shear_modulus_hill':self.hillShear,
                   'shear_max':self.max_Shear,
                   'shear_min':self.min_Shear,
                   'shear_min_axis':self.min_axis_Shear,
                   'shear_max_axis':self.max_axis_Shear,
                   'shear_min_axis_2':self.mix_2nd_axis_Shear,
                   'shear_max_axis_2':self.max_2nd_axis_Shear,
                   'shear_anisotropy':self.anis_Shear,

            
                   'poisson_modulus_voigt':self.voigtPoisson,
                   'poisson_modulus_reuss':self.reussPoisson,
                   'poisson_modulus_hill':self.hillPoisson,
                   'poisson_max':self.max_Poisson,
                   'poisson_min':self.min_Poisson,
                   'poisson_min_axis':self.min_axis_Poisson,
                   'poisson_max_axis':self.max_axis_Poisson,
                   'poisson_min_axis_2':self.min_2nd_axis_Poisson,
                   'poisson_max_axis_2':self.max_2nd_axis_Poisson,
                   'poisson_anisotropy':self.anis_Poisson,
                   
                   'pugh_ratio_voigt':self.voigtPugh,
                   'pugh_ratio_reuss':self.reussPugh,
                   'pugh_ratio_hill':self.hillPugh,
                   'pugh_ratio_max':self.max_Pugh,
                   'pugh_ratio_min':self.min_Pugh,
                   'pugh_ratio_min_axis':self.min_axis_Pugh,
                   'pugh_ratio_max_axis':self.max_axis_Pugh,
                   'pugh_ratio_min_axis_2':self.mix_2nd_axis_Pugh,
                   'pugh_ratio_max_axis_2':self.max_2nd_axis_Pugh,
                   'pugh_ratio_anisotropy':self.anis_Pugh,
                   }
        if self.density != None: 
            tmp_dict.update({  'compressionSpeed_voigt':self.voigtCompressionSpeed, 
                               'compressionSpeed_reuss':self.reussCompressionSpeed, 
                               'compressionSpeed_hill':self.hillCompressionSpeed,
                               'compressionSpeed_max':self.max_CompressionSpeed,
                               'compressionSpeed_min':self.min_CompressionSpeed,
                               'compressionSpeed_min_axis':self.min_axis_CompressionSpeed,
                               'compressionSpeed_max_axis':self.max_axis_CompressionSpeed,
                               'compressionSpeed_anisotropy':self.anis_CompressionSpeed,

                               
                                'shearSpeed_voigt':self.voigtShearSpeed, 
                                'shearSpeed_reuss':self.reussShearSpeed, 
                                'shearSpeed_hill':self.hillShearSpeed,
                                'shearSpeed_max':self.max_ShearSpeed,
                                'shearSpeed_min':self.min_ShearSpeed,
                                'shearSpeed_min_axis':self.min_axis_ShearSpeed,
                                'shearSpeed_max_axis':self.max_axis_ShearSpeed,
                                'shearSpeed_anisotropy':self.anis_ShearSpeed,

                               
                                'ratioCompressionShearSpeed_voigt':self.voigtRatioCompressionShearSpeed, 
                                'ratioCompressionShearSpeed_reuss':self.reussRatioCompressionShearSpeed, 
                                'ratioCompressionShearSpeed_hill':self.hillRatioCompressionShearSpeed,
                                'ratioCompressionShearSpeed_max':self.max_RatioCompressionShearSpeed,
                                'ratioCompressionShearSpeed_min':self.min_RatioCompressionShearSpeed,
                                'ratioCompressionShearSpeed_min_axis':self.min_axis_RatioCompressionShearSpeed,
                                'ratioCompressionShearSpeed_max_axis':self.max_axis_RatioCompressionShearSpeed,
                                'ratioCompressionShearSpeed_anisotropy':self.anis_RatioCompressionShearSpeed,

                               
                                'debyeSpeed_voigt':self.voigtDebyeSpeed, 
                                'debyeSpeed_reuss':self.reussDebyeSpeed, 
                                'debyeSpeed_hill':self.hillDebyeSpeed,
                                'debyeSpeed_max':self.max_DebyeSpeed,
                                'debyeSpeed_min':self.min_DebyeSpeed,
                                'debyeSpeed_min_axis':self.min_axis_DebyeSpeed,
                                'debyeSpeed_max_axis':self.max_axis_DebyeSpeed,
                                'debyeSpeed_anisotropy':self.anis_DebyeSpeed,
                               })
           
        return tmp_dict
        
  
    ##############################################################################
    # Plotting functions
    #############################################################################
    def plot_3D(self, elastic_calc="", npoints=100,show = True):
        
        """
        3D plotting function
        
        Parameters
        ----------
        elastic_calc : str
            This is a string indicating which elastic property to plot
        npoints : int
            The number of coordinate points. The default is 100.
        show : boolean
            This flag indicates if you want to show display the plot. Deafult show = True.

        Returns
        -------
        Returns : A PyvVista StructuredGrid object

        """
        import pyvista as pv
        

        #plotter = pv.Plotter()
        plotter = pv.Plotter(off_screen=True)

        
        
        x = None
        y = None
        z = None
        r = None
        meshes = []
        if elastic_calc == "POISSON":
            func = self.POISSON3D(npoints=100)
            colors = ["red", "green", "blue"]
            for ix, icolor in zip(range(len(func)), colors):
                x = np.array(func[ix][0])
                y = np.array(func[ix][1])
                z = np.array(func[ix][2])
                r = np.array(func[ix][3])
                if np.all((func[ix][0] == 0)):
                    continue
                else:
                    grid = pv.StructuredGrid(x, y, z)
                    meshes.append(grid)
                    if ix == 2:
                        plotter.add_mesh(grid, opacity=0.25, color=icolor)
                    else:
                        plotter.add_mesh(grid, opacity=0.50, color=icolor)

        elif elastic_calc == "SHEAR":
            func = self.SHEAR3D(npoints=100)
            colors = ["green", "blue"]
            for ix, icolor in zip(range(len(func)), colors):
                x = np.array(func[ix][0])
                y = np.array(func[ix][1])
                z = np.array(func[ix][2])
                r = np.array(func[ix][3])
                if np.all((func[ix][0] == 0)):
                    continue
                else:
                    grid = pv.StructuredGrid(x, y, z)
                    meshes.append(grid)
                    if ix == 2:
                        plotter.add_mesh(grid, opacity=0.25, color=icolor)
                    else:
                        plotter.add_mesh(grid, opacity=0.50, color=icolor)

        elif elastic_calc == "LC":
            func = self.LC3D(npoints=100)
            colors = ["green", "red"]
            for ix, icolor in zip(range(len(func)), colors):
                x = np.array(func[ix][0])
                y = np.array(func[ix][1])
                z = np.array(func[ix][2])
                r = np.array(func[ix][3])
                if np.all((func[ix][0] == 0)):
                    continue
                else:
                    grid = pv.StructuredGrid(x, y, z)
                    meshes.append(grid)
                    if ix == 2:
                        plotter.add_mesh(grid, opacity=0.25, color=icolor)
                    else:
                        plotter.add_mesh(grid, opacity=0.50, color=icolor)

        elif elastic_calc == "YOUNG":
            func = self.YOUNG3D(npoints=100)
            colors = ["green"]
            for ix, icolor in zip(range(len(func)), colors):

                x = np.array(func[0])
                y = np.array(func[1])
                z = np.array(func[2])
                r = np.array(func[3])
                if np.all((func[0] == 0)):
                    continue
                else:
                    grid = pv.StructuredGrid(x, y, z)
                    meshes.append(grid)
                    if ix == 2:
                        plotter.add_mesh(grid, opacity=0.25, color=icolor)
                    else:
                        plotter.add_mesh(grid, opacity=0.50, color=icolor)
                        
        elif elastic_calc == "PUGH_RATIO":
            func = self.PUGH_RATIO3D(npoints=100)
            colors = ["green", "blue"]
            for ix, icolor in zip(range(len(func)), colors):
                x = np.array(func[ix][0])
                y = np.array(func[ix][1])
                z = np.array(func[ix][2])
                r = np.array(func[ix][3])
                if np.all((func[ix][0] == 0)):
                    continue
                else:
                    grid = pv.StructuredGrid(x, y, z)
                    meshes.append(grid)
                    if ix == 2:
                        plotter.add_mesh(grid, opacity=0.25, color=icolor)
                    else:
                        plotter.add_mesh(grid, opacity=0.50, color=icolor)
        
        elif elastic_calc == "BULK":
            func = self.BULK3D(npoints=100)
            colors = ["green", "blue"]
            for ix, icolor in zip(range(len(func)), colors):
                x = np.array(func[ix][0])
                y = np.array(func[ix][1])
                z = np.array(func[ix][2])
                r = np.array(func[ix][3])
                if np.all((func[ix][0] == 0)):
                    continue
                else:
                    grid = pv.StructuredGrid(x, y, z)
                    meshes.append(grid)
                    if ix == 2:
                        plotter.add_mesh(grid, opacity=0.25, color=icolor)
                    else:
                        plotter.add_mesh(grid, opacity=0.50, color=icolor)
                        
        if self.density != None:    
            if elastic_calc == "COMPRESSION_SPEED":
                func = self.COMPRESSION_SPEED3D(npoints=100)
                colors = ["green", "blue"]
                for ix, icolor in zip(range(len(func)), colors):
                    x = np.array(func[ix][0])
                    y = np.array(func[ix][1])
                    z = np.array(func[ix][2])
                    r = np.array(func[ix][3])
                    if np.all((func[ix][0] == 0)):
                        continue
                    else:
                        grid = pv.StructuredGrid(x, y, z)
                        meshes.append(grid)
                        if ix == 2:
                            plotter.add_mesh(grid, opacity=0.25, color=icolor)
                        else:
                            plotter.add_mesh(grid, opacity=0.50, color=icolor)
            
            elif elastic_calc == "SHEAR_SPEED":
                func = self.SHEAR_SPEED3D(npoints=100)
                colors = ["green", "blue"]
                for ix, icolor in zip(range(len(func)), colors):
                    x = np.array(func[ix][0])
                    y = np.array(func[ix][1])
                    z = np.array(func[ix][2])
                    r = np.array(func[ix][3])
                    if np.all((func[ix][0] == 0)):
                        continue
                    else:
                        grid = pv.StructuredGrid(x, y, z)
                        meshes.append(grid)
                        if ix == 2:
                            plotter.add_mesh(grid, opacity=0.25, color=icolor)
                        else:
                            plotter.add_mesh(grid, opacity=0.50, color=icolor)
                            
            elif elastic_calc == "RATIO_COMPRESSIONAL_SHEAR":
                func = self.RATIO_COMPRESSIONAL_SHEAR3D(npoints=100)
                colors = ["green", "blue"]
                for ix, icolor in zip(range(len(func)), colors):
                    x = np.array(func[ix][0])
                    y = np.array(func[ix][1])
                    z = np.array(func[ix][2])
                    r = np.array(func[ix][3])
                    if np.all((func[ix][0] == 0)):
                        continue
                    else:
                        grid = pv.StructuredGrid(x, y, z)
                        meshes.append(grid)
                        if ix == 2:
                            plotter.add_mesh(grid, opacity=0.25, color=icolor)
                        else:
                            plotter.add_mesh(grid, opacity=0.50, color=icolor)
                            
            elif elastic_calc == "DEBYE_SPEED":
                func = self.DEBYE_SPEED3D(npoints=100)
                colors = ["green", "blue"]
                for ix, icolor in zip(range(len(func)), colors):
                    x = np.array(func[ix][0])
                    y = np.array(func[ix][1])
                    z = np.array(func[ix][2])
                    r = np.array(func[ix][3])
                    if np.all((func[ix][0] == 0)):
                        continue
                    else:
                        grid = pv.StructuredGrid(x, y, z)
                        meshes.append(grid)
                        if ix == 2:
                            plotter.add_mesh(grid, opacity=0.25, color=icolor)
                        else:
                            plotter.add_mesh(grid, opacity=0.50, color=icolor)
                            
        if elastic_calc in ['DEBYE_SPEED', 'SHEAR_SPEED', 'COMPRESSIONAL_SPEED', 'RATIO_COMPRESSIONAL_SHEAR'] and self.density == None:
            print("You must specify density in kg/m^3 to produce DEBYE_SPEED, SHEAR_SPEED, COMPRESSIONAL_SPEED, and RATIO_COMPRESSIONAL_SHEAR")

        plotter.add_axes()
        plotter.show_grid(color = "black")
        if show:
            plotter.set_background(color="white")
            #plotter.show()
            plotter.show(screenshot=f'{elastic_calc}_Elate_3D.png')
        return meshes
     
    def plot_3D_slice(self, elastic_calc="", npoints=100, normal = (1,0,0), show = True):
        
        """
        3D plotting function
        
        Parameters
        ----------
        elastic_calc : str
            This is a string indicating which elastic property to plot
        npoints : int
            The number of coordinate points. The default is 100.
        show : boolean
            This flag indicates if you want to show display the plot. Deafult show = True.

        Returns
        -------
        Returns : A PyvVista StructuredGrid object

        """
        import pyvista as pv
        # Definition of colors in rgba
        blue = np.array([0, 0, 1, 0.50])
        green = np.array([0, 1, 0, 0.50])
        red = np.array([1, 0, 0, 0.50])
        
       
        #plotter = pv.Plotter()
        plotter = pv.Plotter(off_screen=True)


        plotter.set_background(color="white")
        x = None
        y = None
        z = None
        r = None
        meshes = []
        if elastic_calc == "POISSON":
            func = self.POISSON3D(npoints=100)
            colors = ["red", "green", "blue"]
            for ix, color in zip(range(len(func)), colors):
                x = np.array(func[ix][0])
                y = np.array(func[ix][1])
                z = np.array(func[ix][2])
                r = np.array(func[ix][3])
                if np.all((func[ix][0] == 0)):
                    continue
                else:
                    grid = pv.StructuredGrid(x, y, z)
                    if color == "blue":
                        grid["colors"] = [blue]*len(grid.points)
                    elif color == "green":
                        grid["colors"] = [green]*len(grid.points)
                    elif color == "red":
                        grid["colors"] = [red]*len(grid.points)
                                     
                    meshes.append(grid)

        elif elastic_calc == "SHEAR": 
            func = self.SHEAR3D(npoints=100)
            colors = ["green", "blue"]
            for ix, color in zip(range(len(func)), colors):
                x = np.array(func[ix][0])
                y = np.array(func[ix][1])
                z = np.array(func[ix][2])
                r = np.array(func[ix][3])
                if np.all((func[ix][0] == 0)):
                    continue
                else:
                    grid = pv.StructuredGrid(x, y, z)
                    if color == "blue":
                        grid["colors"] = [blue]*len(grid.points)
                    elif color == "green":
                        grid["colors"] = [green]*len(grid.points)
                    elif color == "red":
                        grid["colors"] = [red]*len(grid.points)
                                     
                    meshes.append(grid)

        elif elastic_calc == "LC": 
            func = self.LC3D(npoints=100)
            colors = ["green", "red"]
            for ix, color in zip(range(len(func)), colors):
                x = np.array(func[ix][0])
                y = np.array(func[ix][1])
                z = np.array(func[ix][2])
                r = np.array(func[ix][3])
                if np.all((func[ix][0] == 0)):
                    continue
                else:
                    grid = pv.StructuredGrid(x, y, z)
                    if color == "blue":
                        grid["colors"] = [blue]*len(grid.points)
                    elif color == "green":
                        grid["colors"] = [green]*len(grid.points)
                    elif color == "red":
                        grid["colors"] = [red]*len(grid.points)
                                     
                    meshes.append(grid)

        elif elastic_calc == "YOUNG":
            func = self.YOUNG3D(npoints=100)
            colors = ["green"]
            for ix, color in zip(range(len(func)), colors):

                x = np.array(func[0])
                y = np.array(func[1])
                z = np.array(func[2])
                r = np.array(func[3])
                if np.all((func[0] == 0)):
                    continue
                else:
                    grid = pv.StructuredGrid(x, y, z)
                    if color == "blue":
                        grid["colors"] = [blue]*len(grid.points)
                    elif color == "green":
                        grid["colors"] = [green]*len(grid.points)
                    elif color == "red":
                        grid["colors"] = [red]*len(grid.points)
                                     
                    meshes.append(grid)  
                        
        elif elastic_calc == "PUGH_RATIO":
            func = self.PUGH_RATIO3D(npoints=100)
            colors = ["green", "blue"]
            for ix, color in zip(range(len(func)), colors):
                x = np.array(func[ix][0])
                y = np.array(func[ix][1])
                z = np.array(func[ix][2])
                r = np.array(func[ix][3])
                if np.all((func[ix][0] == 0)):
                    continue
                else:
                    grid = pv.StructuredGrid(x, y, z)
                    if color == "blue":
                        grid["colors"] = [blue]*len(grid.points)
                    elif color == "green":
                        grid["colors"] = [green]*len(grid.points)
                    elif color == "red":
                        grid["colors"] = [red]*len(grid.points)
                                     
                    meshes.append(grid)
                    
        elif elastic_calc == "BULK": 
            func = self.BULK3D(npoints=100)
            colors = ["green", "blue"]
            for ix, color in zip(range(len(func)), colors):
                x = np.array(func[ix][0])
                y = np.array(func[ix][1])
                z = np.array(func[ix][2])
                r = np.array(func[ix][3])
                if np.all((func[ix][0] == 0)):
                    continue
                else:
                    grid = pv.StructuredGrid(x, y, z)
                    if color == "blue":
                        grid["colors"] = [blue]*len(grid.points)
                    elif color == "green":
                        grid["colors"] = [green]*len(grid.points)
                    elif color == "red":
                        grid["colors"] = [red]*len(grid.points)
                                     
                    meshes.append(grid)

                 
        if self.density != None:    
            if elastic_calc == "COMPRESSION_SPEED":
                func = self.COMPRESSION_SPEED3D(npoints=100)
                colors = ["green", "blue"]
                for ix, color in zip(range(len(func)), colors):
                    x = np.array(func[ix][0])
                    y = np.array(func[ix][1])
                    z = np.array(func[ix][2])
                    r = np.array(func[ix][3])
                    if np.all((func[ix][0] == 0)):
                        continue
                    else:
                        grid = pv.StructuredGrid(x, y, z)
                        if color == "blue":
                            grid["colors"] = [blue]*len(grid.points)
                        elif color == "green":
                            grid["colors"] = [green]*len(grid.points)
                        elif color == "red":
                            grid["colors"] = [red]*len(grid.points)
                                         
                        meshes.append(grid)
            
            elif elastic_calc == "SHEAR_SPEED": 
                func = self.SHEAR_SPEED3D(npoints=100)
                colors = ["green", "blue"]
                for ix, color in zip(range(len(func)), colors):
                    x = np.array(func[ix][0])
                    y = np.array(func[ix][1])
                    z = np.array(func[ix][2])
                    r = np.array(func[ix][3])
                    if np.all((func[ix][0] == 0)):
                        continue
                    else:
                        grid = pv.StructuredGrid(x, y, z)
                        if color == "blue":
                            grid["colors"] = [blue]*len(grid.points)
                        elif color == "green":
                            grid["colors"] = [green]*len(grid.points)
                        elif color == "red":
                            grid["colors"] = [red]*len(grid.points)
                                         
                        meshes.append(grid)
                            
            elif elastic_calc == "RATIO_COMPRESSIONAL_SHEAR":   
                func = self.RATIO_COMPRESSIONAL_SHEAR3D(npoints=100)
                colors = ["green", "blue"]
                for ix, color in zip(range(len(func)), colors):
                    x = np.array(func[ix][0])
                    y = np.array(func[ix][1])
                    z = np.array(func[ix][2])
                    r = np.array(func[ix][3])
                    if np.all((func[ix][0] == 0)):
                        continue
                    else:
                        grid = pv.StructuredGrid(x, y, z)
                        if color == "blue":
                            grid["colors"] = [blue]*len(grid.points)
                        elif color == "green":
                            grid["colors"] = [green]*len(grid.points)
                        elif color == "red":
                            grid["colors"] = [red]*len(grid.points)
                                         
                        meshes.append(grid)
                            
            elif elastic_calc == "DEBYE_SPEED": 
                func = self.DEBYE_SPEED3D(npoints=100)
                colors = ["green", "blue"]
                for ix, color in zip(range(len(func)), colors):
                    x = np.array(func[ix][0])
                    y = np.array(func[ix][1])
                    z = np.array(func[ix][2])
                    r = np.array(func[ix][3])
                    if np.all((func[ix][0] == 0)):
                        continue
                    else:
                        grid = pv.StructuredGrid(x, y, z)
                        if color == "blue":
                            grid["colors"] = [blue]*len(grid.points)
                        elif color == "green":
                            grid["colors"] = [green]*len(grid.points)
                        elif color == "red":
                            grid["colors"] = [red]*len(grid.points)
                                         
                        meshes.append(grid)
                            
        if elastic_calc in ['DEBYE_SPEED', 'SHEAR_SPEED', 'COMPRESSIONAL_SPEED', 'RATIO_COMPRESSIONAL_SHEAR'] and self.density == None:
            print("You must specify density in kg/m^3 to produce DEBYE_SPEED, SHEAR_SPEED, COMPRESSIONAL_SPEED, and RATIO_COMPRESSIONAL_SHEAR")
       
        # plotter.add_mesh(mesh, rgb = True)
        final_mesh = None
        for imesh,mesh in enumerate(meshes):
            if imesh == 0:
                final_mesh = mesh
            else:
                final_mesh = final_mesh + mesh

        plotter.add_mesh_slice(final_mesh, normal = normal, rgb=True)
        plotter.add_axes()
        plotter.show_grid(color = "black")
        if show:
            #plotter.show()
            plotter.show(screenshot=f'{elastic_calc}_Elate_3DSlice.png')
        return meshes
    
    def plot_2D(self, elastic_calc="all", npoints=100, apply_to_plot=None, show= True, ):
        """
        Parameters
        ----------
        elastic_calc : str, optional
            The property to be plotted. The default is "all".
        npoints : TYPE, optional
            The number of coordinate points. The default is 100.
        apply_to_plot : function, optional
            A python function to be applied to each plot. The default is None.
            The following example will create axis that go through origin
            e.g. ::

                def detail(fig, ax):
                fig.set_size_inches((11, 15), forward=True)
                ax.spines["left"].set_position("center")
                ax.spines["bottom"].set_position("center")
                ax.spines["right"].set_color("none")
                ax.spines["top"].set_color("none")
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                rec = patches.Rectangle(
                    (xlim[0], ylim[0]),
                    (xlim[1] - xlim[0]),
                    (ylim[1] - ylim[0]),
                    linewidth=2,
                    edgecolor="grey",
                    facecolor="none",
                )

                ax.add_patch(rec)

                fig.suptitle("")
                title = ax.get_title()
                ax.get_yaxis().set_visible(True)
                ax.set_title(title, y=-0.1)
                ax.tick_params(
                    which="major", axis="y", direction="inout", width=1, length=5, rotation=0
                )

                ax.tick_params(
                    which="major", axis="x", direction="inout", width=1, length=5, rotation=0
                )
        show : boolean
            This flag indicates if you want to show display the plot. Deafult show = True.

        Returns
        -------
        Returns : matplotlib figure object

        """

        fig = plt.figure()
        subTitles = ["XY", "XZ", "YZ"]
        #fig.patch.set_facecolor('lightgray') 

        if elastic_calc == "POISSON":
            func = self.POISSON2D(npoints=npoints)
            colors = ["red", "green", "blue"]
            labels = ["Poisson - Neg", "Poisson - Pos", "Poisson - Max"]
            fig.suptitle("Poisson's ratio")
            for iplane, title in zip(range(len(func)), subTitles):

                ax = fig.add_subplot(1, 3, iplane + 1)
                ax.set_title(title)
                ax.get_yaxis().set_visible(False)
                ax.patch.set_facecolor('white')
                for iplot, color in zip(range(len(func[0])), colors):
                    plt.plot(
                        func[iplane][iplot][0], func[iplane][iplot][1], color=color
                    )

        elif elastic_calc == "SHEAR":
            func = self.SHEAR2D(npoints=npoints)
            colors = ["green", "blue"]
            labels = ["Shear - Max", "Shear -"]
            fig.suptitle("Shear modulus")
            for iplane, title in zip(range(len(func)), subTitles):

                ax = fig.add_subplot(1, 3, iplane + 1)
                ax.set_title(title)
                ax.get_yaxis().set_visible(False)
                ax.patch.set_facecolor('white')
                for iplot, color in zip(range(len(func[0])), colors):
                    plt.plot(
                        func[iplane][iplot][0], func[iplane][iplot][1], color=color
                    )

        elif elastic_calc == "LC":
            func = self.LC2D(npoints=npoints)
            colors = ["green", "red"]
            labels = ["LC-positive", "LC-negative"]
            fig.suptitle("Linear Compression")
            for iplane, title in zip(range(len(func)), subTitles):

                ax = fig.add_subplot(1, 3, iplane + 1)
                ax.set_title(title)
                ax.get_yaxis().set_visible(False)
                ax.patch.set_facecolor('white')
                for iplot, color, label in zip(range(len(func[0])), colors, labels):
                    plt.plot(
                        func[iplane][iplot][0],
                        func[iplane][iplot][1],
                        color=color,
                        label=label,
                    )

        elif elastic_calc == "YOUNG":
            func = self.YOUNG2D(npoints=npoints)
            color = "green"
            label = "YOUNG"
            fig.suptitle("Young's Modulus")
            for iplane, title in zip(range(len(func)), subTitles):
                ax = fig.add_subplot(1, 3, iplane + 1)
                ax.set_title(title)
                ax.get_yaxis().set_visible(False)
                ax.patch.set_facecolor('white')
                plt.plot(func[iplane][0], func[iplane][1], color=color, label=label)
                
        elif elastic_calc == "PUGH_RATIO":
            func = self.PUGH_RATIO2D(npoints=npoints)
            colors = ["green", "blue"]
            labels = ["Pugh's Ratio - Max", "Pugh's Ratio "]
            fig.suptitle("Pugh's Ratio")
            for iplane, title in zip(range(len(func)), subTitles):

                ax = fig.add_subplot(1, 3, iplane + 1)
                ax.set_title(title)
                ax.get_yaxis().set_visible(False)
                ax.patch.set_facecolor('white')
                for iplot, color in zip(range(len(func[0])), colors):
                    plt.plot(
                        func[iplane][iplot][0], func[iplane][iplot][1], color=color
                    )
                    
        elif elastic_calc == "BULK":
            func = self.BULK2D(npoints=npoints)
            colors = ["green", "blue"]
            labels = ["Bulk Modulus - Max", "Bulk Modulus "]
            fig.suptitle("Bulk Modulus")
            for iplane, title in zip(range(len(func)), subTitles):

                ax = fig.add_subplot(1, 3, iplane + 1)
                ax.set_title(title)
                ax.get_yaxis().set_visible(False)
                ax.patch.set_facecolor('white')
                for iplot, color in zip(range(len(func[0])), colors):
                    plt.plot(
                        func[iplane][iplot][0], func[iplane][iplot][1], color=color
                    )
                    
        elif elastic_calc == "all":
            func = self.YOUNG2D(npoints=npoints)
            color = "green"
            label = "YOUNG"
            # fig.suptitle("Young's Modulus")
            subTitles = [
                "Young's Modulus in (XY) plane",
                "Young's Modulus in (XZ) plane",
                "Young's Modulus in (YZ) plane",
            ]
            for iplane, title in zip(range(len(func)), subTitles):
                ax = fig.add_subplot(4, 3, iplane + 1)
                ax.set_title(title)
                ax.get_yaxis().set_visible(False)
                ax.patch.set_facecolor('white')
                plt.plot(func[iplane][0], func[iplane][1], color=color, label=label)
                if apply_to_plot is not None:
                    apply_to_plot(fig, ax)
            func = self.LC2D(npoints=npoints)
            colors = ["green", "red"]
            labels = ["LC-positive", "LC-negative"]
            # fig.suptitle("Linear Compression")
            subTitles = [
                "Bulk Modulus in (XY) plane",
                "Bulk Modulus in (XZ) plane",
                "Bulk Modulus in (YZ) plane",
            ]
            for iplane, title in zip(range(len(func)), subTitles):

                ax = fig.add_subplot(4, 3, iplane + 4)
                ax.set_title(title)
                ax.get_yaxis().set_visible(False)
                ax.patch.set_facecolor('white')
                for iplot, color, label in zip(range(len(func[0])), colors, labels):
                    plt.plot(
                        func[iplane][iplot][0],
                        func[iplane][iplot][1],
                        color=color,
                        label=label,
                    )
                if apply_to_plot is not None:
                    apply_to_plot(fig, ax)
            func = self.SHEAR2D(npoints=npoints)
            colors = ["green", "blue"]
            labels = ["Shear - Max", "Shear -"]
            fig.suptitle("Shear modulus")
            subTitles = [
                "Shear Modulus in (XY) plane",
                "Shear Modulus in (XZ) plane",
                "Shear Modulus in (YZ) plane",
            ]
            for iplane, title in zip(range(len(func)), subTitles):

                ax = fig.add_subplot(4, 3, iplane + 7)
                ax.set_title(title)
                ax.get_yaxis().set_visible(False)
                ax.patch.set_facecolor('white')
                for iplot, color in zip(range(len(func[0])), colors):
                    plt.plot(
                        func[iplane][iplot][0], func[iplane][iplot][1], color=color
                    )
                if apply_to_plot is not None:
                    apply_to_plot(fig, ax)
            func = self.POISSON2D(npoints=npoints)
            colors = ["red", "green", "blue"]
            labels = ["Poisson - Neg", "Poisson - Pos", "Poisson - Max"]
            fig.suptitle("Poisson's ratio")
            subTitles = [
                "Poisson's Ratio in (XY) plane",
                "Poisson's Ratio in (XZ) plane",
                "Poisson's Ratio in (YZ) plane",
            ]
            for iplane, title in zip(range(len(func)), subTitles):

                ax = fig.add_subplot(4, 3, iplane + 10)
                ax.set_title(title)
                ax.get_yaxis().set_visible(False)
                ax.patch.set_facecolor('white')
                for iplot, color in zip(range(len(func[0])), colors):
                    plt.plot(
                        func[iplane][iplot][0], func[iplane][iplot][1], color=color
                    )
                if apply_to_plot is not None:
                    apply_to_plot(fig, ax)
                    
        if self.density != None :
            if elastic_calc == "COMPRESSION_SPEED":
                func = self.COMPRESSION_SPEED2D(npoints=npoints)
                colors = ["green", "blue"]
                labels = ["Compression speed - Max", "Compression speed -"]
                fig.suptitle("Compression speed")
              
                for iplane, title in zip(range(len(func)), subTitles):
    
                    ax = fig.add_subplot(1, 3, iplane + 1)
                    ax.set_title(title)
                    ax.get_yaxis().set_visible(False)
                    ax.patch.set_facecolor('white')
                    for iplot, color in zip(range(len(func[0])), colors):
                        plt.plot(
                            func[iplane][iplot][0], func[iplane][iplot][1], color=color
                        )
            if elastic_calc == "SHEAR_SPEED":
                func = self.SHEAR_SPEED2D(npoints=npoints)
                colors = ["green", "blue"]
                labels = ["Shear speed - Max", "Shear speed -"]
                fig.suptitle("Shear speed")
                for iplane, title in zip(range(len(func)), subTitles):
    
                    ax = fig.add_subplot(1, 3, iplane + 1)
                    ax.set_title(title)
                    ax.get_yaxis().set_visible(False)
                    for iplot, color in zip(range(len(func[0])), colors):
                        plt.plot(
                            func[iplane][iplot][0], func[iplane][iplot][1], color=color
                        )
            if elastic_calc == "RATIO_COMPRESSIONAL_SHEAR":
                func = self.RATIO_COMPRESSIONAL_SHEAR2D(npoints=npoints)
                colors = ["green", "blue"]
                labels = ["Ratio compression/shear - Max", "Ratio compression/shear -"]
                fig.suptitle("Ratio compression/shear")
                for iplane, title in zip(range(len(func)), subTitles):
    
                    ax = fig.add_subplot(1, 3, iplane + 1)
                    ax.set_title(title)
                    ax.get_yaxis().set_visible(False)
                    ax.patch.set_facecolor('white')
                    for iplot, color in zip(range(len(func[0])), colors):
                        plt.plot(
                            func[iplane][iplot][0], func[iplane][iplot][1], color=color
                        )
            if elastic_calc == "DEBYE_SPEED":
                func = self.DEBYE_SPEED2D(npoints=npoints)
                colors = ["green", "blue"]
                labels = ["Debye speed - Max", "Debye speed -"]
                fig.suptitle("Debye speed")
                for iplane, title in zip(range(len(func)), subTitles):
    
                    ax = fig.add_subplot(1, 3, iplane + 1)
                    ax.set_title(title)
                    ax.get_yaxis().set_visible(False)
                    ax.patch.set_facecolor('white')
                    for iplot, color in zip(range(len(func[0])), colors):
                        plt.plot(
                            func[iplane][iplot][0], func[iplane][iplot][1], color=color
                        )
        if elastic_calc in ['DEBYE_SPEED', 'SHEAR_SPEED', 'COMPRESSIONAL_SPEED', 'RATIO_COMPRESSIONAL_SHEAR'] and self.density == None:
            print("You must specify density in kg/m^3 to produce DEBYE_SPEED, SHEAR_SPEED, COMPRESSIONAL_SPEED, and RATIO_COMPRESSIONAL_SHEAR")

            
            
            
        if show:
            #plt.show()
            plt.savefig(f'{elastic_calc}_Elate_2D.png', dpi=300, bbox_inches='tight')
            
        return fig



#    def update_widths(self, title, unit, min_val, max_val, anisotropy):
#        self.title_width = max(self.title_width, len(title))
#        self.unit_width = max(self.unit_width, len(unit))
#        self.value_width = max(self.value_width, len(f"{min_val:.3f}"), len(f"{max_val:.3f}"))
#        self.anisotropy_width = max(self.anisotropy_width, len(anisotropy))

    def update_widths(self, titles, unit, min_val, max_val, anisotropy):
        max_title_length = max([len(title) for title in titles])
        self.title_width = max(self.title_width, max_title_length)
        self.unit_width = max(self.unit_width, len(unit))
        self.value_width = max(self.value_width, len(f"{min_val:.3f}"), len(f"{max_val:.3f}"))
        self.anisotropy_width = max(self.anisotropy_width, len(anisotropy))



    def print_header(self):
        print(f"{' ':<{self.title_width}} {' ':<{self.unit_width}} {'Min':>{self.value_width}} {'Max':>{self.value_width}}   {'Anisotropy':>{self.anisotropy_width}}")
        print("-" * (self.title_width + self.unit_width + 2*self.value_width + self.anisotropy_width + 8))  # 8 for spaces and || symbols


    def format_values(self, title, unit, min_val, max_val, anisotropy, min_axis, max_axis, second_min_axis=None, second_max_axis=None):
        #self.update_widths(title, unit, min_val, max_val, anisotropy)
        print(f"{title:<{self.title_width}} {unit:<{self.unit_width}} {min_val:>{self.value_width}.3f} {max_val:>{self.value_width}.3f} {anisotropy:>{self.anisotropy_width}}")
        print(f"         Min Axis:    {tuple(min_axis)}")
        print(f"         Max Axis:    {tuple(max_axis)}")
        if second_min_axis and second_max_axis:
            print(f"         Second Min Axis: {tuple(second_min_axis)}")
            print(f"         Second Max Axis: {tuple(second_max_axis)}")










    def print_properties_2D(self):
        """
        Print summary method
        
        Parameters
        ----------
        None
        
        Returns
        -------
        Returns : None

        """
        print("\n")
        framed_message = [
            "***********************************************************",
            "* Summary of averaged properties based on the Voigt,      *",
            "* Reuss, Hill (VRH) approaches. Note for 2D material,     *",
            "* the obtained elastic parameters with the VRH approaches *",
            "* are not directly applicable. Data presented below used  *",
            "* explicit 2D approaches and might be different.          *",
            "***********************************************************"
        ]

        for line in framed_message:
            print(line)

        # Predefined headers and labels
        headers = ["Voigt", "Reuss", "Hill"]
        content_labels = [
            "Bulk modulus       (N/m)",
            "Shear modulus      (N/m)",
            "Young's modulus    (N/m)",
            "Poisson's ratio",
            "Pugh's Ratio"
        ]

        # Data based on calculations
        content_values = [
            [self.elas.averages()[0][0], self.elas.averages()[1][0], self.elas.averages()[2][0]],
            [self.elas.averages()[0][2], self.elas.averages()[1][2], self.elas.averages()[2][2]],
            [self.elas.averages()[0][1], self.elas.averages()[1][1], self.elas.averages()[2][1]],
            [self.elas.averages()[0][3], self.elas.averages()[1][3], self.elas.averages()[2][3]],
            [self.voigtPugh, self.reussPugh, self.hillPugh]
        ]

        # Determine the maximum width for consistent formatting
        max_label_width = max([len(label) for label in content_labels])
        max_width = max(max_label_width, len(headers[0]), len(headers[1]), len(headers[2])) + 2
        col_width = 12 #max_width + 9  # for the numbers, based on format 9.3f

        # Print header
        header_str = "{:<{width}}".format("", width=max_width)
        for h in headers:
            header_str += "{:<{width}}".format(h, width=col_width)
        print(header_str)
        print('-' * len(header_str))

        # Print content values under each header
        for label, values in zip(content_labels, content_values):
            row_str = "{:<{width}}".format(label, width=max_width)
            for val in values:
                row_str += "{:<{width}.3f}".format(val, width=col_width)
            print(row_str)

        # Additional content if density is not None
        if self.density is not None:
            additional_labels = [
                "Compression Speed  (km/s)",
                "Shear Speed        (km/s)",
                "Ratio Vc/Vs",
                "Debye Speed        (km/s)"
            ]
            additional_values = [
                [self.voigtCompressionSpeed/math.sqrt(1E+9), self.reussCompressionSpeed/math.sqrt(1E+9), self.hillCompressionSpeed/math.sqrt(1E+9)],
                [self.voigtShearSpeed/math.sqrt(1E+9), self.reussShearSpeed/math.sqrt(1E+9), self.hillShearSpeed/math.sqrt(1E+9)],
                [self.voigtRatioCompressionShearSpeed, self.reussRatioCompressionShearSpeed, self.hillRatioCompressionShearSpeed],
                [self.voigtDebyeSpeed/math.sqrt(1E+9), self.reussDebyeSpeed/math.sqrt(1E+9), self.hillDebyeSpeed/math.sqrt(1E+9)]
            ]

            for label, values in zip(additional_labels, additional_values):
                row_str = "{:<{width}}".format(label, width=max_width)
                for val in values:
                    row_str += "{:<{width}.3f}".format(val, width=col_width)
                print(row_str)
            print('-' * len(header_str))



        print("\n") 
        print("Eigenvalues of compliance matrix")
        print('*' * (len(header_str) - 20))
        print("\n  lamda_1  lamda_2  lamda_3  lamda_4  lamda_5  lamda_6")
        print('-' * len(header_str))
        eigenval = sorted(np.linalg.eig(self.elas.CVoigt)[0])
        print("%9.3f %9.3f %9.3f %9.3f %9.3f %9.3f" % tuple(eigenval))
        print('-' * len(header_str))
        if eigenval[0] <= 0:
            print(
                '<Class="error">Eigenvalue matrix is not definite positive,, crystal is mechanically unstable<br/>'
            )
        print("\n")
        print("Variations of the elastic moduli")
        print('*' * (len(header_str) - 20))

        minE = minimize(self.elas.Young, 2)
        maxE = maximize(self.elas.Young, 2)
        minLC = minimize(self.elas.LC, 2)
        maxLC = maximize(self.elas.LC, 2)
        minG = minimize(self.elas.shear, 3)
        maxG = maximize(self.elas.shear, 3)
        minK = minimize(self.elas.Bulk, 3)
        maxK = maximize(self.elas.Bulk, 3)
        minNu = minimize(self.elas.Poisson, 3)
        maxNu = maximize(self.elas.Poisson, 3)
        minPugh = minimize(self.elas.Pugh_ratio, 3)
        maxPugh = maximize(self.elas.Pugh_ratio, 3)

        anisE = "%8.3f" % (maxE[1] / minE[1])
        anisLC = ("%8.3f" % (maxLC[1] / minLC[1])) if minLC[1] > 0 else "&infin;"
        anisG = "%8.3f" % (maxG[1] / minG[1])
        anisK = "%8.3f" % (maxK[1] / minK[1])
        anisNu = (
            ("%8.3f" % (maxNu[1] / minNu[1])) if minNu[1] * maxNu[1] > 0 else "&infin;"
        )
        anisPugh = "%8.3f" % (maxPugh[1] / minPugh[1])


        minEaxis = list(np.around(np.array(dirVec(*minE[0])), 3))
        maxEaxis = list(np.around(np.array(dirVec(*maxE[0])), 3))
        minLCaxis = list(np.around(np.array(dirVec(*minLC[0])), 3))
        maxLCaxis = list(np.around(np.array(dirVec(*maxLC[0])), 3))

        minGaxis = list(np.around(np.array(dirVec1(*minG[0])), 3))
        maxGaxis = list(np.around(np.array(dirVec1(*maxG[0])), 3))
        minG2ndaxis = list(np.around(np.array(dirVec2(*minG[0])), 3))
        maxG2ndaxis = list(np.around(np.array(dirVec2(*maxG[0])), 3))
        
        minKaxis = list(np.around(np.array(dirVec1(*minK[0])), 3))
        maxKaxis = list(np.around(np.array(dirVec1(*maxK[0])), 3))
        minK2ndaxis = list(np.around(np.array(dirVec2(*minK[0])), 3))
        maxK2ndaxis = list(np.around(np.array(dirVec2(*maxK[0])), 3))

        minNUaxis = list(np.around(np.array(dirVec1(*minNu[0])), 3))
        maxNUaxis = list(np.around(np.array(dirVec1(*maxNu[0])), 3))
        minNU2ndaxis = list(np.around(np.array(dirVec2(*minNu[0])), 3))
        maxNU2ndaxis = list(np.around(np.array(dirVec2(*maxNu[0])), 3))
        
        minPughaxis = list(np.around(np.array(dirVec1(*minPugh[0])), 3))
        maxPughaxis = list(np.around(np.array(dirVec1(*maxPugh[0])), 3))
        minPugh2ndaxis = list(np.around(np.array(dirVec2(*minPugh[0])), 3))
        maxPugh2ndaxis = list(np.around(np.array(dirVec2(*maxPugh[0])), 3))


        all_titles = ["Linear Compression (10^12 m/N)"]
        self.update_widths(all_titles, 'N/m', 0, 0, 'anisE')

        self.print_header()
        self.format_values("Young's Modulus (N/m)", "", minE[1], maxE[1], anisE, minEaxis, maxEaxis)
        self.format_values("Linear Compression (10^12 m/N)", " ", minLC[1], maxLC[1], anisLC, minLCaxis, maxLCaxis)
        #self.format_values("Stifness Constant (N/m)", "", minK[1], maxK[1], anisK, minKaxis, maxKaxis, minK2ndaxis, maxK2ndaxis)
        self.format_values("Shear Modulus (N/m)", "", minG[1], maxG[1], anisG, minGaxis, maxGaxis, minG2ndaxis, maxG2ndaxis)
        self.format_values("Poisson's Ratio", "", minNu[1], maxNu[1], anisNu, minNUaxis, maxNUaxis, minNU2ndaxis, maxNU2ndaxis)
        self.format_values("Pugh's Ratio", "", minPugh[1], maxPugh[1], anisPugh, minPughaxis, maxPughaxis, minPugh2ndaxis, maxPugh2ndaxis)
        
        if self.density != None:
            
            minVc = minimize(self.elas.Compression_Speed, 3)
            maxVc = maximize(self.elas.Compression_Speed, 3)

            minVs = minimize(self.elas.Shear_Speed, 3)
            maxVs = maximize(self.elas.Shear_Speed, 3)
            
            minRatio = minimize(self.elas.Ratio_Compression_Shear, 3)
            maxRatio = maximize(self.elas.Ratio_Compression_Shear , 3)
            
            minDebyeSpeed = minimize(self.elas.Debye_Speed, 3)
            maxDebyeSpeed = maximize(self.elas.Debye_Speed, 3) 
            
            anisVc = "%8.3f" % (maxVc[1] / minVc[1])
            anisVs = "%8.3f" % (maxVs[1] / minVs[1])
            anisRatio = "%8.3f" % (maxRatio[1] / minRatio[1])
            anisDebyeSpeed = "%8.3f" % (maxDebyeSpeed[1] / minDebyeSpeed[1])
            
            
            minVcaxis = list(np.around(np.array(dirVec1(*minVc[0])), 3))
            maxVcaxis = list(np.around(np.array(dirVec1(*maxVc[0])), 3))
            minVc2ndaxis = list(np.around(np.array(dirVec2(*minVc[0])), 3))
            maxVc2ndaxis = list(np.around(np.array(dirVec2(*maxVc[0])), 3))
            
            minVsaxis = list(np.around(np.array(dirVec1(*minVs[0])), 3))
            maxVsaxis = list(np.around(np.array(dirVec1(*maxVs[0])), 3))
            minVs2ndaxis = list(np.around(np.array(dirVec2(*minVs[0])), 3))
            maxVs2ndaxis = list(np.around(np.array(dirVec2(*maxVs[0])), 3))
            
            minRatioaxis = list(np.around(np.array(dirVec1(*minRatio[0])), 3))
            maxRatioaxis = list(np.around(np.array(dirVec1(*maxRatio[0])), 3))
            minRatio2ndaxis = list(np.around(np.array(dirVec2(*minRatio[0])), 3))
            maxRatio2ndaxis = list(np.around(np.array(dirVec2(*maxRatio[0])), 3))
    
            minDebyeSpeedaxis = list(np.around(np.array(dirVec1(*minDebyeSpeed[0])), 3))
            maxDebyeSpeedaxis = list(np.around(np.array(dirVec1(*maxDebyeSpeed[0])), 3))
            minDebyeSpeed2ndaxis = list(np.around(np.array(dirVec2(*minDebyeSpeed[0])), 3))
            maxDebyeSpeed2ndaxis = list(np.around(np.array(dirVec2(*maxDebyeSpeed[0])), 3))
            
            self.format_values("Compressional Speed (Km/s)", "", minVc[1]/1000/math.sqrt(1E+9), maxVc[1]/1000/math.sqrt(1E+9), anisVc, minVcaxis, maxVcaxis, minVc2ndaxis, maxVc2ndaxis)
            self.format_values("Shear Speed (Km/s)", "", minVs[1]/1000/math.sqrt(1E+9), maxVs[1]/1000/math.sqrt(1E+9), anisVs, minVsaxis, maxVsaxis, minVc2ndaxis, maxVc2ndaxis)
            self.format_values("Compressional/Shear Speed", "", minRatio[1], maxRatio[1], anisRatio, minRatioaxis, maxRatioaxis, minRatio2ndaxis, maxRatio2ndaxis)
            self.format_values("Debye Speed (Km/s)", "", minDebyeSpeed[1]/1000/math.sqrt(1E+9), maxDebyeSpeed[1]/1000/math.sqrt(1E+9),anisDebyeSpeed, minDebyeSpeedaxis, maxDebyeSpeedaxis, minDebyeSpeed2ndaxis, maxDebyeSpeed2ndaxis) 




    def print_properties(self):
        """
        Print summary method
        
        Parameters
        ----------
        None
        
        Returns
        -------
        Returns : None

        """
        print("\n")
        framed_message = [
            "*********************************************************",
            "* Summary of averaged properties for 3D materials       *",
            "*********************************************************"
        ]

        for line in framed_message:
            print(line)

        # Predefined headers and labels
        headers = ["Voigt", "Reuss", "Hill"]
        content_labels = [
            "Bulk modulus       (GPa)",
            "Shear modulus      (GPa)",
            "Young's modulus    (GPa)",
            "Poisson's ratio",
            "Pugh's Ratio"
        ]

        # Data based on your calculations
        content_values = [
            [self.elas.averages()[0][0], self.elas.averages()[1][0], self.elas.averages()[2][0]],
            [self.elas.averages()[0][2], self.elas.averages()[1][2], self.elas.averages()[2][2]],
            [self.elas.averages()[0][1], self.elas.averages()[1][1], self.elas.averages()[2][1]],
            [self.elas.averages()[0][3], self.elas.averages()[1][3], self.elas.averages()[2][3]],
            [self.voigtPugh, self.reussPugh, self.hillPugh]
        ]

        # Determine the maximum width for consistent formatting
        max_label_width = max([len(label) for label in content_labels])
        max_width = max(max_label_width, len(headers[0]), len(headers[1]), len(headers[2])) + 2
        col_width = 12 #max_width + 9  # for the numbers, based on format 9.3f

        # Print header
        header_str = "{:<{width}}".format("", width=max_width)
        for h in headers:
            header_str += "{:<{width}}".format(h, width=col_width)
        print(header_str)
        print('-' * len(header_str))

        # Print content values under each header
        for label, values in zip(content_labels, content_values):
            row_str = "{:<{width}}".format(label, width=max_width)
            for val in values:
                row_str += "{:<{width}.3f}".format(val, width=col_width)
            print(row_str)

        # Additional content if density is not None
        if self.density is not None:
            additional_labels = [
                "Compression Speed  (km/s)",
                "Shear Speed        (km/s)",
                "Ratio Vc/Vs",
                "Debye Speed        (km/s)"
            ]
            additional_values = [
                [self.voigtCompressionSpeed, self.reussCompressionSpeed, self.hillCompressionSpeed],
                [self.voigtShearSpeed, self.reussShearSpeed, self.hillShearSpeed],
                [self.voigtRatioCompressionShearSpeed, self.reussRatioCompressionShearSpeed, self.hillRatioCompressionShearSpeed],
                [self.voigtDebyeSpeed, self.reussDebyeSpeed, self.hillDebyeSpeed]
            ]

            for label, values in zip(additional_labels, additional_values):
                row_str = "{:<{width}}".format(label, width=max_width)
                for val in values:
                    row_str += "{:<{width}.3f}".format(val, width=col_width)
                print(row_str)
            print('-' * len(header_str))



        print("\n") 
        print("Eigenvalues of compliance matrix")
        print('*' * (len(header_str) - 20))
        print("\n  lamda_1  lamda_2  lamda_3  lamda_4  lamda_5  lamda_6")
        print('-' * len(header_str))
        eigenval = sorted(np.linalg.eig(self.elas.CVoigt)[0])
        print("%9.3f %9.3f %9.3f %9.3f %9.3f %9.3f" % tuple(eigenval))
        print('-' * len(header_str))
        if eigenval[0] <= 0:
            print(
                '<div class="error">Stiffness matrix is not definite positive, crystal is mechanically unstable<br/>'
            )
        print("\n")
        print("Variations of the elastic moduli")
        print('*' * (len(header_str) - 20))

        minE = minimize(self.elas.Young, 2)
        maxE = maximize(self.elas.Young, 2)
        minLC = minimize(self.elas.LC, 2)
        maxLC = maximize(self.elas.LC, 2)
        minG = minimize(self.elas.shear, 3)
        maxG = maximize(self.elas.shear, 3)
        minK = minimize(self.elas.Bulk, 3)
        maxK = maximize(self.elas.Bulk, 3)
        minNu = minimize(self.elas.Poisson, 3)
        maxNu = maximize(self.elas.Poisson, 3)
        minPugh = minimize(self.elas.Pugh_ratio, 3)
        maxPugh = maximize(self.elas.Pugh_ratio, 3)

        anisE = "%8.3f" % (maxE[1] / minE[1])
        anisLC = ("%8.3f" % (maxLC[1] / minLC[1])) if minLC[1] > 0 else "&infin;"
        anisG = "%8.3f" % (maxG[1] / minG[1])
        anisK = "%8.3f" % (maxK[1] / minK[1])
        anisNu = (
            ("%8.3f" % (maxNu[1] / minNu[1])) if minNu[1] * maxNu[1] > 0 else "&infin;"
        )
        anisPugh = "%8.3f" % (maxPugh[1] / minPugh[1])


        minEaxis = list(np.around(np.array(dirVec(*minE[0])), 3))
        maxEaxis = list(np.around(np.array(dirVec(*maxE[0])), 3))
        minLCaxis = list(np.around(np.array(dirVec(*minLC[0])), 3))
        maxLCaxis = list(np.around(np.array(dirVec(*maxLC[0])), 3))

        minGaxis = list(np.around(np.array(dirVec1(*minG[0])), 3))
        maxGaxis = list(np.around(np.array(dirVec1(*maxG[0])), 3))
        minG2ndaxis = list(np.around(np.array(dirVec2(*minG[0])), 3))
        maxG2ndaxis = list(np.around(np.array(dirVec2(*maxG[0])), 3))
        
        minKaxis = list(np.around(np.array(dirVec1(*minK[0])), 3))
        maxKaxis = list(np.around(np.array(dirVec1(*maxK[0])), 3))
        minK2ndaxis = list(np.around(np.array(dirVec2(*minK[0])), 3))
        maxK2ndaxis = list(np.around(np.array(dirVec2(*maxK[0])), 3))

        minNUaxis = list(np.around(np.array(dirVec1(*minNu[0])), 3))
        maxNUaxis = list(np.around(np.array(dirVec1(*maxNu[0])), 3))
        minNU2ndaxis = list(np.around(np.array(dirVec2(*minNu[0])), 3))
        maxNU2ndaxis = list(np.around(np.array(dirVec2(*maxNu[0])), 3))
        
        minPughaxis = list(np.around(np.array(dirVec1(*minPugh[0])), 3))
        maxPughaxis = list(np.around(np.array(dirVec1(*maxPugh[0])), 3))
        minPugh2ndaxis = list(np.around(np.array(dirVec2(*minPugh[0])), 3))
        maxPugh2ndaxis = list(np.around(np.array(dirVec2(*maxPugh[0])), 3))

        all_titles = ["Linear Compression (TPa^-1)"]
        self.update_widths(all_titles, '(TPa^-1)', 0, 0, 'anisE')

        self.print_header()
        self.format_values("Young's Modulus (GPa)", "", minE[1], maxE[1], anisE, minEaxis, maxEaxis)
        self.format_values("Linear Compression (TPa^-1)", " ", minLC[1], maxLC[1], anisLC, minLCaxis, maxLCaxis)
        self.format_values("Stifness Constant (GPa)", "", minK[1], maxK[1], anisK, minKaxis, maxKaxis, minK2ndaxis, maxK2ndaxis)
        self.format_values("Shear Modulus (GPa)", "", minG[1], maxG[1], anisG, minGaxis, maxGaxis, minG2ndaxis, maxG2ndaxis)
        self.format_values("Poisson's Ratio", "", minNu[1], maxNu[1], anisNu, minNUaxis, maxNUaxis, minNU2ndaxis, maxNU2ndaxis)
        self.format_values("Pugh's Ratio", "", minPugh[1], maxPugh[1], anisPugh, minPughaxis, maxPughaxis, minPugh2ndaxis, maxPugh2ndaxis)

        
        if self.density != None:
            
            minVc = minimize(self.elas.Compression_Speed, 3)
            maxVc = maximize(self.elas.Compression_Speed, 3)

            minVs = minimize(self.elas.Shear_Speed, 3)
            maxVs = maximize(self.elas.Shear_Speed, 3)
            
            minRatio = minimize(self.elas.Ratio_Compression_Shear, 3)
            maxRatio = maximize(self.elas.Ratio_Compression_Shear , 3)
            
            minDebyeSpeed = minimize(self.elas.Debye_Speed, 3)
            maxDebyeSpeed = maximize(self.elas.Debye_Speed, 3) 
            
            anisVc = "%8.3f" % (maxVc[1] / minVc[1])
            anisVs = "%8.3f" % (maxVs[1] / minVs[1])
            anisRatio = "%8.3f" % (maxRatio[1] / minRatio[1])
            anisDebyeSpeed = "%8.3f" % (maxDebyeSpeed[1] / minDebyeSpeed[1])
            
            
            minVcaxis = list(np.around(np.array(dirVec1(*minVc[0])), 3))
            maxVcaxis = list(np.around(np.array(dirVec1(*maxVc[0])), 3))
            minVc2ndaxis = list(np.around(np.array(dirVec2(*minVc[0])), 3))
            maxVc2ndaxis = list(np.around(np.array(dirVec2(*maxVc[0])), 3))
            
            minVsaxis = list(np.around(np.array(dirVec1(*minVs[0])), 3))
            maxVsaxis = list(np.around(np.array(dirVec1(*maxVs[0])), 3))
            minVs2ndaxis = list(np.around(np.array(dirVec2(*minVs[0])), 3))
            maxVs2ndaxis = list(np.around(np.array(dirVec2(*maxVs[0])), 3))
            
            minRatioaxis = list(np.around(np.array(dirVec1(*minRatio[0])), 3))
            maxRatioaxis = list(np.around(np.array(dirVec1(*maxRatio[0])), 3))
            minRatio2ndaxis = list(np.around(np.array(dirVec2(*minRatio[0])), 3))
            maxRatio2ndaxis = list(np.around(np.array(dirVec2(*maxRatio[0])), 3))
    
            minDebyeSpeedaxis = list(np.around(np.array(dirVec1(*minDebyeSpeed[0])), 3))
            maxDebyeSpeedaxis = list(np.around(np.array(dirVec1(*maxDebyeSpeed[0])), 3))
            minDebyeSpeed2ndaxis = list(np.around(np.array(dirVec2(*minDebyeSpeed[0])), 3))
            maxDebyeSpeed2ndaxis = list(np.around(np.array(dirVec2(*maxDebyeSpeed[0])), 3))
            
            self.format_values("Compressional Speed (Km/s)", "", minVc[1]/1000, maxVc[1]/1000, anisVc, minVcaxis, maxVcaxis, minVc2ndaxis, maxVc2ndaxis)
            self.format_values("Shear Speed (Km/s)", "", minVs[1]/1000, maxVs[1]/1000, anisVs, minVsaxis, maxVsaxis, minVc2ndaxis, maxVc2ndaxis)
            self.format_values("Compressional/Shear Speed", "", minRatio[1], maxRatio[1], anisRatio, minRatioaxis, maxRatioaxis, minRatio2ndaxis, maxRatio2ndaxis)
            self.format_values("Debye Speed (Km/s)", "", minDebyeSpeed[1]/1000, maxDebyeSpeed[1]/1000,anisDebyeSpeed, minDebyeSpeedaxis, maxDebyeSpeedaxis, minDebyeSpeed2ndaxis, maxDebyeSpeed2ndaxis) 





class Elastic:
    """
    Elastic tensor obejct
    
    Parameters
    ----------
    s : str
        This is a list of list that contains the row of the elastic tensor.
    desnity : float, optional
        The density of the material in kg/m^3

    """

    def __init__(self, s ,density = None):
        """Initialize the elastic tensor from a string"""
        self.density = density
        
        if not s:
            raise ValueError("no matrix was provided")

        # Argument can be a 6-line string, a list of list, or a string representation of the list of list
        try:
            if type(json.loads(s)) == list:
                s = json.loads(s)
        except:
            pass

        if type(s) == str:
            # Remove braces and pipes
            s = s.replace("|", " ").replace("(", " ").replace(")", " ")

            # Remove empty lines
            lines = [line for line in s.split("\n") if line.strip()]
            if len(lines) != 6:
                raise ValueError("should have six rows")

            # Convert to float
            try:
                mat = [list(map(float, line.split())) for line in lines]
            except:
                raise ValueError("not all entries are numbers")
        elif type(s) == list:
            # If we already have a list, simply use it
            mat = s
        else:
            raise ValueError("invalid argument as matrix")

        # Make it into a square matrix
        mat = np.array(mat)
        if mat.shape != (6, 6):
            # Is it upper triangular?
            if list(map(len, mat)) == [6, 5, 4, 3, 2, 1]:
                mat = [[0] * i + mat[i] for i in range(6)]
                mat = np.array(mat)

            # Is it lower triangular?
            if list(map(len, mat)) == [1, 2, 3, 4, 5, 6]:
                mat = [mat[i] + [0] * (5 - i) for i in range(6)]
                mat = np.array(mat)

        if mat.shape != (6, 6):
            raise ValueError("should be a square matrix")

        # Check that is is symmetric, or make it symmetric
        if np.linalg.norm(np.tril(mat, -1)) == 0:
            mat = mat + np.triu(mat, 1).transpose()
        if np.linalg.norm(np.triu(mat, 1)) == 0:
            mat = mat + np.tril(mat, -1).transpose()
        if np.linalg.norm(mat - mat.transpose()) > 1e-3:
            raise ValueError("should be symmetric, or triangular")
        elif np.linalg.norm(mat - mat.transpose()) > 0:
            mat = 0.5 * (mat + mat.transpose())

        # Store it
        self.CVoigt = mat

        # Put it in a more useful representation
        try:
            self.SVoigt = np.linalg.inv(self.CVoigt)
        except:
            raise ValueError("matrix is singular")

        VoigtMat = [[0, 5, 4], [5, 1, 3], [4, 3, 2]]

        def SVoigtCoeff(p, q):
            return 1.0 / ((1 + p // 3) * (1 + q // 3))

        self.Smat = [
            [
                [
                    [
                        SVoigtCoeff(VoigtMat[i][j], VoigtMat[k][l])
                        * self.SVoigt[VoigtMat[i][j]][VoigtMat[k][l]]
                        for i in range(3)
                    ]
                    for j in range(3)
                ]
                for k in range(3)
            ]
            for l in range(3)
        ]
        return


    def isOrthorhombic(self):
        """
        A method to determine if the structure is Orthohombic
        
        Parameters
        ----------
        None
            
        Returns
        -------
        Returns : bool
    
        """

        def iszero(x):
            return abs(x) < 1.0e-3

        return (
            iszero(self.CVoigt[0][3])
            and iszero(self.CVoigt[0][4])
            and iszero(self.CVoigt[0][5])
            and iszero(self.CVoigt[1][3])
            and iszero(self.CVoigt[1][4])
            and iszero(self.CVoigt[1][5])
            and iszero(self.CVoigt[2][3])
            and iszero(self.CVoigt[2][4])
            and iszero(self.CVoigt[2][5])
            and iszero(self.CVoigt[3][4])
            and iszero(self.CVoigt[3][5])
            and iszero(self.CVoigt[4][5])
        )

    def isCubic(self):
        """
        A method to determine if the structure is Cubic
        
        Parameters
        ----------
        None
            
        Returns
        -------
        Returns : bool
    
        """
        def iszero(x):
            return abs(x) < 1.0e-3

        return (
            iszero(self.CVoigt[0][3])
            and iszero(self.CVoigt[0][4])
            and iszero(self.CVoigt[0][5])
            and iszero(self.CVoigt[1][3])
            and iszero(self.CVoigt[1][4])
            and iszero(self.CVoigt[1][5])
            and iszero(self.CVoigt[2][3])
            and iszero(self.CVoigt[2][4])
            and iszero(self.CVoigt[2][5])
            and iszero(self.CVoigt[3][4])
            and iszero(self.CVoigt[3][5])
            and iszero(self.CVoigt[4][5])
            and iszero(self.CVoigt[0][0] - self.CVoigt[1][1])
            and iszero(self.CVoigt[0][0] - self.CVoigt[2][2])
            and iszero(self.CVoigt[0][0] - self.CVoigt[1][1])
            and iszero(self.CVoigt[0][0] - self.CVoigt[2][2])
            and iszero(self.CVoigt[3][3] - self.CVoigt[4][4])
            and iszero(self.CVoigt[3][3] - self.CVoigt[5][5])
            and iszero(self.CVoigt[0][1] - self.CVoigt[0][2])
            and iszero(self.CVoigt[0][1] - self.CVoigt[1][2])
        )

    def Young(self, x):
        """
        A method to calculate the Young's Modulus
        
        Parameters
        ----------
        x : Size 2 tuple of the spherical angle to calculate the value
            The first index is theta
            The second index is phi
            
        Returns
        -------
        Returns : float
    
        """
        a = dirVec(x[0], x[1])
        r = sum(
            [
                a[i] * a[j] * a[k] * a[l] * self.Smat[i][j][k][l]
                for i in range(3)
                for j in range(3)
                for k in range(3)
                for l in range(3)
            ]
        )
        return 1 / r

    def Young_2(self, x, y):
        """
        Another method to calculate the Young's Modulus
        
        Parameters
        ----------
        x : float
            Represents the spherical theta angle
        y : float
            Represents the spherical phi angle
            
        Returns
        -------
        Returns : float
    
        """
        a = dirVec(x, y)
        r = sum(
            [
                a[i] * a[j] * a[k] * a[l] * self.Smat[i][j][k][l]
                for i in range(3)
                for j in range(3)
                for k in range(3)
                for l in range(3)
            ]
        )
        return 1 / r

    def LC(self, x):
        """
        A method to calculate the Linear compression
        
        Parameters
        ----------
        x : Size 2 tuple of the spherical angle to calculate the value
            The first index is theta
            The second index is phi
            
        Returns
        -------
        Returns : float
    
        """
        a = dirVec(x[0], x[1])
        r = sum(
            [
                a[i] * a[j] * self.Smat[i][j][k][k]
                for i in range(3)
                for j in range(3)
                for k in range(3)
            ]
        )
        return 1000 * r

    def LC_2(self, x, y):
        """
        Another method to calculate the Young's Modulus
        
        Parameters
        ----------
        x : float
            Represents the spherical theta angle
        y : float
                Represents the spherical phi angle
            
        Returns
        -------
        Returns : float
    
        """
        a = dirVec(x, y)
        r = sum(
            [
                a[i] * a[j] * self.Smat[i][j][k][k]
                for i in range(3)
                for j in range(3)
                for k in range(3)
            ]
        )
        return 1000 * r

    def shear(self, x):
        """
        A method to calculate the Young's Modulus
        
        Parameters
        ----------
        x : Size 3 tuple of the euler angles to calculate the value
            The first index is theta
            The second index is phi
            The second index is chi
            
        Returns
        -------
        Returns : float
    
        """
        a = dirVec(x[0], x[1])
        b = dirVec2(x[0], x[1], x[2])
        r = sum(
            [
                a[i] * b[j] * a[k] * b[l] * self.Smat[i][j][k][l]
                for i in range(3)
                for j in range(3)
                for k in range(3)
                for l in range(3)
            ]
        )
        return 1 / (4 * r)

    def Poisson(self, x):
        """
        A method to calculate the Young's Modulus
        
        Parameters
        ----------
        x : Size 3 tuple of the euler angles to calculate the value
            The first index is theta
            The second index is phi
            The second index is chi
            
        Returns
        -------
        Returns : float
    
        """
        a = dirVec(x[0], x[1])
        b = dirVec2(x[0], x[1], x[2])
        r1 = sum(
            [
                a[i] * a[j] * b[k] * b[l] * self.Smat[i][j][k][l]
                for i in range(3)
                for j in range(3)
                for k in range(3)
                for l in range(3)
            ]
        )
        r2 = sum(
            [
                a[i] * a[j] * a[k] * a[l] * self.Smat[i][j][k][l]
                for i in range(3)
                for j in range(3)
                for k in range(3)
                for l in range(3)
            ]
        )
        return -r1 / r2

    def averages(self):
        """
        A method to calculate Voigt, Reuss, and Hill averages of the elastic moduli
        
        Parameters
        ----------
        None
            
        Returns
        -------
        Returns : A size 3 list of list containing the elastic moduli
            The first index of the outer list is Voigt averages
            The second index of the outer list is Reuss averages
            The first index of the outer list is Hill averages
            
            The corresponding indicex on the inner lists are
                The first index is the bulk modulus
                The first index is the young's modulus
                The first index is the shear modulus
                The first index is the poisson's ratio
    
        """
        A = (self.CVoigt[0][0] + self.CVoigt[1][1] + self.CVoigt[2][2]) / 3
        B = (self.CVoigt[1][2] + self.CVoigt[0][2] + self.CVoigt[0][1]) / 3
        C = (self.CVoigt[3][3] + self.CVoigt[4][4] + self.CVoigt[5][5]) / 3
        a = (self.SVoigt[0][0] + self.SVoigt[1][1] + self.SVoigt[2][2]) / 3
        b = (self.SVoigt[1][2] + self.SVoigt[0][2] + self.SVoigt[0][1]) / 3
        c = (self.SVoigt[3][3] + self.SVoigt[4][4] + self.SVoigt[5][5]) / 3

        KV = (A + 2 * B) / 3
        GV = (A - B + 3 * C) / 5

        KR = 1 / (3 * a + 6 * b)
        GR = 5 / (4 * a - 4 * b + 3 * c)

        KH = (KV + KR) / 2
        GH = (GV + GR) / 2

        return [
            [
                KV,
                1 / (1 / (3 * GV) + 1 / (9 * KV)),
                GV,
                (1 - 3 * GV / (3 * KV + GV)) / 2,
            ],
            [
                KR,
                1 / (1 / (3 * GR) + 1 / (9 * KR)),
                GR,
                (1 - 3 * GR / (3 * KR + GR)) / 2,
            ],
            [
                KH,
                1 / (1 / (3 * GH) + 1 / (9 * KH)),
                GH,
                (1 - 3 * GH / (3 * KH + GH)) / 2,
            ],
        ]

    def shear2D(self, x):
        """
        A method to calculate the Shear modulus in a plane
        
        Parameters
        ----------
        x : Size 2 tuple of the spherical angle to calculate the value
            The first index is theta
            The second index is phi
            
        Returns
        -------
        Returns : size 2 tuple of floats.
            The first index is the positive value
            The second index is the maximum value
    
        """
        
        ftol = 0.001
        xtol = 0.01

        def func1(z):
            return self.shear([x[0], x[1], z])

        r1 = optimize.minimize(
            func1,
            np.pi / 2.0,
            args=(),
            method="Powell",
            options={"xtol": xtol, "ftol": ftol},
        )  # , bounds=[(0.0,np.pi)])

        def func2(z):
            return -self.shear([x[0], x[1], z])

        r2 = optimize.minimize(
            func2,
            np.pi / 2.0,
            args=(),
            method="Powell",
            options={"xtol": xtol, "ftol": ftol},
        )  # , bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun))

    def shear3D(self, x, y, guess1=np.pi / 2.0, guess2=np.pi / 2.0):
        """
        A method to calculate the Shear modulus in 3D
        
        Parameters
        ----------
        x : float
            Represents the spherical theta value 
        y : float
            Represents the spherical phi value 
        guess1 : float
            A starting guess in the minimization scheme to determine the positive value. Default is pi/2.
        guess1 : float
            A second guess in the minimization scheme to determine the maximum value. Default is pi/2.
            
        Returns
        -------
        Returns : size 4 tuple of floats.
            The first index is the positive value
            The second index is the maximum value
            The third index is the solutions of the optimization of the positive value
            The fourth index is thesolutions of the optimization of the maximum value
    
        """
        tol = 0.005

        def func1(z):
            return self.shear([x, y, z])

        r1 = optimize.minimize(
            func1, guess1, args=(), method="COBYLA", options={"tol": tol}
        )  # , bounds=[(0.0,np.pi)])

        def func2(z):
            return -self.shear([x, y, z])

        r2 = optimize.minimize(
            func2, guess2, args=(), method="COBYLA", options={"tol": tol}
        )  # , bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun), float(r1.x), float(r2.x))

    def Pugh_ratio(self,x):
        """
        A method to calculate the Pugh's Ratio
        
        Parameters
        ----------
        x : Size 3 tuple of the euler angles to calculate the value
            The first index is theta
            The second index is phi
            The second index is chi
            
        Returns
        -------
        Returns : float
    
        """
        return self.averages()[2][0]/self.shear(x)
    
    def Bulk(self,x):
        """
        A method to calculate the Bulk Ratio
        
        Parameters
        ----------
        x : Size 3 tuple of the euler angles to calculate the value
            The first index is theta
            The second index is phi
            The second index is chi
            
        Returns
        -------
        Returns : float
    
        """
        return 2*self.shear(x) * (1+ self.Poisson(x)) / (3 * (1  - 2*self.Poisson(x)))

    def Compression_Speed(self,x):
        """
        A method to calculate the Compression Speed
        
        Parameters
        ----------
        x : Size 3 tuple of the euler angles to calculate the value
            The first index is theta
            The second index is phi
            The second index is chi
            
        Returns
        -------
        Returns : float
    
        """
        return ((10**9)*(self.averages()[2][0]+(4/3)*self.shear(x))/self.density)**0.5
    
    def Shear_Speed(self,x):
        """
        A method to calculate the Shear Speed
        
        Parameters
        ----------
        x : Size 3 tuple of the euler angles to calculate the value
            The first index is theta
            The second index is phi
            The second index is chi
            
        Returns
        -------
        Returns : float
    
        """
        return (self.shear(x)*(10**9) /self.density)**0.5
    
    def Ratio_Compression_Shear(self,x):
        """
        A method to calculate the Compression Speed
        
        Parameters
        ----------
        x : Size 3 tuple of the euler angles to calculate the value
            The first index is theta
            The second index is phi
            The second index is chi
            
        Returns
        -------
        Returns : float
    
        """
        return (self.Compression_Speed(x)/self.Shear_Speed(x))**2
    
    def Debye_Speed(self,x):
        """
        A method to calculate the Debye Speed
        
        Parameters
        ----------
        x : Size 3 tuple of the euler angles to calculate the value
            The first index is theta
            The second index is phi
            The second index is chi
            
        Returns
        -------
        Returns : float
    
        """
        return self.Compression_Speed(x)*self.Shear_Speed(x)/(2*self.Shear_Speed(x)**3 +self.Compression_Speed(x)**3)**(1/3)
        
    def Poisson2D(self, x):
        """
        A method to calculate the Poisson's Ratio in a plane
        
        Parameters
        ----------
        x : Size 2 tuple of the spherical angle to calculate the value
            The first index is theta
            The second index is phi
            
        Returns
        -------
        Returns : size 3 tuple of floats.
            The first index is the positive value
            The second index is the maximum value
            The second index is the negativem value
    
        """
        ftol = 0.001
        xtol = 0.01

        def func1(z):
            return self.Poisson([x[0], x[1], z])

        r1 = optimize.minimize(
            func1,
            np.pi / 2.0,
            args=(),
            method="Powell",
            options={"xtol": xtol, "ftol": ftol},
        )  # , bounds=[(0.0,np.pi)])

        def func2(z):
            return -self.Poisson([x[0], x[1], z])

        r2 = optimize.minimize(
            func2,
            np.pi / 2.0,
            args=(),
            method="Powell",
            options={"xtol": xtol, "ftol": ftol},
        )  # , bounds=[(0.0,np.pi)])
        return (min(0, float(r1.fun)), max(0, float(r1.fun)), -float(r2.fun))

    def poisson3D(self, x, y, guess1=np.pi / 2.0, guess2=np.pi / 2.0):
        """
        A method to calculate the Shear modulus in 3D
        
        Parameters
        ----------
        x : float
            Represents the spherical theta value 
        y : float
            Represents the spherical phi value 
        guess1 : float
            A starting guess in the minimization scheme to determine the positive value. Default is pi/2.
        guess1 : float
            A second guess in the minimization scheme to determine the maximum value. Default is pi/2.
            
        Returns
        -------
        Returns : size 5 tuple of floats.
            The first index is the positive value
            The second index is the maximum value
            The third index is the maximum value
            The fourth index is the solutions of the optimization of the positive value
            The fifth index is thesolutions of the optimization of the maximum value
    
        """
        
        tol = 0.005

        def func1(z):
            return self.Poisson([x, y, z])

        r1 = optimize.minimize(
            func1, guess1, args=(), method="COBYLA", options={"tol": tol}
        )  # , bounds=[(0.0,np.pi)])

        def func2(z):
            return -self.Poisson([x, y, z])

        r2 = optimize.minimize(
            func2, guess2, args=(), method="COBYLA", options={"tol": tol}
        )  # , bounds=[(0.0,np.pi)])
        return (
            min(0, float(r1.fun)),
            max(0, float(r1.fun)),
            -float(r2.fun),
            float(r1.x),
            float(r2.x),
        )
    
    def bulk2D(self, x):
        """
        A method to calculate the Bulk modulus in a plane
        
        Parameters
        ----------
        x : Size 2 tuple of the spherical angle to calculate the value
            The first index is theta
            The second index is phi
            
        Returns
        -------
        Returns : size 2 tuple of floats.
            The first index is the positive value
            The second index is the maximum value
    
        """
        
        ftol = 0.001
        xtol = 0.01

        def func1(z):
            return self.Bulk([x[0], x[1], z])

        r1 = optimize.minimize(
            func1,
            np.pi / 2.0,
            args=(),
            method="Powell",
            options={"xtol": xtol, "ftol": ftol},
        )  # , bounds=[(0.0,np.pi)])

        def func2(z):
            return -self.Bulk([x[0], x[1], z])

        r2 = optimize.minimize(
            func2,
            np.pi / 2.0,
            args=(),
            method="Powell",
            options={"xtol": xtol, "ftol": ftol},
        )  # , bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun))

    def bulk3D(self, x, y, guess1=np.pi / 2.0, guess2=np.pi / 2.0):
        """
        A method to calculate the Bulk modulus in 3D
        
        Parameters
        ----------
        x : float
            Represents the spherical theta value 
        y : float
            Represents the spherical phi value 
        guess1 : float
            A starting guess in the minimization scheme to determine the positive value. Default is pi/2.
        guess1 : float
            A second guess in the minimization scheme to determine the maximum value. Default is pi/2.
            
        Returns
        -------
        Returns : size 4 tuple of floats.
            The first index is the positive value
            The second index is the maximum value
            The third index is the solutions of the optimization of the positive value
            The fourth index is thesolutions of the optimization of the maximum value
    
        """
        tol = 0.005

        def func1(z):
            return self.Bulk([x, y, z])

        r1 = optimize.minimize(
            func1, guess1, args=(), method="COBYLA", options={"tol": tol}
        )  # , bounds=[(0.0,np.pi)])

        def func2(z):
            return -self.Bulk([x, y, z])

        r2 = optimize.minimize(
            func2, guess2, args=(), method="COBYLA", options={"tol": tol}
        )  # , bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun), float(r1.x), float(r2.x))
    
    
    def pugh_ratio2D(self, x):
        """
        A method to calculate the Pugh's Ratio in a plane
        
        Parameters
        ----------
        x : Size 2 tuple of the spherical angle to calculate the value
            The first index is theta
            The second index is phi
            
        Returns
        -------
        Returns : size 2 tuple of floats.
            The first index is the positive value
            The second index is the maximum value
        """
        ftol = 0.001
        xtol = 0.01

        def func1(z):
            return self.Pugh_ratio([x[0], x[1], z])

        r1 = optimize.minimize(
            func1,
            np.pi / 2.0,
            args=(),
            method="Powell",
            options={"xtol": xtol, "ftol": ftol},
        )  # , bounds=[(0.0,np.pi)])

        def func2(z):
            return -self.Pugh_ratio([x[0], x[1], z])

        r2 = optimize.minimize(
            func2,
            np.pi / 2.0,
            args=(),
            method="Powell",
            options={"xtol": xtol, "ftol": ftol},
        )  # , bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun))

    def pugh_ratio3D(self, x, y, guess1=np.pi / 2.0, guess2=np.pi / 2.0):
        """
        A method to calculate the Shear modulus in 3D
        
        Parameters
        ----------
        x : float
            Represents the spherical theta value 
        y : float
            Represents the spherical phi value 
        guess1 : float
            A starting guess in the minimization scheme to determine the positive value. Default is pi/2.
        guess1 : float
            A second guess in the minimization scheme to determine the maximum value. Default is pi/2.
            
        Returns
        -------
        Returns : size 4 tuple of floats.
            The first index is the positive value
            The second index is the maximum value
            The third index is the solutions of the optimization of the positive value
            The fourth index is thesolutions of the optimization of the maximum value
    
        """
        tol = 0.005

        def func1(z):
            return self.Pugh_ratio([x, y, z])

        r1 = optimize.minimize(
            func1, guess1, args=(), method="COBYLA", options={"tol": tol}
        )  # , bounds=[(0.0,np.pi)])

        def func2(z):
            return -self.Pugh_ratio([x, y, z])

        r2 = optimize.minimize(
            func2, guess2, args=(), method="COBYLA", options={"tol": tol}
        )  # , bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun), float(r1.x), float(r2.x))
    
    def compressionSpeed2D(self, x, density = None):
        """
        A method to calculate the Compression Speed in a plane
        
        Parameters
        ----------
        x : Size 2 tuple of the spherical angle to calculate the value
            The first index is theta
            The second index is phi
        density : float
            This is the desnity of the structure if it is provided. The units should be in kg/m^3. The Default is None
            
        Returns
        -------
        Returns : size 2 tuple of floats.
            The first index is the positive value
            The second index is the maximum value
        """
        ftol = 0.001
        xtol = 0.01

        def func1(z):
            return self.Compression_Speed(x[0],x[1],z)


        r1 = optimize.minimize(
            func1,
            np.pi / 2.0,
            args=(),
            method="Powell",
            options={"xtol": xtol, "ftol": ftol},
        )  # , bounds=[(0.0,np.pi)])

        def func2(z):
            return -self.Compression_Speed(x[0],x[1],z)


        r2 = optimize.minimize(
            func2,
            np.pi / 2.0,
            args=(),
            method="Powell",
            options={"xtol": xtol, "ftol": ftol},
        )  # , bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun))
 
    def compressionSpeed3D(self, x, y, guess1=np.pi / 2.0, guess2=np.pi / 2.0, density = None):
        """
        A method to calculate the Compression in 3D
        
        Parameters
        ----------
        x : float
            Represents the spherical theta value 
        y : float
            Represents the spherical phi value 
        guess1 : float
            A starting guess in the minimization scheme to determine the positive value. Default is pi/2.
        guess1 : float
            A second guess in the minimization scheme to determine the maximum value. Default is pi/2.
        density : float
            This is the desnity of the structure if it is provided. The units should be in kg/m^3. The Default is None
            
        Returns
        -------
        Returns : size 4 tuple of floats.
            The first index is the positive value
            The second index is the maximum value
            The third index is the solutions of the optimization of the positive value
            The fourth index is thesolutions of the optimization of the maximum value
    
        """
        tol = 0.005

        def func1(z):
            return self.Compression_Speed(x,y,z)
    
        r1 = optimize.minimize(
            func1, guess1, args=(), method="COBYLA", options={"tol": tol}
        )  # , bounds=[(0.0,np.pi)])
    
        def func2(z):
            return -self.Compression_Speed(x,y,z)
    
        r2 = optimize.minimize(
            func2, guess2, args=(), method="COBYLA", options={"tol": tol}
        )  # , bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun), float(r1.x), float(r2.x))
    
    def shearSpeed2D(self, x, density = None):
        """
        A method to calculate the Shear Speed in a plane
        
        Parameters
        ----------
        x : Size 2 tuple of the spherical angle to calculate the value
            The first index is theta
            The second index is phi
        density : float
            This is the desnity of the structure if it is provided. The units should be in kg/m^3. The Default is None
            
        Returns
        -------
        Returns : size 2 tuple of floats.
            The first index is the positive value
            The second index is the maximum value
        """
        ftol = 0.001
        xtol = 0.01

        def func1(z):
            return self.Shear_Speed(x[0],x[1],z)
        r1 = optimize.minimize(
            func1,
            np.pi / 2.0,
            args=(),
            method="Powell",
            options={"xtol": xtol, "ftol": ftol},
        )  # , bounds=[(0.0,np.pi)])

        def func2(z):
            return -self.Shear_Speed(x[0],x[1],z)

        r2 = optimize.minimize(
            func2,
            np.pi / 2.0,
            args=(),
            method="Powell",
            options={"xtol": xtol, "ftol": ftol},
        )  # , bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun))

    def shearSpeed3D(self, x, y, guess1=np.pi / 2.0, guess2=np.pi / 2.0, density=None):
        """
        A method to calculate the Shear Speed in 3D
        
        Parameters
        ----------
        x : float
            Represents the spherical theta value 
        y : float
            Represents the spherical phi value 
        guess1 : float
            A starting guess in the minimization scheme to determine the positive value. Default is pi/2.
        guess1 : float
            A second guess in the minimization scheme to determine the maximum value. Default is pi/2.
        density : float
            This is the desnity of the structure if it is provided. The units should be in kg/m^3. The Default is None
            
        Returns
        -------
        Returns : size 4 tuple of floats.
            The first index is the positive value
            The second index is the maximum value
            The third index is the solutions of the optimization of the positive value
            The fourth index is thesolutions of the optimization of the maximum value
    
        """
        tol = 0.005

        def func1(z):
            return self.Shear_Speed(x,y,z)

        r1 = optimize.minimize(
            func1, guess1, args=(), method="COBYLA", options={"tol": tol}
        )  # , bounds=[(0.0,np.pi)])

        def func2(z):
            return -self.Shear_Speed(x,y,z)

        r2 = optimize.minimize(
            func2, guess2, args=(), method="COBYLA", options={"tol": tol}
        )  # , bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun), float(r1.x), float(r2.x))
    
    def ratio_compressional_shear2D(self, x, density = None):
        """
        A method to calculate the ratio of the compression speed to the shear speed in a plane
        
        Parameters
        ----------
        x : Size 2 tuple of the spherical angle to calculate the value
            The first index is theta
            The second index is phi
        density : float
            This is the desnity of the structure if it is provided. The units should be in kg/m^3. The Default is None
            
        Returns
        -------
        Returns : size 2 tuple of floats.
            The first index is the positive value
            The second index is the maximum value
        """
        ftol = 0.001
        xtol = 0.01
        
        def func1(z):
            return self.Ratio_Compression_Shear(x[0],x[1],z)

        r1 = optimize.minimize(
            func1,
            np.pi / 2.0,
            args=(),
            method="Powell",
            options={"xtol": xtol, "ftol": ftol},
        )  # , bounds=[(0.0,np.pi)])

        def func2(z):
            return -self.Ratio_Compression_Shear(x[0],x[1],z)

        r2 = optimize.minimize(
            func2,
            np.pi / 2.0,
            args=(),
            method="Powell",
            options={"xtol": xtol, "ftol": ftol},
        )  # , bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun))
    
    def ratio_compressional_shear3D(self, x, y, guess1=np.pi / 2.0, guess2=np.pi / 2.0,density = None):
        """
        A method to calculate the ratio of the compression speed to the shear speed in 3D
        
        Parameters
        ----------
        x : float
            Represents the spherical theta value 
        y : float
            Represents the spherical phi value 
        guess1 : float
            A starting guess in the minimization scheme to determine the positive value. Default is pi/2.
        guess1 : float
            A second guess in the minimization scheme to determine the maximum value. Default is pi/2.
        density : float
            This is the desnity of the structure if it is provided. The units should be in kg/m^3. The Default is None
            
        Returns
        -------
        Returns : size 4 tuple of floats.
            The first index is the positive value
            The second index is the maximum value
            The third index is the solutions of the optimization of the positive value
            The fourth index is thesolutions of the optimization of the maximum value
    
        """
        tol = 0.005

        def func1(z):
            return self.Ratio_Compression_Shear(x,y,z)

        r1 = optimize.minimize(
            func1, guess1, args=(), method="COBYLA", options={"tol": tol}
        )  # , bounds=[(0.0,np.pi)])

        def func2(z):
            return -self.Ratio_Compression_Shear(x,y,z)

        r2 = optimize.minimize(
            func2, guess2, args=(), method="COBYLA", options={"tol": tol}
        )  # , bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun), float(r1.x), float(r2.x))
     
    def debyeSpeed2D(self, x,density = None):
        """
        A method to calculate the Debye speed in a plane
        
        Parameters
        ----------
        x : Size 2 tuple of the spherical angle to calculate the value
            The first index is theta
            The second index is phi
        density : float
            This is the desnity of the structure if it is provided. The units should be in kg/m^3. The Default is None
        Returns
        -------
        Returns : size 2 tuple of floats.
            The first index is the positive value
            The second index is the maximum value
        """
        ftol = 0.001
        xtol = 0.01

        def v_p(z):
            return (3*self.averages()[2][0]*(10**9)*(1-self.Poisson([x[0], x[1], z]))/(2*density*(1+self.Poisson([x[0], x[1], z]))))**0.5

        def v_s(z):
            return ((1/density)*(10**9)*self.shear([x[0], x[1], z]))**0.5
        
        def func1(z):
            return self.Debye_Speed(x[0],x[1],z)
        r1 = optimize.minimize(
            func1,
            np.pi / 2.0,
            args=(),
            method="Powell",
            options={"xtol": xtol, "ftol": ftol},
        )  # , bounds=[(0.0,np.pi)])

        def func2(z):
            return -self.Debye_Speed(x[0],x[1],z)

        r2 = optimize.minimize(
            func2,
            np.pi / 2.0,
            args=(),
            method="Powell",
            options={"xtol": xtol, "ftol": ftol},
        )  # , bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun))

    def debyeSpeed3D(self, x, y, guess1=np.pi / 2.0, guess2=np.pi / 2.0,density= None):
        """
        A method to calculate the ratio of the Debye speed in 3D
        
        Parameters
        ----------
        x : float
            Represents the spherical theta value 
        y : float
            Represents the spherical phi value 
        guess1 : float
            A starting guess in the minimization scheme to determine the positive value. Default is pi/2.
        guess1 : float
            A second guess in the minimization scheme to determine the maximum value. Default is pi/2.
        density : float
            This is the desnity of the structure if it is provided. The units should be in kg/m^3. The Default is None
            
        Returns
        -------
        Returns : size 4 tuple of floats.
            The first index is the positive value
            The second index is the maximum value
            The third index is the solutions of the optimization of the positive value
            The fourth index is thesolutions of the optimization of the maximum value
    
        """
        tol = 0.005
        
        def v_p(z):
            return (3*self.averages()[2][0]*(10**9)*(1-self.Poisson([x, y, z]))/(2*density*(1+self.Poisson([x, y, z]))))**0.5

        def v_s(z):
            return ((1/density)*(10**9)*self.shear([x, y, z]))**0.5
        
        def func1(z):
            return self.Debye_Speed(x,y,z)

        r1 = optimize.minimize(
            func1, guess1, args=(), method="COBYLA", options={"tol": tol}
        )  # , bounds=[(0.0,np.pi)])

        def func2(z):
            return -self.Debye_Speed(x,y,z)

        r2 = optimize.minimize(
            func2, guess2, args=(), method="COBYLA", options={"tol": tol}
        )  # , bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun), float(r1.x), float(r2.x))


class ElasticOrtho(Elastic):
    """An elastic tensor, for the specific case of an orthorhombic system"""

    def __init__(self, arg):
        """Initialize from a matrix, or from an Elastic object"""
        if type(arg) == str:
            Elastic.__init__(self, arg)
        elif isinstance(arg, Elastic):
            self.CVoigt = arg.CVoigt
            self.SVoigt = arg.SVoigt
            self.Smat = arg.Smat
            self.density = arg.density
        else:
            raise TypeError(
                "ElasticOrtho constructor argument should be string or Elastic object"
            )

    def Young(self, x):
        ct2 = math.cos(x[0]) ** 2
        st2 = 1 - ct2
        cf2 = math.cos(x[1]) ** 2
        sf2 = 1 - cf2
        s11 = self.Smat[0][0][0][0]
        s22 = self.Smat[1][1][1][1]
        s33 = self.Smat[2][2][2][2]
        s44 = 4 * self.Smat[1][2][1][2]
        s55 = 4 * self.Smat[0][2][0][2]
        s66 = 4 * self.Smat[0][1][0][1]
        s12 = self.Smat[0][0][1][1]
        s13 = self.Smat[0][0][2][2]
        s23 = self.Smat[1][1][2][2]
        return 1 / (
            ct2 ** 2 * s33
            + 2 * cf2 * ct2 * s13 * st2
            + cf2 * ct2 * s55 * st2
            + 2 * ct2 * s23 * sf2 * st2
            + ct2 * s44 * sf2 * st2
            + cf2 ** 2 * s11 * st2 ** 2
            + 2 * cf2 * s12 * sf2 * st2 ** 2
            + cf2 * s66 * sf2 * st2 ** 2
            + s22 * sf2 ** 2 * st2 ** 2
        )

    def LC(self, x):
        ct2 = math.cos(x[0]) ** 2
        cf2 = math.cos(x[1]) ** 2
        s11 = self.Smat[0][0][0][0]
        s22 = self.Smat[1][1][1][1]
        s33 = self.Smat[2][2][2][2]
        s12 = self.Smat[0][0][1][1]
        s13 = self.Smat[0][0][2][2]
        s23 = self.Smat[1][1][2][2]
        return 1000 * (
            ct2 * (s13 + s23 + s33)
            + (cf2 * (s11 + s12 + s13) + (s12 + s22 + s23) * (1 - cf2)) * (1 - ct2)
        )

    def shear(self, x):
        ct = math.cos(x[0])
        ct2 = ct * ct
        st2 = 1 - ct2
        cf = math.cos(x[1])
        sf = math.sin(x[1])
        sf2 = sf * sf
        cx = math.cos(x[2])
        cx2 = cx * cx
        sx = math.sin(x[2])
        sx2 = 1 - cx2
        s11 = self.Smat[0][0][0][0]
        s22 = self.Smat[1][1][1][1]
        s33 = self.Smat[2][2][2][2]
        s44 = 4 * self.Smat[1][2][1][2]
        s55 = 4 * self.Smat[0][2][0][2]
        s66 = 4 * self.Smat[0][1][0][1]
        s12 = self.Smat[0][0][1][1]
        s13 = self.Smat[0][0][2][2]
        s23 = self.Smat[1][1][2][2]
        r = (
            ct2 * ct2 * cx2 * s44 * sf2
            + cx2 * s44 * sf2 * st2 * st2
            + 4 * cf ** 3 * ct * cx * (-2 * s11 + 2 * s12 + s66) * sf * st2 * sx
            + 2
            * cf
            * ct
            * cx
            * sf
            * (
                ct2 * (s44 - s55)
                + (
                    4 * s13
                    - 4 * s23
                    - s44
                    + s55
                    - 4 * s12 * sf2
                    + 4 * s22 * sf2
                    - 2 * s66 * sf2
                )
                * st2
            )
            * sx
            + s66 * sf2 * sf2 * st2 * sx2
            + cf ** 4 * st2 * (4 * ct2 * cx2 * s11 + s66 * sx2)
            + ct2
            * (
                2 * cx2 * (2 * s33 + sf2 * (-4 * s23 - s44 + 2 * s22 * sf2)) * st2
                + s55 * sf2 * sx2
            )
            + cf ** 2
            * (
                ct2 * ct2 * cx2 * s55
                + ct2
                * (
                    -2 * cx2 * (4 * s13 + s55 - 2 * (2 * s12 + s66) * sf2) * st2
                    + s44 * sx2
                )
                + st2
                * (
                    cx2 * s55 * st2
                    + 2 * (2 * s11 - 4 * s12 + 2 * s22 - s66) * sf2 * sx2
                )
            )
        )
        return 1 / r

    def Poisson(self, x):
        ct = math.cos(x[0])
        ct2 = ct * ct
        st2 = 1 - ct2
        cf = math.cos(x[1])
        sf = math.sin(x[1])
        cx = math.cos(x[2])
        sx = math.sin(x[2])
        s11 = self.Smat[0][0][0][0]
        s22 = self.Smat[1][1][1][1]
        s33 = self.Smat[2][2][2][2]
        s44 = 4 * self.Smat[1][2][1][2]
        s55 = 4 * self.Smat[0][2][0][2]
        s66 = 4 * self.Smat[0][1][0][1]
        s12 = self.Smat[0][0][1][1]
        s13 = self.Smat[0][0][2][2]
        s23 = self.Smat[1][1][2][2]

        return (
            -(ct ** 2 * cx ** 2 * s33 * st2)
            - cf ** 2 * cx ** 2 * s13 * st2 * st2
            - cx ** 2 * s23 * sf ** 2 * st2 * st2
            + ct * cx * s44 * sf * st2 * (ct * cx * sf + cf * sx)
            - ct ** 2 * s23 * (ct * cx * sf + cf * sx) ** 2
            - cf ** 2 * s12 * st2 * (ct * cx * sf + cf * sx) ** 2
            - s22 * sf ** 2 * st2 * (ct * cx * sf + cf * sx) ** 2
            + cf * ct * cx * s55 * st2 * (cf * ct * cx - sf * sx)
            - cf * s66 * sf * st2 * (ct * cx * sf + cf * sx) * (cf * ct * cx - sf * sx)
            - ct ** 2 * s13 * (cf * ct * cx - sf * sx) ** 2
            - cf ** 2 * s11 * st2 * (cf * ct * cx - sf * sx) ** 2
            - s12 * sf ** 2 * st2 * (cf * ct * cx - sf * sx) ** 2
        ) / (
            ct ** 4 * s33
            + 2 * cf ** 2 * ct ** 2 * s13 * st2
            + cf ** 2 * ct ** 2 * s55 * st2
            + 2 * ct ** 2 * s23 * sf ** 2 * st2
            + ct ** 2 * s44 * sf ** 2 * st2
            + cf ** 4 * s11 * st2 * st2
            + 2 * cf ** 2 * s12 * sf ** 2 * st2 * st2
            + cf ** 2 * s66 * sf ** 2 * st2 * st2
            + s22 * sf ** 4 * st2 * st2
        )


