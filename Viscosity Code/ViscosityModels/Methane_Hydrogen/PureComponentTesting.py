import sys
import os

# Add ViscosityCode to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Methane_Hydrogen.HythaneDataReader import viscData

#Methane data
methane_data = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Methane\Methane - NIST DATA.csv", skiprows=1)
methane_T, methane_P, methane_eta =  methane_data['T [K] '], methane_data['P [MPa]'], methane_data['eta [µPa*s]']

Chuang_0 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hythane\Chuang et al. (6,6).csv", skiprows=5)
Chuang_0_T, Chuang_0_P, Chuang_0_eta = Chuang_0['T [K] '], Chuang_0['P [Mpa]'], Chuang_0['eta [µPa*s]']


#Hydrogen data
Chuang_1976 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hydrogen\Chuang.csv", skiprows=5)
Chuang_1976_T, Chuang_1976_P, Chuang_1976_eta = Chuang_1976['T [K] '], Chuang_1976['P [Mpa]'], Chuang_1976['eta [µPa*s]']
Barua = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hydrogen\Barua.csv", skiprows=5)
Barua_T, Barua_P, Barua_eta = Barua['T [K] '], Barua['P [Mpa]'], Barua['eta [µPa*s]']

Chuang_100 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hythane\Chuang et al. (1,6).csv", skiprows=5)
Chuang_100_T, Chuang_100_P, Chuang_100_eta = Chuang_100['T [K] '], Chuang_100['P [Mpa]'], Chuang_100['eta [µPa*s]']


def ComparisonPlot(T, component, savename):
    plt.figure(figsize=(14,7))
    if component=="hydrogen" or component=="Hydrogen":
        plt.scatter(Chuang_1976_P, Chuang_1976_eta, label='Chuang et al. (1973)', edgecolors='black', facecolors='none', marker='o', )
        plt.scatter(Barua_P, Barua_eta, label='Barua et al.', edgecolors='black', facecolors='none', marker='^')

        plt.scatter(Chuang_100_P, Chuang_100_eta, label='Chuang et al. (1976)', color='red', marker='+')

    elif component=="methane" or component=="Methane":
        plt.scatter(methane_P, methane_eta, label='NIST', edgecolors='black', facecolors='none', marker='s')

        plt.scatter(Chuang_0_P, Chuang_0_eta, label='Chuang et al. (1976)', color='red', marker='2')

    else:
        raise ValueError("Invalid component. Choose 'hydrogen' or 'methane'.")
    
    plt.grid()
    plt.xlabel('Pressure [MPa]')
    plt.ylabel('Viscosity [µPa$\cdot$s]')
    
    # Save the figure
    plt.savefig(f"{savename}.png", dpi=500, bbox_inches='tight')
    plt.clf()