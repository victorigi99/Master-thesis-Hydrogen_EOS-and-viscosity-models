import sys
import os

# Add ViscosityCode to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time

import numpy as np
from neqsim.thermo import TPflash, fluid
from Hydrogen.HydrogenModels import Muzny_mod
from Methane.PureMethaneModel import LBCMod_methane, LBCMod_methane_mix
from Methane_Hydrogen.MixturePlotting import SingleARDPlot, GroupedARDPlot, viscosityPlot


def getMixingRule(T, P, mixingRule, moleFractions):
    if mixingRule == "Linear":
        eta = LinearMixingRule(T, P, moleFractions)
    elif mixingRule == "Custom":
        eta = CustomMixingRule(T, P, moleFractions)
    elif mixingRule == "LBC":
        eta = LBCMixture(T, P, moleFractions)
    elif mixingRule == "LBCMod_methane":
        eta = LBCMod_methane_mix(T, P, moleFractions)
    elif mixingRule == "PFCT":
        eta = PFCTMixture(T, P, moleFractions)
    elif mixingRule == "PFCTMod":
        eta = PFCTMixtureMod(T, P, moleFractions)
    else:
        raise ValueError("Invalid mixing rule")
    return eta

def LBCMixture(T, P, moleFractions):
    fluid1 = fluid("srk")
    fluid1.addComponent("hydrogen", moleFractions[0])
    fluid1.addComponent("methane", moleFractions[1])
    fluid1.setMixingRule("classic")  # Define mixing rule
    fluid1.setTemperature(T, "K") 
    fluid1.setPressure(P, "MPa")
    TPflash(fluid1)
    fluid1.getPhase(0).getPhysicalProperties().setViscosityModel("LBC")
    fluid1.initProperties()     # Calculate properties
    eta = fluid1.getViscosity("Pas") * 10**6  # [µPa*s]
    return eta

def PFCTMixture(T, P, moleFractions):
    fluid1 = fluid("srk")
    fluid1.addComponent("hydrogen", moleFractions[0])
    fluid1.addComponent("methane", moleFractions[1])
    fluid1.setMixingRule("classic")  # Define mixing rule
    fluid1.setTemperature(T, "K") 
    fluid1.setPressure(P, "MPa")
    TPflash(fluid1)
    fluid1.getPhase(0).getPhysicalProperties().setViscosityModel("PFCT")
    fluid1.initProperties()     # Calculate properties
    eta = fluid1.getViscosity("Pas") * 10**6  # [µPa*s]
    return eta


def LinearMixingRule(T, P, moleFractions):
    methaneVisc = LBCMod_methane(T, P)
    hydrogenVisc = Muzny_mod(T, P)
    moleFractionSum = moleFractions[0] + moleFractions[1]
    #Checking that molefractions are correct:
    if moleFractionSum != 1:
        raise ValueError("Mole fractions do not summarize to 1. Check Mole fractions.")

    visc = methaneVisc*moleFractions[1] + hydrogenVisc*moleFractions[0]
    return visc

def CustomMixingRule(T, P, moleFractions):
    methaneVisc = LBCMod_methane(T, P)
    hydrogenVisc = Muzny_mod(T, P)
    moleFractionSum = moleFractions[0] + moleFractions[1]
    #Checking that molefractions are correct:
    if moleFractionSum != 1:
        raise ValueError("Mole fractions do not summarize to 1. Check Mole fractions.")

    visc = methaneVisc*moleFractions[1] + hydrogenVisc*moleFractions[0]
    return visc

def PFCTMixtureMod(T, P, moleFractions):
    eta_PFCT = PFCTMixture(T, P, moleFractions)
    A = 0.5 * np.exp(-(moleFractions[0] - 0.5)**2 / 0.4) - 7/(1 + np.exp(-8*(moleFractions[0] - 1.21)))  #Adds/subtracts from whole pressure range
    B = moleFractions[1] * 1/(1+np.exp(-0.5*(P-20.5))) - 0.32/((1+np.exp(-2*(P-10))) * (1+np.exp(2*(P-40)))) * 1/((1+np.exp(-40*(moleFractions[0]-0.33))) * (1+np.exp(40*(moleFractions[0]-0.8)))) #Adds/subtracts from high pressure range
    C = (P/50)/((1+np.exp(-5*(P-1))) * (1+np.exp(5*(P-9.8)))) * 1/(1+np.exp(20*(moleFractions[0]-0.6)))      #Adds/subtracts from low pressure range
    visc = eta_PFCT - A + B - C
    return visc


#SingleARDPlot([273.15], [0.787, 0.213], ["Linear", "Custom"], "Mixture_[0.787, 0.213]")
#GroupedARDPlot([273.15], [ [0, 1], [0.1942, 0.8058], [0.3375, 0.6625], [0.5337, 0.4663], [0.787, 0.213], [1, 0]], ["Linear", "LBC", "PFCT", "PFCTMod"], "Mixture_GroupedPlot")

'''
GroupedARDPlot([273.15], [ [0, 1], [0.1, 0.9], [0.1942, 0.8058], [0.2, 0.8], [0.3375, 0.6625], [0.5, 0.5], [0.5337, 0.4663], [0.787, 0.213], [1, 0]], ["Linear", "LBC", "PFCT", "PFCTMod"], "Mixture_Grouped_273.15")
GroupedARDPlot([298.15], [ [0, 1], [0.1, 0.9], [0.2, 0.8], [0.5, 0.5], [0.896, 0.104], [1, 0]], ["Linear", "LBC", "PFCT", "PFCTMod"], "Mixture_Grouped_298.15")
GroupedARDPlot([323.15], [ [0, 1], [0.1, 0.9], [0.2, 0.8], [0.5, 0.5], [0.896, 0.104], [1, 0]], ["Linear", "LBC", "PFCT", "PFCTMod"], "Mixture_Grouped_323.15")
'''

#GroupedARDPlot([273.15], [ [0, 1], [0.1, 0.9], [0.1942, 0.8058], [0.2, 0.8], [0.3375, 0.6625], [0.5, 0.5], [0.5337, 0.4663], [0.787, 0.213], [1, 0]], ["Linear", "LBC", "PFCT", "PFCTMod"], "Mixture_Grouped_273.15_TEST")
#GroupedARDPlot([298.15], [ [0, 1], [0.1, 0.9], [0.2, 0.8], [0.5, 0.5], [0.896, 0.104], [1, 0]], ["Linear", "LBC", "PFCT", "PFCTMod"], "Mixture_Grouped_298.15_TEST")
#GroupedARDPlot([323.15], [ [0, 1], [0.1, 0.9], [0.2, 0.8], [0.5, 0.5], [0.896, 0.104], [1, 0]], ["Linear", "LBC", "PFCT", "PFCTMod"], "Mixture_Grouped_323.15_TEST")
####FIKS [0.1, 0.9] FOR 298
T = 323.15  # K
P = 29.71  # MPa
composition = [0.1, 0.9]  # [Hydrogen, Methane]

#print("Methane:", LBCMod_methane(T, P))
#print("Hydrogen:", Muzny_mod(T, P))
#print(f"PFCTMixtureMod({T, P, composition}):", PFCTMixtureMod(T, P, composition))

#Lave trykk må ha mer Metan (mindre hydrogen)
#Høye tryk må ha mer Hydrogen (mindre metan)


#viscosityPlot([223.15, 273.15, 295.15, 298.15, 303.15, 323.15], [[0.1, 0.9], [0.1942, 0.8058], [0.2, 0.8], [0.3375, 0.6625], [0.5, 0.5], [0.5337, 0.46663], [0.787, 0.213], [0.9, 0.1]], "All_Mixture_Viscosities")
