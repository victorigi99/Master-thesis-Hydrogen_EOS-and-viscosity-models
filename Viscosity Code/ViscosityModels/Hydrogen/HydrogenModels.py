import sys
import os

# Add ViscosityCode to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from neqsim.thermo import TPflash, fluid
from Hydrogen.plotting import SingleARDPlot, ViscPlot, ARDMultipleModelsPlot


def getModelViscosity(T, P, viscosityModel):
    if viscosityModel == "Muzny":
        eta = Muzny(T, P)    #[µPa*s]
    elif viscosityModel == "LBC":
        eta = LBC(T, P)   #[µPa*s]
    elif viscosityModel == "PFCT":
        eta = PFCT(T, P)   #[µPa*s]
    elif viscosityModel == "F-Theory":
        eta = F_theory(T, P)   #[µPa*s]
    elif viscosityModel == "Modified Muzny":
        eta = Muzny_mod(T, P)   #[µPa*s]
    elif viscosityModel == "Muzny_mod":
        eta = Muzny_mod(T, P)   #[µPa*s]
    elif viscosityModel == "MuznyMod":
        eta = Muzny_mod(T, P)   #[µPa*s]
    else:
        raise ValueError("Invalid viscosity model")
    return eta

#The following model is used by NIST and is described in: https://pubs.acs.org/doi/10.1021/je301273j 
def Muzny(T, P):
    #Create a fluid to get the hydrogen density
    fluid1 = fluid("srk")
    fluid1.addComponent("hydrogen", 1.0)
    fluid1.setTemperature(T, "K")
    fluid1.setPressure(P, "MPa")
    TPflash(fluid1)
    fluid1.initProperties()     # Calculate properties
    rho = fluid1.getPhase(0).getDensity_Leachman()
    a = [2.09630e-1, -4.55274e-1, 1.43602e-1, -3.35325e-2, 2.76981e-3]
    b = [-0.1870, 2.4871, 3.7151, -11.0972, 9.0965, -3.8292, 0.5166]
    c = [0, 6.43449673, 4.56334068e-2, 2.32797868e-1, 9.58326120e-1, 1.27941189e-1, 3.63576595e-1]
    Tc = 33.145         #[K] (Source: NIST)
    rho_sc = 90.909090909      #[kg/m^3] 
    Tr = T / Tc
    M = 2.01588         #[g/mol] molar mass

    rho_r = rho / rho_sc

    sigma = 0.297       #[nm] scaling parameter
    epsilon_kb = 30.41  #[K] scaling parameter

    Tstar = T * 1/(epsilon_kb)
    sstar = 0       #Creating an sstar object
    for i in range(5):
        sstar += a[i]*(np.log(Tstar))**i
    Sstar = np.exp(sstar)

    Bstar_eta = 0       #Creating Bstar_eta object
    for i in range(7):
        Bstar_eta += b[i]*(Tstar)**(-i)

    B_eta = Bstar_eta * sigma**3

    eta_0 = (0.021357 * (M*T)**0.5) / (sigma**2 *Sstar)
    eta_1 = B_eta * eta_0

    eta = eta_0 + eta_1*rho + c[1]*rho_r**2 * np.exp(c[2]*Tr + c[3]/Tr + (c[4]*rho_r**2)/(c[5]+Tr) + c[6]*rho_r**6)
    return eta      #[µPa*s]


def F_theory(T, P):
    # Check if the temperature and pressure are within reasonable limits
    if T <= 0 or P <= 0:
        raise ValueError(f"Invalid temperature or pressure: T={T}, P={P}")
    fluid1 = fluid("srk")
    fluid1.addComponent("hydrogen", 1)
    fluid1.setTemperature(T, "K") 
    fluid1.setPressure(P, "MPa")
    TPflash(fluid1)
    fluid1.getPhase(0).getPhysicalProperties().setViscosityModel("friction theory")
    fluid1.initProperties()     # Calculate properties
    eta = fluid1.getViscosity("Pas") * 10**6  # [µPa*s]
    return eta


def PFCT(T, P):
    # Check if the temperature and pressure are within reasonable limits
    if T <= 0 or P <= 0:
        raise ValueError(f"Invalid temperature or pressure: T={T}, P={P}")
    fluid1 = fluid("srk")
    fluid1.addComponent("hydrogen", 1)
    fluid1.setTemperature(T, "K") 
    fluid1.setPressure(P, "MPa")
    TPflash(fluid1)
    fluid1.getPhase(0).getPhysicalProperties().setViscosityModel("PFCT")
    fluid1.initProperties()     # Calculate properties
    eta = fluid1.getViscosity("Pas") * 10**6  # [µPa*s]
    return eta


def LBC(T, P):
    # Check if the temperature and pressure are within reasonable limits
    if T <= 0 or P <= 0:
        raise ValueError(f"Invalid temperature or pressure: T={T}, P={P}")
    fluid1 = fluid("srk")
    fluid1.addComponent("hydrogen", 1)
    fluid1.setTemperature(T, "K") 
    fluid1.setPressure(P, "MPa")
    TPflash(fluid1)
    fluid1.getPhase(0).getPhysicalProperties().setViscosityModel("LBC")
    fluid1.initProperties()     # Calculate properties
    eta = fluid1.getViscosity("Pas") * 10**6  # [µPa*s]
    return eta


#print("muzny:", muzny_hydrogen(300, 20))


#MODIFY MUZNY HERE:
def Muzny_mod(T, P):
    #Create a fluid to get the hydrogen density
    fluid1 = fluid("srk")
    fluid1.addComponent("hydrogen", 1.0)
    fluid1.setTemperature(T, "K")
    fluid1.setPressure(P, "MPa")
    TPflash(fluid1)
    fluid1.initProperties()     # Calculate properties
    rho = fluid1.getPhase(0).getDensity_Leachman()
    a = [2.09630e-1, -4.55274e-1, 1.43602e-1, -3.35325e-2, 2.76981e-3]
    b = [-0.1870, 2.4871, 3.7151, -11.0972, 9.0965, -3.8292, 0.5166]
    c = [0, 6.43449673, 4.56334068e-2, 2.32797868e-1, 9.58326120e-1, 1.27941189e-1, 3.63576595e-1]
    Tc = 33.145         #[K] (Source: NIST)
    rho_sc = 90.909090909      #[kg/m^3] 
    Tr = T / Tc
    M = 2.01588         #[g/mol] molar mass

    rho_r = rho / rho_sc

    sigma = 0.297       #[nm] scaling parameter
    epsilon_kb = 30.41  #[K] scaling parameter

    Tstar = T * 1/(epsilon_kb)
    sstar = 0       #Creating an sstar object
    for i in range(5):
        sstar += a[i]*(np.log(Tstar))**i
    Sstar = np.exp(sstar)

    Bstar_eta = 0       #Creating Bstar_eta object
    for i in range(7):
        Bstar_eta += b[i]*(Tstar)**(-i)

    B_eta = Bstar_eta * sigma**3

    eta_0 = (0.021357 * (M*T)**0.5) / (sigma**2 *Sstar)
    eta_1 = B_eta * eta_0

    eta = eta_0 + eta_1*rho + c[1]*rho_r**2 * np.exp(c[2]*Tr + c[3]/Tr + (c[4]*rho_r**2)/(c[5]+Tr) + c[6]*rho_r**6)
    A = 0.002 * (P)*(405/T)**4.6 + 0.173 * (1 + 0.05/(1+np.exp(-0.5*(P-20))))
    B = 0.13 * (1 / (1 + np.exp(0.5*(T-405)))) + 0.1 * (1 / (1 + np.exp(-0.05*(T-390)))) * (1 + 2/(1+np.exp(0.9*(P-17))))
    C = 0.5 * (1- 1/(1 + np.exp(1.0*(T-355)))) * (1 - 1/(1 + np.exp(-0.3*(P-10)))) * (T/600)**2.5 - 0.15 * (1 - 1/(1 + np.exp(0.8*(P-19)))) * (420/T) * (1- 1/(1 + np.exp(1.0*(T-415))))
    return eta - A + B - C   #[µPa*s]

#SingleARDPlot([273.15, 298.15, 323.15, 348.15, 373.15, 398.15, 423.15, 473.15, 523.15], "Modified Muzny", "HYDROGEN_MuznyMod (-2, 2)")
#SingleARDPlot([273.15, 298.15, 323.15, 348.15, 373.15, 398.15, 423.15, 473.15, 523.15], "Muzny", "HYDROGEN_Muzny (-2, 2)")

#SingleARDPlot([273.15, 298.15, 323.15, 348.15, 373.15, 398.15, 423.15, 473.15, 523.15], "Modified Muzny", "HYDROGEN_MuznyMod_TESTTEST (-5, 5)")

#ViscPlot([273.15, 298.15, 323.15, 348.15], "Muzny", "MIXTURE_Pure_HYDROGEN_comparisonLOWTEMP")
#ViscPlot([273.15, 298.15, 323.15, 348.15, 373.15, 398.15, 423.15, 473.15, 523.15], "Muzny", "MIXTURE_Pure_HYDROGEN_comparisonHIGHTEMP")

#print("Muzny:", Muzny(273.15, 20))
#print("Muzny_mod:", Muzny_mod(273.15, 20))

ARDMultipleModelsPlot([273.15, 298.15, 323.15, 348.15, 373.15, 398.15, 423.15], ["LBC", "Muzny", "F-Theory", "PFCT"], "HYDROGEN_model_comparison")