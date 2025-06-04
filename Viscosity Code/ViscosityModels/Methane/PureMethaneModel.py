import sys
import os

# Add ViscosityCode to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import numpy as np
from Methane.ARDPlot_methane import GroupedARDPlot, SingleARDPlot
from neqsim.thermo import TPflash, fluid


#Source: https://www.sciencedirect.com/science/article/pii/S1003995309601092
def methane_visc1(T, P):
    Tc = 190.564        #[K] (Source: NIST)
    Pc = 4.5992      #[MPa] (Source: NIST)
    
    #Coefficients:
    A1 = -2.25711259E-2
    A2 = -1.31338399E-4
    A3 = 3.44353097E-6
    A4 = -4.69476607E-8
    A5 = 2.23030860E-2
    A6 = -5.56421194E-3
    A7 = 2.90880717E-5
    A8 = -1.90511457
    A9 = 1.14082882
    A10 = -2.25890087E-1

    Tr = T / Tc
    Pr = P / Pc

    numerator = A1 + A2*Pr + A3*Pr**2 + A4*Pr**3 + A5*Tr + A6*Tr**2
    denominator = 1 + A7*Pr + A8*Tr + A9*Tr**2 + A10*Tr**3

    eta = numerator / denominator
    return eta * 1000     # [µPa*s]




def LBC(T, P):
    # Check if the temperature and pressure are within reasonable limits
    if T <= 0 or P <= 0:
        raise ValueError(f"Invalid temperature or pressure: T={T}, P={P}")
    fluid1 = fluid("srk")
    fluid1.addComponent("methane", 1)
    fluid1.setTemperature(T, "K") 
    fluid1.setPressure(P, "MPa")
    TPflash(fluid1)
    fluid1.getPhase(0).getPhysicalProperties().setViscosityModel("LBC")
    fluid1.initProperties()     # Calculate properties
    eta = fluid1.getViscosity("Pas") * 10**6  # [µPa*s]
    return eta

def F_theory(T, P):
    # Check if the temperature and pressure are within reasonable limits
    if T <= 0 or P <= 0:
        raise ValueError(f"Invalid temperature or pressure: T={T}, P={P}")
    fluid1 = fluid("srk")
    fluid1.addComponent("methane", 1)
    fluid1.setTemperature(T, "K") 
    fluid1.setPressure(P, "MPa")
    TPflash(fluid1)
    fluid1.getPhase(0).getPhysicalProperties().setViscosityModel("friction theory")
    fluid1.initProperties()     # Calculate properties
    eta = fluid1.getViscosity("Pas") * 10**6  # [µPa*s]
    return eta



#High-precision viscosity measurements on methane (Vogel)
#Tror ikke denne er helt riktig...
def Vogel_methane(T, P):
    MW_methane = 16.0428      #[g/mol]
    Tc = 190.564
    Pc = 4.5992     #[MPa]
    rho_c = 162.66  #[kg/m^3]

    #Creating a fluid object to get density from GERG2008
    fluid1 = fluid("srk")
    fluid1.addComponent("methane", 1.0)
    fluid1.setTemperature(T, "K")
    fluid1.setPressure(P, "MPa")
    TPflash(fluid1)
    fluid1.initProperties()             #Calculate thermodynamic- and fluid properties
    rho = fluid1.getPhase(0).getDensity_GERG2008()    #[kg/m^3]

    #Constants for methane
    a = [0.215309028, -0.46256942, 0.051313823, 0.030320660, -0.0070047029]
    b = [-0.19572881E+02, 0.21973999E+03, -0.10153226E+04, 0.247101251E+04, -0.33751717E+04, 0.24916597E+04, -0.78726086E+03, 0.14085455E+02, -0.34664158E+00]
    e = [[0, 0, -0.302256904347E+01, 0.311150846518E+01, 0.672852409238E+01, -0.109330775541E+01],
         [0, 0, 0.176965130175E+02, -0.215685107769E+02, 0.102387524315E+02, -0.120030749419E+01]]
    f1 = 0.211009923406E+02
    g1 = 0.310860501398E+01
    omega = 0.37333    #[nm]
    sum = 0
    for i in range(5):
        sum += a[i]*(np.log(T)**i)
    OMEGA = np.exp(sum)

    eta0 = (0.021357*(MW_methane*T)**0.5) / (omega**2 * OMEGA)    #[uPa*s] 

    B_n = 0
    T_star = T/160.78
    for i in range(0, 7):
        B_n += 0.6022137*omega**3 * (b[i]*T_star**(-0.25*i) + b[7]*T_star**(-2.5) + b[8]*T_star**(-5.5))
    eta1 = B_n * eta0 

    delta = rho / rho_c
    tau = T / Tc

    deltaEta_h = 0
    delta0 = g1   #mangler g_l, står ikke i artikkelen
    for i in range(0, 2):
        for j in range(2, 6):
            deltaEta_h += e[i][j] * delta**i / tau**j +f1*(delta/(delta0-delta) - delta/delta0)

    eta = eta0 + eta1*11.636 + deltaEta_h
    return eta

#print(Vogel_methane(273.15, 20))



def GEP(T, P):
    #Creating a fluid object to get properties
    fluid1 = fluid("srk")
    fluid1.addComponent("methane", 1.0)
    fluid1.setTemperature(T, "K")
    fluid1.setPressure(P, "MPa")
    TPflash(fluid1)
    fluid1.initProperties()             #Calculate thermodynamic- and fluid properties
    rho = fluid1.getPhase(0).getDensity_GERG2008() * 0.001   #[g/cm3]
    T_pc = fluid1.getPhase(0).getPseudoCriticalTemperature()
    P_pc = fluid1.getPhase(0).getPseudoCriticalPressure() * 0.1  #[MPa]
    MW = fluid1.getPhase(0).getMolarMass() *1000   #[g/mol]

    T_pr = T / T_pc
    P_pr = P / P_pc

    A = np.exp(8.8081*rho) + rho*(-0.045254*P_pr**2 + 0.14255*MW - 0.14255*T_pr + 10.695)
    B = T_pr * np.sqrt(1.3569*MW - 2*P_pr - T_pr + 20.75) - 6.8507*rho*(P_pr + T_pr)*(rho - 0.83696)
    eta = (A + B) *10**-3
    return eta * 1000 #Convert to uPa*s

def GEP_mod(T, P):
    #Creating a fluid object to get properties
    fluid1 = fluid("srk")
    fluid1.addComponent("methane", 1.0)
    fluid1.setTemperature(T, "K")
    fluid1.setPressure(P, "MPa")
    TPflash(fluid1)
    fluid1.initProperties()             #Calculate thermodynamic- and fluid properties
    rho = fluid1.getPhase(0).getDensity_GERG2008() * 0.001   #[g/cm3]
    T_pc = fluid1.getPhase(0).getPseudoCriticalTemperature()
    P_pc = fluid1.getPhase(0).getPseudoCriticalPressure() * 0.1  #[MPa]
    MW = fluid1.getPhase(0).getMolarMass() * 1000   #[g/mol]

    T_pr = T / T_pc
    P_pr = P / P_pc

    A = np.exp(8.8081*rho) + rho*(-0.045254*P_pr**2 + 0.14255*MW - 0.14255*T_pr + 10.695)
    B = T_pr * np.sqrt(1.3569*MW - 2*P_pr - T_pr + 20.75) - 6.8507*rho*(P_pr + T_pr)*(rho - 0.83696)
    C = 0.1*((T/285)**3.5 - 1)*(1/(1 + np.exp(-0.05*(T-380)))) #+ 0.001*(1/(1 + np.exp(-0.4*(T-400))))*(10**(-5)*(T-400)**5 + 1)*((P/16)**3 - 1)   #+ 0.05*(1/(1 + np.exp(-0.2*(T-400))))*((P/16)**4 - 1)
    eta = (A + B - C) *10**-3
    return eta * 1000 #Convert to uPa*s

#print("Viscosity GEP =", GEP_methane(273.15, 20))

#GroupedARDPlot([283.15, 343.15, 373.15, 523.15], [LBC, F_theory, GEP], "methane_viscosity_TEST")
#SingleARDPlot([283.15, 298.15, 313.15, 343.15, 373.15, 388.15, 418.15, 433.15, 478.15, 523.15], GEP_mod, "methane_viscosity_GEP")

def LBCMod_methane(T, P):
    # Check if the temperature and pressure are within reasonable limits
    if T <= 0 or P <= 0:
        raise ValueError(f"Invalid temperature or pressure: T={T}, P={P}")
    fluid1 = fluid("srk")
    fluid1.addComponent("methane", 1)
    fluid1.setTemperature(T, "K") 
    fluid1.setPressure(P, "MPa")
    TPflash(fluid1)
    fluid1.getPhase(0).getPhysicalProperties().setViscosityModel("LBC")
    fluid1.initProperties()     # Calculate properties

    if T >= 345:
        term_A = (1.1*(T/345 - 1)**1.2)
    else:
        term_A = 0.64*(T/345 - 1) * (1 - 0.4*np.exp(-(T-298.15)**2 / 100)/(1+np.exp(-(P-21)))) 

    
    A =  10**-6 * ( term_A + 0.27 * np.exp(-(T-430)**2 / 9000) * np.exp(-(P-21)**2 / 35)/(1+np.exp(-(P-15))) )

    #A = np.real( 10**-6 * ( 1.1*(T/345 - 1)**1.2 + 0.27 * np.exp(-(T-430)**2 / 9000) * np.exp(-(P-21)**2 / 35)/(1+np.exp(-(P-15))) ) )
    B = 10**(-8) * (  1.2 * (300/T)**3 * P**1.2/(1+np.exp(-0.6*(P-20))) + 30 * (1 - T/400) * np.exp(-(P-15)**2 / 20) + 13*np.exp(-(P-12.8)**2 / 7) )
    C =  10**(-8) * ( 1/(1+np.exp((P-15)))*( 2*P * np.exp(-(T-375)**2 / 10000) + 8.0 * (P/4 - 1)) + 10.0 * np.exp(-(T-260)**2 / 300)/(1+np.exp(0.9*(P-12))) )
    D = 10**(-6) * ( 0.62 * (1 - T/430) * 1/(1+np.exp(-0.5*(P-26))) + 0.306*np.exp(-(T-270)**2/50)/(1+np.exp(-(P-25))) - (265.2/T) * np.exp(-(T-270)**2/160)/(1+np.exp(-0.5*(P-25))) )
    eta = fluid1.getViscosity("Pas") + A - B + C - D # [Pa*s]
    eta *= 10**6  # [µPa*s]
    return eta


#SingleARDPlot([268.15, 273.15, 298.15, 313.15, 343.15, 373.15, 388.15, 418.15, 433.15, 478.15], LBCMod_methane, "methane_viscosity_LBC_MethaneMod")

#SingleARDPlot([268.15, 273.15, 298.15, 313.15, 343.15, 373.15, 388.15, 418.15, 433.15, 478.15], LBC, "methane_viscosity_LBC")
#print("Her kommer LBC_mod(268.15, 10):", LBCMod_methane(268.15, 10))

#ViscPlot([273.15, 298.15, 323.15], "LBC", "MIXTURE_Pure_METHANE_comparison")


def LBCMod_methane_mix(T, P, composition):
    # Check if the temperature and pressure are within reasonable limits
    if T <= 0 or P <= 0:
        raise ValueError(f"Invalid temperature or pressure: T={T}, P={P}")
    fluid1 = fluid("srk")
    fluid1.addComponent("methane", composition[0])
    fluid1.addComponent("hydrogen", composition[1])
    fluid1.setTemperature(T, "K") 
    fluid1.setPressure(P, "MPa")
    TPflash(fluid1)
    fluid1.getPhase(0).getPhysicalProperties().setViscosityModel("LBC")
    fluid1.initProperties()     # Calculate properties

    if T >= 345:
        term_A = (1.1*(T/345 - 1)**1.2)
    else:
        term_A = 0.64*(T/345 - 1) * (1 - 0.4*np.exp(-(T-298.15)**2 / 100)/(1+np.exp(-(P-21)))) 

    
    A =  10**-6 * ( term_A + 0.27 * np.exp(-(T-430)**2 / 9000) * np.exp(-(P-21)**2 / 35)/(1+np.exp(-(P-15))) )

    #A = np.real( 10**-6 * ( 1.1*(T/345 - 1)**1.2 + 0.27 * np.exp(-(T-430)**2 / 9000) * np.exp(-(P-21)**2 / 35)/(1+np.exp(-(P-15))) ) )
    B = 10**(-8) * (  1.2 * (300/T)**3 * P**1.2/(1+np.exp(-0.6*(P-20))) + 30 * (1 - T/400) * np.exp(-(P-15)**2 / 20) + 13*np.exp(-(P-12.8)**2 / 7) )
    C =  10**(-8) * ( 1/(1+np.exp((P-15)))*( 2*P * np.exp(-(T-375)**2 / 10000) + 8.0 * (P/4 - 1)) + 10.0 * np.exp(-(T-260)**2 / 300)/(1+np.exp(0.9*(P-12))) )
    D = 10**(-6) * ( 0.62 * (1 - T/430) * 1/(1+np.exp(-0.5*(P-26))) + 0.306*np.exp(-(T-270)**2/50)/(1+np.exp(-(P-25))) - (265.2/T) * np.exp(-(T-270)**2/160)/(1+np.exp(-0.5*(P-25))) )
    eta = fluid1.getViscosity("Pas") + A - B + C - D # [Pa*s]
    eta *= 10**6  # [µPa*s]
    return eta

#print("LBCMod_methane(273.15, 20):", LBCMod_methane(273.15, 20))