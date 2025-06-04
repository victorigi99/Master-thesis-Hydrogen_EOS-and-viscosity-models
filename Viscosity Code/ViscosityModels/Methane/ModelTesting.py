from PureMethaneModel import LBCMod_methane, LBC
from ARDPlot_methane import methane_data
import pandas as pd
import numpy as np


methane_T, methane_P, methane_eta =  methane_data['T [K] '], methane_data['P [MPa]'], methane_data['eta [µPa*s]']

def getModelViscosity(T, P, viscosityModel):
    if viscosityModel == "LBC":
        eta = LBC(T, P)   #[µPa*s]
    elif viscosityModel == "LBC_mod":
        eta = LBCMod_methane(T, P)    #[µPa*s]
    else:
        raise ValueError("Invalid viscosity model")
    
    return eta

def ARDViscIsotherm_methane(targetTemp, viscModel):
    dataList = []
    for i in range(len(methane_T)):
        margin = abs(methane_T[i] - targetTemp)
        if margin <= 0.5:
            modelViscosity = getModelViscosity(methane_T[i], methane_P[i], viscModel)
            ARD = 100 * (modelViscosity - methane_eta[i]) / methane_eta[i]
            dataList.append([methane_T[i], methane_P[i], methane_eta[i], "NIST", modelViscosity, ARD])

    return dataList

def modelPerformance(T):
    #dataList object: [Temperature, Pressure, Viscosity, Author, ModelViscosity, ARD, markerSymbol (for plot)]
    viscosityModels = ["LBC", "LBC_mod"]
    averageARDList = []
    maxARDList = []

    

    for viscosityModel in viscosityModels:
        dataList = ARDViscIsotherm_methane(T, viscosityModel)
        if not dataList:
            print(f"No data found at {T}K")
            break

        averageARD = np.mean([abs(row[5]) for row in dataList])
        maxARD = max([abs(row[5]) for row in dataList])
        minPressure = min([row[1] for row in dataList])
        maxPressure = max([row[1] for row in dataList])
        dataPoints = len(dataList)

        averageARDList.append(averageARD)
        maxARDList.append(maxARD)

    print("\nALL pressures:")
    print(f"Temperature: {T:.2f}K || Pressure range: {minPressure:.2f}MPa - {maxPressure:.2f}MPa || Number of datapoints: {dataPoints}")
    print(f"Average ARD LBC: {averageARDList[0]:.2f}% | Average ARD LBC_mod: {averageARDList[1]:.2f}%")
    print(f"Max ARD LBC: {maxARDList[0]:.2f}% | Max ARD LBC_mod: {maxARDList[1]:.2f}%")


#modelPerformance(523.15)

def modelPerformanceALL(viscModel):
    #dataList object: [Temperature, Pressure, Viscosity, Author, ModelViscosity, ARD, markerSymbol (for plot)]
    ARD_list = []

    for i in range(len(methane_T)):
        modelViscosity = getModelViscosity(methane_T[i], methane_P[i], viscModel)
        ARD = 100 * (modelViscosity - methane_eta[i]) / methane_eta[i]
        ARD_list.append(abs(ARD))

    avgARD = np.mean(ARD_list)
    maxARD = np.max(ARD_list)
    print(f"Average ARD for {viscModel}: {avgARD:.2f}%")
    print(f"Max ARD for {viscModel}: {maxARD:.2f}%")
    print("Number of points:", len(ARD_list))

modelPerformanceALL("LBC")
modelPerformanceALL("LBC_mod")
