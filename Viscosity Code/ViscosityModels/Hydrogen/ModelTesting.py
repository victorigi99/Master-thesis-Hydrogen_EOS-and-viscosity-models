import sys
import os

# Add ViscosityCode to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from plotting import ARDViscIsotherm
from Hydrogen.HydrogenDataReader import viscData
from Hydrogen.HydrogenModels import getModelViscosity


def modelPerformance(T):
    #dataList object: [Temperature, Pressure, Viscosity, Author, ModelViscosity, ARD, markerSymbol (for plot)]
    viscosityModels = ["Muzny", "Muzny_mod"]
    averageARDList = []
    maxARDList = []

    

    for viscosityModel in viscosityModels:
        #data_row object: [Temperature, Pressure, Viscosity, Author, ModelViscosity, ARD, markerSymbol (for plot)]
        dataMatrix = ARDViscIsotherm(T, viscosityModel)
        if not dataMatrix:
            print(f"No data found at {T}K")
            break

        averageARD = np.mean([abs(row[5]) for row in dataMatrix])
        maxARD = max([abs(row[5]) for row in dataMatrix])
        minPressure = min([row[1] for row in dataMatrix])
        maxPressure = max([row[1] for row in dataMatrix])
        dataPoints = len(dataMatrix)

        averageARDList.append(averageARD)
        maxARDList.append(maxARD)

    print("\nALL pressures:")
    print(f"Temperature: {T:.2f}K || Pressure range: {minPressure:.2f}MPa - {maxPressure:.2f}MPa || Number of datapoints: {dataPoints}")
    print(f"Average ARD Muzny: {averageARDList[0]:.2f}% | Average ARD Muzny_mod: {averageARDList[1]:.2f}%")
    print(f"Max ARD Muzny: {maxARDList[0]:.2f}% | Max ARD Muzny_mod: {maxARDList[1]:.2f}%")


#modelPerformance(523.15)

def modelPerformanceALL(viscModel):
    #dataList object: [Temperature, Pressure, Viscosity, Author, ModelViscosity, ARD, markerSymbol (for plot)]
    ARD_list = []

    data_dict = viscData()
    dataList = []
    for key, data in data_dict.items():
        for i in range(len(data[0])):
            #data_row = (data[0].iloc[i], data[1].iloc[i], data[2].iloc[i], key)
            modelViscosity = getModelViscosity(data[0].iloc[i], data[1].iloc[i], viscModel)
            ARD = 100 * (modelViscosity - data[2].iloc[i]) / data[2].iloc[i]
            data_row = [data[0].iloc[i], data[1].iloc[i], data[2].iloc[i], key, modelViscosity, ARD, data[3]]
            #data_row object: [Temperature, Pressure, Viscosity, Author, ModelViscosity, ARD, markerSymbol (for plot)]
            ARD_list.append(abs(ARD))
            dataList.append(data_row)

    avgARD = np.mean(ARD_list)
    maxARD = np.max(ARD_list)
    print(f"Average ARD for {viscModel}: {avgARD:.2f}%")
    print(f"Max ARD for {viscModel}: {maxARD:.2f}%")
    print("Number of points:", len(ARD_list))

modelPerformanceALL("Muzny")
modelPerformanceALL("Muzny_mod")