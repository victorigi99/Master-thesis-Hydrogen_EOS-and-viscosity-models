import sys
import os

# Add ViscosityCode to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from KTAModel import ARDViscIsotherm
from Helium.heliumDataReader import viscData
from Helium.KTAModel import getModelViscosity


def modelPerformance(T):
    #dataList object: [Temperature, Pressure, Viscosity, Author, ModelViscosity, ARD, markerSymbol (for plot)]
    viscosityModels = ["KTA", "KTA_tweak"]
    averageARDList_low = []
    averageARDList_high = []
    maxARDList_low = []
    maxARDList_high = []

    for viscosityModel in viscosityModels:
        dataList = ARDViscIsotherm(T, viscosityModel)
        if not dataList:
            print(f"No data found at {T}K")
            break
        lowPressures = [row for row in dataList if row[1] <= 20]
        highPressures = [row for row in dataList if row[1] > 20]

        if lowPressures:
            averageARD_20 = np.mean([abs(row[5]) for row in lowPressures])
            maxARD_20 = max([abs(row[5]) for row in lowPressures])
            minPressure_20 = min([row[1] for row in lowPressures])
            maxPressure_20 = max([row[1] for row in lowPressures])
            dataPoints_20 = len(lowPressures)

            averageARDList_low.append(averageARD_20)
            maxARDList_low.append(maxARD_20)

        if highPressures:
            averageARD_40 = np.mean([abs(row[5]) for row in highPressures])
            maxARD_40 = max([abs(row[5]) for row in highPressures])
            minPressure_40 = min([row[1] for row in highPressures])
            maxPressure_40 = max([row[1] for row in highPressures])
            dataPoints_40 = len(highPressures)

            averageARDList_high.append(averageARD_40)
            maxARDList_high.append(maxARD_40)

            


    print("\n Low pressures:")
    print(f"Temperature: {T:.2f}K || Pressure range: {minPressure_20:.2f}MPa - {maxPressure_20:.2f}MPa || Number of datapoints: {dataPoints_20}")
    print(f"Average ARD KTA: {averageARDList_low[0]:.2f}% | Average ARD KTA_tweak: {averageARDList_low[1]:.2f}%")
    print(f"Max ARD KTA: {maxARDList_low[0]:.2f}% | Max ARD KTA_tweak: {maxARDList_low[1]:.2f}%")

    if highPressures:
        print("\n High pressures:")
        print(f"Temperature: {T:.2f}K | Pressure range: {minPressure_40:.2f}MPa - {maxPressure_40:.2f}MPa || Number of datapoints: {dataPoints_40}")
        print(f"Average ARD KTA: {averageARDList_high[0]:.2f}% | Average ARD KTA_tweak: {averageARDList_high[1]:.2f}%")
        print(f"Max ARD KTA: {maxARDList_high[0]:.2f}% | Max ARD KTA_tweak: {maxARDList_high[1]:.2f}%")

    if not highPressures:
        print("\n### No high pressures above 20MPa ###")


#modelPerformance(473.15)

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

modelPerformanceALL("KTA")
modelPerformanceALL("KTA_mod")
modelPerformanceALL("KTA_tweak")

