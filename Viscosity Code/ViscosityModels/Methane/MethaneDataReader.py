import pandas as pd

NIST = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Methane\Methane - NIST DATA.csv", skiprows=1)


Owuna_methane = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hythane\Owuna et al (4,3).csv", skiprows=5)
Chuang_methane = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hythane\Chuang et al. (6,6).csv", skiprows=5)
#Betken does not contain pure methane data

def viscData(dataset=None):
    data_dict = {
        "NIST" : [NIST['T [K] '], NIST['P [MPa]'], NIST['eta [µPa*s]'], 'o'],

        "Owuna et al.*" : [Owuna_methane['T [K] '], Owuna_methane['P [Mpa]'], Owuna_methane['eta [µPa*s]'], 'D'],

        "Chuang et al.*" : [Chuang_methane['T [K] '], Chuang_methane['P [Mpa]'], Chuang_methane['eta [µPa*s]'], '*']

    }

    if dataset:
        if dataset in data_dict:
            return data_dict[dataset]
        else:
            raise ValueError(f"Dataset '{dataset}' not found. Please check the dataset name.")
    else:
        return data_dict
