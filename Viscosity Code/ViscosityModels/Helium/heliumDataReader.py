import pandas as pd

Seibt = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Helium\Seibt.csv", skiprows=5)
Kestin_1977 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode/Experimental Data/Helium/Kestin 1977.csv", skiprows=5)
Kestin_1972 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode/Experimental Data/Helium/Kestin 1972.csv", skiprows=5)
Kestin_1971 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode/Experimental Data/Helium/Kestin 1971.csv", skiprows=5)
Kestin_1963 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode/Experimental Data/Helium/Kestin 1963.csv", skiprows=5)
Kestin_1959 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode/Experimental Data/Helium/Kestin 1959.csv", skiprows=5)
Kao = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode/Experimental Data/Helium/Kao.csv", skiprows=5)
Kalelkar = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode/Experimental Data/Helium/Kalelkar.csv", skiprows=5)
Flynn = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode/Experimental Data/Helium/Flynn.csv", skiprows=5)


def viscData(dataset=None):
    data_dict = {
        "Seibt et al." : [Seibt['T [K] '], Seibt['P [Mpa]'], Seibt['eta_293.15 [µPa*s]'], 'o'],

        #"Kestin 1977" : [Kestin_1977['T [K] '], Kestin_1977['P [Mpa]'], Kestin_1977['eta [µPa*s]'], 's' ],

        "Kestin et al. 1972" : [Kestin_1972['T [K] '], Kestin_1972['P [Mpa]'], Kestin_1972['eta [µPa*s]'], '^'],

        "Kestin et al. 1971" : [Kestin_1971['T [K] '], Kestin_1971['P [Mpa]'], Kestin_1971['eta [µPa*s]'], '2'],

        "Kestin and Whitelaw 1963" : [Kestin_1963['T [K] '], Kestin_1963['P [Mpa]'], Kestin_1963['eta [µPa*s]'], 'p'],

        "Kestin and Leidenfrost 1959" : [Kestin_1959['T [K] '], Kestin_1959['P [Mpa]'], Kestin_1959['eta [µPa*s]'], '+'],

        "Kao and Kobayashi" : [Kao['T [K] '], Kao['P [Mpa]'], Kao['eta [µPa*s]'], 'D'],

        "Kalelkar and Kestin" : [Kalelkar['T [K] '], Kalelkar['P [Mpa]'], Kalelkar['eta [µPa*s]'], 'd'],

        "Flynn et al." : [Flynn['T [K] '], Flynn['P [Mpa]'], Flynn['eta [µPa*s]'], 'x']

    }

    if dataset:
        if dataset in data_dict:
            return data_dict[dataset]
        else:
            raise ValueError(f"Dataset '{dataset}' not found. Please check the dataset name.")
    else:
        return data_dict

