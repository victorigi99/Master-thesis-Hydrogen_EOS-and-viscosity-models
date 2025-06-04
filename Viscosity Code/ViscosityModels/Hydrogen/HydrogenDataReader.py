import pandas as pd

Michels = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hydrogen\Michels.csv", skiprows=5)
Gracki = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hydrogen\Gracki.csv", skiprows=5)
Chuang = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hydrogen\Chuang.csv", skiprows=5)
Barua = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hydrogen\Barua.csv", skiprows=5)
Golubev = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hydrogen\Golubev.csv", skiprows=5)
Kestin = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hydrogen\Kestin.csv", skiprows=5) 

Betken = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hydrogen\Betken et al. (1,2).csv", skiprows=5)
Owuna = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hydrogen\Owuna et al (0,3).csv", skiprows=5)
Chuang_100 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hydrogen\Chuang et al. (1,6).csv", skiprows=5)


def viscData(dataset=None):
    data_dict = {
        "Michels et al." : [Michels['T [K] '], Michels['P [Mpa]'], Michels['eta [µPa*s]'], 'o'],

        "Gracki et al." : [Gracki['T [K] '], Gracki['P [Mpa]'], Gracki['eta [µPa*s]'], '^'],

        "Chuang et al." : [Chuang['T [K] '], Chuang['P [Mpa]'], Chuang['eta [µPa*s]'], '2'],

        "Barua et al." : [Barua['T [K] '], Barua['P [Mpa]'], Barua['eta [µPa*s]'], '+'],

        "Golubev and Petrov" : [Golubev['T [K] '], Golubev['P [Mpa]'], Golubev['eta [µPa*s]'], 's'],

        "Kestin and Wang" : [Kestin['T [K] '], Kestin['P [Mpa]'], Kestin['eta [µPa*s]'], 'x'],

        "Betken et al.*" : [Betken['T [K] '], Betken['P [Mpa]'], Betken['eta [µPa*s]'], '4'],

        "Owuna et al.*" : [Owuna['T [K] '], Owuna['P [Mpa]'], Owuna['eta [µPa*s]'], 'D'],

        "Chuang et al.*" : [Chuang_100['T [K] '], Chuang_100['P [Mpa]'], Chuang_100['eta [µPa*s]'], '*']

    }

    if dataset:
        if dataset in data_dict:
            return data_dict[dataset]
        else:
            raise ValueError(f"Dataset '{dataset}' not found. Please check the dataset name.")
    else:
        return data_dict
