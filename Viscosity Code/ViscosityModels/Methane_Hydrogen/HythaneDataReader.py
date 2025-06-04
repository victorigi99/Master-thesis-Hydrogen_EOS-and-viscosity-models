import pandas as pd

Chuang_100 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hythane\Chuang et al. (1,6).csv", skiprows=5)
Chuang_79 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hythane\Chuang et al. (2,6).csv", skiprows=5)
Chuang_53 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hythane\Chuang et al. (3,6).csv", skiprows=5)
Chuang_34 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hythane\Chuang et al. (4,6).csv", skiprows=5)
Chuang_19 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hythane\Chuang et al. (5,6).csv", skiprows=5)
Chuang_0 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hythane\Chuang et al. (6,6).csv", skiprows=5)

Owuna_10 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hythane\Owuna et al. (1,3).csv", skiprows=5)
Owuna_20 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hythane\Owuna et al. (2,3).csv", skiprows=5)
Owuna_50 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hythane\Owuna et al. (3,3).csv", skiprows=5)

Betken_100 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hythane\Betken et al. (1,2).csv", skiprows=5)
Betken_90 = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Hythane\Betken et al. (2,2).csv", skiprows=5)

methane_data = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Methane\Methane - NIST DATA.csv", skiprows=1)

#   viscData object: [temperature [K], pressure [MPa], viscosity [µPa*s], [x_H2, x_CH4], plot marker]

def viscData(dataset=None):
    data_dict = {
        "Chuang et al." : [[Chuang_100['T [K] '], Chuang_100['P [Mpa]'], Chuang_100['eta [µPa*s]'], [1, 0], "o"],
                           [Chuang_79['T [K] '], Chuang_79['P [Mpa]'], Chuang_79['eta [µPa*s]'], [0.787, 0.213], "o"],
                           [Chuang_53['T [K] '], Chuang_53['P [Mpa]'], Chuang_53['eta [µPa*s]'], [0.5337, 0.4663], "o"],
                           [Chuang_34['T [K] '], Chuang_34['P [Mpa]'], Chuang_34['eta [µPa*s]'], [0.3375, 0.6625], "o"],
                           [Chuang_19['T [K] '], Chuang_19['P [Mpa]'], Chuang_19['eta [µPa*s]'], [0.1942, 0.8058], "o"],
                           [Chuang_0['T [K] '], Chuang_0['P [Mpa]'], Chuang_0['eta [µPa*s]'], [0, 1], "o"]],

        "Owuna et al." : [[Owuna_10['T [K] '], Owuna_10['P [Mpa]'], Owuna_10['eta [µPa*s]'], [0.1, 0.9], "x"],
                          [Owuna_20['T [K] '], Owuna_20['P [Mpa]'], Owuna_20['eta [µPa*s]'], [0.2, 0.8], "x"],
                          [Owuna_50['T [K] '], Owuna_50['P [Mpa]'], Owuna_50['eta [µPa*s]'], [0.5, 0.5], "x"]],

        "Betken et al." : [[Betken_100['T [K] '], Betken_100['P [Mpa]'], Betken_100['eta [µPa*s]'], [1.0, 0.0], "^"],
                           [Betken_90['T [K] '], Betken_90['P [Mpa]'], Betken_90['eta [µPa*s]'], [0.896, 0.104], "^"]],
                        
        "NIST Methane Data" : [[methane_data['T [K] '], methane_data['P [MPa]'], methane_data['eta [µPa*s]'], [0, 1], "s"]]
    }

    if dataset:
        if dataset in data_dict:
            return data_dict[dataset]
        else:
            raise ValueError(f"Dataset '{dataset}' not found. Please check the dataset name.")
    else:
        return data_dict

