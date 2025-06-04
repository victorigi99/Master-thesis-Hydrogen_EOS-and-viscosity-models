import sys
import os

# Add ViscosityCode to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import timeit
import numpy as np
from Methane.ModelTesting import getModelViscosity




def timetesting(viscmodel):
    temperatures = np.linspace(273.15, 573.15, 50)
    pressures = np.linspace(1, 50, 50)
    timelist = []
    iterations = 20
    for i in range(iterations):
        t0 = timeit.default_timer()
        for T in temperatures:
            for P in pressures:
                viscosity = getModelViscosity(T, P, viscmodel)
        t1 = timeit.default_timer()
        timelist.append(t1 - t0)
        print(f"Iteration {i+1}: {t1 - t0:.2f} s")
    print(f"Average time for {viscmodel} over {iterations} iterations: {np.mean(timelist):.2f} s")

timetesting("LBC_mod")