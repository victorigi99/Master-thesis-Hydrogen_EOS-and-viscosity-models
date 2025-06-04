import sys
import os

# Add ViscosityCode to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from Methane.ARDPlot_methane import ViscPlot


ViscPlot([273.15, 298.15, 323.15], "LBC", "MIXTURE_Pure_METHANE_comparison")