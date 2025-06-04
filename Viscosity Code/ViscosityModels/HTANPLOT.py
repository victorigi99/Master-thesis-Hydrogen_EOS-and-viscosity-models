import matplotlib.pyplot as plt
import numpy as np

T = np.linspace(-1200, 1200, 2401)
x = np.linspace(800, 1200, 2401)
y = [1] * len(x)  # Create a list of ones with the same length as x
T_f = 90.69
dT = T - T_f
HTAN = (np.exp(dT) - np.exp(-dT)) / (np.exp(dT) + np.exp(-dT))

fig = plt.figure(figsize=(14, 6)) 
plt.plot(T, HTAN, color="blue", label="HTAN before adjustment")
plt.plot(x, y, linestyle='--', color="green", label="after adjustment")
plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
plt.xlabel("$\Delta$T [K]", fontsize=18)
plt.ylabel("HTAN", fontsize=18)
plt.xlim(-1200, 1200)
plt.ylim(-2, 2)
plt.legend(fontsize=12)

# Add more ticks on the x-axis
x_ticks = np.arange(-1200, 1201, 200)  # Set ticks every 200 units
plt.xticks(x_ticks, fontsize=12)  # Apply the ticks and optionally adjust font size

plt.grid()
fig.savefig("HTANPLOT.png", dpi=300, bbox_inches='tight')
plt.clf()