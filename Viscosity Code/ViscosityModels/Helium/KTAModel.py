import sys
import os

# Add ViscosityCode to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Helium.heliumDataReader import viscData

#from https://www.sciencedirect.com/science/article/pii/S0149197024004670#bib28:
def KTA(T):
    eta = 3.674*10**(-7) * T**(0.7)    #[Pa*s]
    return eta

#from https://www.sciencedirect.com/science/article/pii/S0149197024004670#bib28:
def KTA_mod(T):
    eta = 3.817*10**(-7) * T**(0.6938)   #[Pa*s]
    return eta

### DETTE ER DEN NYE VISKOSITETSMODELLEN FOR HELIUM ###
def KTA_tweak(T, P):
    P_crit = 0.22832   #[MPa]   (Source: NIST)
    #eta = 1e-7*(3.817 * T**(0.6938) + (P**((2 - T/300)**(5.05)) / (T*P_crit)) - (np.exp(-(T-325)**2 / 1000))*(T**((2-300/T)**(2) - 1)) + (np.exp(-(T-325)**2 / 1000))*(P/25)**(2.7)) * 10**6   #[µPa*s]
    if T > 500:
        A = 0
    else:
        A = (2 - T/300)**(5.05)
    B = (2-300/T)**(2) - 1
    C = (1-0.3/(1+np.e**(-0.5*(T-450))))/(1+np.e**(-0.5*(T-377))) - 1.5/(1+np.e**(-0.5*(T-572)))
    #C = 1/(1+np.e**(-0.5*(T-377)))
    #C = (1/(1+np.e**(-0.5*(T-377)))) * (1/(1+np.e**(-0.5*(T-500))))
    eta = 1e-7 * (3.817 * T**(0.6938) + P**A/(T*P_crit) + (np.exp(-(T-325)**2 / 1000))*((P/25)**(2.7) - T**B ) - C)   #[Pa*s]
    return eta

def getModelViscosity(T, P, viscosityModel):
    if viscosityModel == "KTA":
        eta = KTA(T) * 10**6    #[µPa*s]
    elif viscosityModel == "KTA_mod":
        eta = KTA_mod(T) * 10**6    #[µPa*s]
    elif viscosityModel == "KTA_tweak":
        eta = KTA_tweak(T, P) * 10**6    #[µPa*s]
    else:
        raise ValueError("Invalid viscosity model")
    
    return eta

#print("KTA_tweak =", getModelViscosity(303.15, 7, "KTA_tweak"))

def ARDViscIsotherm(targetTemp, viscModel):
    data_dict = viscData()
    dataList = []
    for key, data in data_dict.items():
        for i in range(len(data[0])):
            margin = abs(data[0][i] - targetTemp)
            if margin <= 0.5:
                #data_row = (data[0].iloc[i], data[1].iloc[i], data[2].iloc[i], key)
                modelViscosity = getModelViscosity(data[0].iloc[i], data[1].iloc[i], viscModel)
                ARD = 100 * (modelViscosity - data[2].iloc[i]) / data[2].iloc[i]
                data_row = [data[0].iloc[i], data[1].iloc[i], data[2].iloc[i], key, modelViscosity, ARD, data[3]]
                #data_row object: [Temperature, Pressure, Viscosity, Author, ModelViscosity, ARD, markerSymbol (for plot)]
                dataList.append(data_row)

    return dataList

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')



def ViscPlot(Temperatures, saveName):
    plt.figure(figsize=(14,7))
    
    # Define colormap
    color_map = plt.cm.get_cmap('jet')
    norm = mpl.colors.Normalize(vmin=min(Temperatures), vmax=max(Temperatures))

    marker_legend = {}  # To store unique markers and authors for the legend
    
    for T in Temperatures:
        dataMatrix = ARDViscIsotherm(T, "KTA")  # The viscosity model is arbitrary
        first_point = True
        color = color_map(norm(T))  # Assign a color based on temperature

        for row in dataMatrix:
            pressures = row[1]
            viscosities = row[2]
            markers = row[6]  # Marker symbol for the plot
            author = row[3]  # Author name for the legend

            # Define fill style for different markers
            if markers in ["+", "x", "2"]:  # Line-based markers should be filled
                scatter_kwargs = {"color": color}  # Ensures visibility
            else:  # Circle ("o") and square ("s") should be unfilled
                scatter_kwargs = {"facecolors": "none", "edgecolors": color}

            # Plot the points
            if first_point:  # Only add label for the first point
                plt.scatter(pressures, viscosities, marker=markers, label=author, s=50, **scatter_kwargs)
                first_point = False
            else:
                plt.scatter(pressures, viscosities, marker=markers, s=50, **scatter_kwargs)

            # Store the marker for legend (unique marker + author combination)
            if (markers, author) not in marker_legend:
                if markers in ["+", "x", "2"]:  # Filled markers
                    marker_legend[(markers, author)] = plt.Line2D([0], [0], marker=markers, color='black', markerfacecolor=color, linestyle='None', markersize=10)
                else:  # Unfilled markers
                    marker_legend[(markers, author)] = plt.Line2D([0], [0], marker=markers, color='black', markerfacecolor='none', linestyle='None', markersize=10)

    # Tick params
    plt.tick_params(axis='both', labelsize=14)  # Increase axis numbers' font size
    plt.xlabel("Pressure [MPa]", fontsize=18)
    plt.ylabel(r'Viscosity [µPa$\cdot$s]', fontsize=18)

    # Modify the legend to have black text
    legend_labels = [f"{author}" for _, author in marker_legend.keys()]
    fig_legend = plt.legend(marker_legend.values(), legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=14)
    
    # Customize legend appearance
    for text in fig_legend.get_texts():
        text.set_color('black')  # Set legend text to black
    
    for handle in fig_legend.legend_handles:
        handle.set_color('black')  # Set the edge color of the markers to black
        handle.set_gapcolor('none')  # Ensure the markers are unfilled (if applicable)

    plt.grid()

    # Create colorbar
    sm = mpl.cm.ScalarMappable(cmap=color_map, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical')
    
    # Set explicit ticks and labels
    cbar_ticks = Temperatures  # Use the temperature values as ticks
    cbar.set_ticks(cbar_ticks)  # Ensure the colorbar has ticks at the min and max temperature

    # Explicitly set the labels corresponding to those ticks
    cbar.set_ticklabels([f'{T}' for T in Temperatures], fontsize=12)  # Set the labels to be the temperature values

    # Adjust layout to leave space for the label above the colorbar
    plt.subplots_adjust(right=0.85)  # Add extra space on the right for the label

    # Manually place the label above the colorbar using fig.text()
    cbar_pos = cbar.ax.get_position()  # Get the current position of the colorbar
    plt.figtext(cbar_pos.x0 + cbar_pos.width / 2 - 0.035, cbar_pos.y1 + 0.02, "Temperature [K]", ha='center', fontsize=13)

    # Save the figure
    plt.savefig(f"{saveName}.png", dpi=500, bbox_inches='tight')
    plt.clf()

#ViscPlot([273.15, 293.15, 298.15, 323.15, 373.15, 423.15, 473.15], "All_Helium_Viscosities")


    


def ARDPlot(Temperatures, viscModels, saveName):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    axs = axs.flatten()

    # Define colors for each viscosity model
    colors = {
        "KTA": "black",
        "KTA_mod": "red",
        "KTA_tweak": "blue"
    }

    # Dictionary to keep track of labels added to the legend
    legend_labels = {}

    for i, T in enumerate(Temperatures):
        for viscModel in viscModels:
            dataList = ARDViscIsotherm(T, viscModel)
            for row in dataList:
                pressures = row[1]
                ARDs = row[5]
                markerSymbol = row[6]
                facecolor = colors[viscModel] if markerSymbol in ['x', '2', '+'] else 'none'
                label = f"{viscModel} ({row[3]})"
                if label not in legend_labels:
                    axs[i].scatter(pressures, ARDs, label=label, marker=markerSymbol, edgecolors=colors[viscModel], facecolors=facecolor)
                    legend_labels[label] = axs[i].scatter([], [], label=label, marker=markerSymbol, edgecolors=colors[viscModel], facecolors=facecolor)
                else:
                    axs[i].scatter(pressures, ARDs, marker=markerSymbol, edgecolors=colors[viscModel], facecolors=facecolor)
            axs[i].axhline(y=0, color='black', linestyle='-', linewidth=0.7)
        axs[i].set_ylim(-2, 2)  # Set y-axis limits
        if i >= 2:  # Only set x-label for the bottom plots
            axs[i].set_xlabel("Pressure [MPa]", fontsize=16)
        axs[i].set_title(f"{T}K", fontsize=16)
        axs[i].grid()
        axs[i].tick_params(axis='both', labelsize=11.5)  # Increase axis numbers' font size
    
    fig.text(-0.02, 0.5, r'$100\cdot \left(\eta_{\mathrm{calc}} - \eta_{\mathrm{exp}}\right)/\eta_{\mathrm{exp}}$', 
             ha='center', va='center', rotation='vertical', fontsize=19)    
    #fig.suptitle("ARD Plots", fontsize=16, y=0.93)     #Title for the whole plotgroup

    # Create a single legend from the plot elements
    handles, labels = [], []
    for handle, label in legend_labels.items():
        handles.append(label)
        labels.append(handle)
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=11)

    fig.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(f"{saveName}.png", dpi=500, bbox_inches='tight')
    plt.clf()



#ARDPlot([273.15, 298.15, 373.15, 473.15], ["KTA", "KTA_mod"], "KTA and KTA_mod")

#ViscPlot([273.15, 293.15, 298.15, 323.15], "ViscPlotLowTemp")
#ViscPlot([373.15, 423.15, 473.15], "ViscPlotHighTemp")



import matplotlib.patches as mpatches

def ARDPlot2(Temperatures, viscModels, saveName):
    num_plots = len(Temperatures)
    rows = num_plots // 2 + num_plots % 2  # Determine rows dynamically
    cols = 2  # Always two columns
    
    fig, axs = plt.subplots(rows, cols, figsize=(16, 16))
    axs = axs.flatten()  # Flatten for easy indexing

    # Define colors for each viscosity model
    colors = {
        "KTA": "black",
        "KTA_mod": "red",
        "KTA_tweak": "blue"
    }

    # Dictionary to keep track of labels added to the legend
    legend_labels = {}

    for i, T in enumerate(Temperatures):
        for viscModel in viscModels:
            dataList = ARDViscIsotherm(T, viscModel)
            for row in dataList:
                pressures = row[1]
                ARDs = row[5]
                markerSymbol = row[6]
                facecolor = colors[viscModel] if markerSymbol in ['x', '2', '+'] else 'none'
                label = row[3]
                if label not in legend_labels:
                    axs[i].scatter(pressures, ARDs, label=label, marker=markerSymbol, 
                                   edgecolors=colors[viscModel], facecolors=facecolor, s=70)
                    legend_labels[label] = axs[i].scatter([], [], label=label, marker=markerSymbol, 
                                                          edgecolors=colors[viscModel], facecolors=facecolor, s=70)
                else:
                    axs[i].scatter(pressures, ARDs, marker=markerSymbol, 
                                   edgecolors=colors[viscModel], facecolors=facecolor, s=70)
            axs[i].axhline(y=0, color='black', linestyle='-', linewidth=0.7)
        
        axs[i].set_ylim(-2, 2)  # Set y-axis limits
        axs[i].set_title(f"{T}K", fontsize=20)
        axs[i].grid(True, which='both')
        axs[i].set_yticks(np.arange(-2, 2.1, 0.5))  #adds finer grid lines
        axs[i].tick_params(axis='both', labelsize=14)

        # Set x-label only for bottom row
        if i >= len(Temperatures) - 2:  
            axs[i].set_xlabel("Pressure [MPa]", fontsize=20)

    fig.text(-0.02, 0.5, r'$100\cdot \left(\eta_{\mathrm{calc}} - \eta_{\mathrm{exp}}\right)/\eta_{\mathrm{exp}}$', 
             ha='center', va='center', rotation='vertical', fontsize=26) 
    # Remove any unused subplot (if any)
    for j in range(len(Temperatures), len(axs)):
        fig.delaxes(axs[j])

    # Create a single legend from the plot elements
    handles, labels = [], []
    for handle, label in legend_labels.items():
        handles.append(label)
        labels.append(handle)
    
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=20)

    # Manually create legend entries for viscosity model colors
    color_legend_patches = [
        mpatches.Patch(color='black', label='KTA'),
        mpatches.Patch(color='red', label='KTA$_{mod}$'),
        mpatches.Patch(color='blue', label='KTA$_{new}$')
    ]

    # Add second legend for colors
    fig.legend(handles=color_legend_patches, loc='upper center', ncol=3, 
            bbox_to_anchor=(0.5, 0.99), frameon=False, fontsize=15)

    fig.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(f"{saveName}.png", dpi=500, bbox_inches='tight')
    plt.clf()



#ARDPlot2([273.15, 293.15, 298.15, 323.15, 373.15, 423.15, 473.15], ["KTA","KTA_mod", "KTA_tweak"], "HELIUM_ARD")