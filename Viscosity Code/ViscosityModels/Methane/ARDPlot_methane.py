import sys
import os

# Add ViscosityCode to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


methane_data = pd.read_csv(r"C:\Users\akser\OneDrive - NTNU\MASTEROPPGAVE\ViscosityCode\Experimental Data\Methane\Methane - NIST DATA.csv", skiprows=1)
methane_T, methane_P, methane_eta =  methane_data['T [K] '], methane_data['P [MPa]'], methane_data['eta [µPa*s]']



def GroupedARDPlot(Temperatures, viscModels, saveName):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))  # Create subplots, max 4 temperatures
    axs = axs.flatten()

    # Define colors using function names
    colors = {
        "LBC": "black",
        "F_theory": "red",
        "methaneTest1": "green",
        "GEP": "blue"
    }
    
    # Dictionary to keep track of labels added to the legend
    legend_handles = {}

    for idx, T in enumerate(Temperatures):
        for viscModel in viscModels:
            model_name = viscModel.__name__  # Convert function to string
            ARDs = []
            pressures = []

            for i in range(len(methane_T)):
                if T == methane_T[i]:
                    ARD = 100 * (viscModel(T, methane_P[i]) - methane_eta[i]) / methane_eta[i]
                    ARDs.append(ARD)
                    pressures.append(methane_P[i])
                    
                    markerSymbol = 'o'  # Define a default marker
                    facecolor = colors[model_name] if markerSymbol in ['p', 'x', '2', '+'] else 'none'
                    label = model_name

                    # Add label only once per model
                    if label not in legend_handles:
                        scatter = axs[idx].scatter(pressures[-1], ARDs[-1], label=label, marker=markerSymbol, 
                                                   edgecolors=colors[model_name], facecolors=facecolor)
                        legend_handles[label] = scatter  # Store scatter handle
                    else:
                        axs[idx].scatter(pressures[-1], ARDs[-1], marker=markerSymbol, 
                                         edgecolors=colors[model_name], facecolors=facecolor)

            axs[idx].axhline(y=0, color='black', linestyle='-', linewidth=0.7)
        
        axs[idx].set_ylim(-6, 6.1)
        if idx >= 2:  # Set x-label for bottom plots only
            axs[idx].set_xlabel("Pressure [MPa]", fontsize=14)
        axs[idx].set_title(f"{T}K", fontsize=14)
        axs[idx].grid()

    # Add y-label for the entire figure
    fig.text(0, 0.5, r'$100 \left(\eta_{\mathrm{model}} - \eta_{\mathrm{experimental}}\right)/\eta_{\mathrm{experimental}}$', 
             ha='center', va='center', rotation='vertical', fontsize=18)

    # Extract scatter handles and labels **correctly**
    handles = list(legend_handles.values())  # Get scatter plot objects
    labels = list(legend_handles.keys())  # Get model names (strings)

    # Create a single legend at the bottom
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.05), frameon=False, fontsize=14)

    fig.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(f"{saveName}.png", dpi=300, bbox_inches='tight')
    plt.clf()



import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def SingleARDPlot(Temperatures, viscosityModel, saveName):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define discrete color map and norm
    color_map = plt.cm.get_cmap('jet', len(Temperatures))
    norm = mpl.colors.BoundaryNorm(boundaries=np.append(Temperatures, Temperatures[-1] + 1), ncolors=len(Temperatures))

    for idx, T in enumerate(Temperatures):
        ARDs = []
        pressures = []

        for i in range(len(methane_T)):
            if T == methane_T[i]:
                ARD = 100 * (viscosityModel(T, methane_P[i]) - methane_eta[i]) / methane_eta[i]
                ARDs.append(ARD)
                pressures.append(methane_P[i])

        # Get color for temperature and plot hollow markers
        color = color_map(norm(T))  # Unique color
        ax.scatter(pressures, ARDs, edgecolors=[color], facecolors='none', label=f'{T:.2f} K')

    # Create colorbar linked to the temperature colors
    # Calculate boundaries for discrete bins
    boundaries = np.convolve(Temperatures, [0.5, 0.5], 'valid')
    boundaries = np.concatenate((
        [Temperatures[0] - (boundaries[0] - Temperatures[0])],
        boundaries,
        [Temperatures[-1] + (Temperatures[-1] - boundaries[-1])]
    ))

    # Create BoundaryNorm
    norm = mpl.colors.BoundaryNorm(boundaries, ncolors=len(Temperatures))

    # Colorbar (ticks at Temperatures, now aligned to color centers)
    sm = mpl.cm.ScalarMappable(cmap=color_map, norm=norm)
    sm.set_array([])  # Empty array for colorbar
    cbar = fig.colorbar(sm, ax=ax, ticks=Temperatures)
    cbar.ax.set_title('Temperature [K]', fontsize=12, pad=10)
    #cbar.set_label('Temperature [K]', fontsize=10, rotation=0, loc='center', labelpad=20)
    cbar.ax.set_yticklabels([f'{temp:.2f}' for temp in Temperatures])  # Optional: Format tick labels


    # Plot formatting
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel("Pressure [MPa]", fontsize=14)
    ax.set_ylabel(r'$100 \left(\eta_{\mathrm{calc}} - \eta_{\mathrm{exp}}\right)/\eta_{\mathrm{exp}}$', fontsize=16)
    ax.set_yticks(np.arange(-4, 4.1, 0.5))
    ax.set_ylim(-2, 2.01)
    #ax.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.3), frameon=False, fontsize=12)
    #ax.set_title("LBC$_{modified}$", fontsize=20)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{saveName}.png", dpi=300, bbox_inches='tight')
    plt.clf()



from Methane.MethaneDataReader import viscData

def ARDViscIsotherm(targetTemp, viscModel):
    from Methane.ModelTesting import getModelViscosity
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

def ViscPlot(Temperatures, viscosityModel, saveName):
    plt.figure(figsize=(14,7))
    
    # Define colormap
    color_map = plt.cm.get_cmap('jet')
    norm = mpl.colors.Normalize(vmin=min(Temperatures), vmax=max(Temperatures))

    marker_legend = {}  # To store unique markers and authors for the legend
    
    for T in Temperatures:
        dataMatrix = ARDViscIsotherm(T, viscosityModel)  # The viscosity model is arbitrary
        first_point = True
        color = color_map(norm(T))  # Assign a color based on temperature

        for row in dataMatrix:
            pressures = row[1]
            viscosities = row[2]
            markers = row[6]  # Marker symbol for the plot
            author = row[3]  # Author name for the legend

            # Define fill style for different markers
            if markers in ["+", "x", "2", "4"]:  # Line-based markers should be filled
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
                if markers in ["+", "x", "2", "4"]:  # Filled markers
                    marker_legend[(markers, author)] = plt.Line2D([0], [0], marker=markers, color='black', markerfacecolor=color, linestyle='None', markersize=10)
                else:  # Unfilled markers
                    marker_legend[(markers, author)] = plt.Line2D([0], [0], marker=markers, color='black', markerfacecolor='none', linestyle='None', markersize=10)

    # Tick params
    plt.tick_params(axis='both', labelsize=14)  # Increase axis numbers' font size
    plt.xlabel("Pressure [MPa]", fontsize=18)
    plt.ylabel(r'Viscosity [µPa$\cdot$s]', fontsize=18)
    plt.xlim(-0.5 , 41)
    plt.ylim(9, 30)

    # Modify the legend to have black text
    legend_labels = [f"{author}" for _, author in marker_legend.keys()]
    fig_legend = plt.legend(marker_legend.values(), legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=14)
    
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