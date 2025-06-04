import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


from Methane_Hydrogen.HythaneDataReader import viscData


'''def ARDViscIsotherm(targetTemp, mixingRule, composition):
    from Methane_Hydrogen.MixtureModel import getMixingRule
    data_dict = viscData()
    dataList = []
    for key, data in data_dict.items():
        for i in range(len(data[0])):
            sjekkvariabel0 = data[0]
            sjekkvariabel1 = data[0][i]
            margin = abs(data[0][i] - targetTemp)
            if margin <= 0.5 and data[3].iloc[i] == composition:
                #data_row = (data[0].iloc[i], data[1].iloc[i], data[2].iloc[i], key)
                modelViscosity = getMixingRule(data[0].iloc[i], data[1].iloc[i], mixingRule, composition)
                ARD = 100 * (modelViscosity - data[2].iloc[i]) / data[2].iloc[i]
                data_row = [data[0].iloc[i], data[1].iloc[i], data[2].iloc[i], key, modelViscosity, ARD, data[3]]
                #data_row object: [Temperature, Pressure, Viscosity, Author, ModelViscosity, ARD, markerSymbol (for plot)]
                dataList.append(data_row)

    return dataList'''

def ARDViscIsotherm(targetTemp, mixingRule, composition):
    from Methane_Hydrogen.MixtureModel import getMixingRule
    data_dict = viscData()
    dataList = []
    
    for key, dataset_list in data_dict.items():  # ✅ Iterate over each dataset
        for data in dataset_list:  # ✅ Iterate over individual matrices
            T_values, P_values, eta_values, mole_fractions, marker = data  # ✅ Unpack the matrix

            for i in range(len(T_values)):  # ✅ Iterate through rows in the matrix
                margin = abs(T_values.iloc[i] - targetTemp)  # Ensure single value

                # Ensure correct comparison for mole fractions
                if margin <= 0.6 and np.all(np.isclose(mole_fractions, composition)):
                    modelViscosity = getMixingRule(T_values.iloc[i], P_values.iloc[i], mixingRule, composition)
                    ARD = 100 * (modelViscosity - eta_values.iloc[i]) / eta_values.iloc[i]
                    data_row = [T_values.iloc[i], P_values.iloc[i], eta_values.iloc[i], key, modelViscosity, ARD, marker]
                    dataList.append(data_row)

    return dataList



def SingleARDPlot(Temperatures, composition, mixingRules, saveName):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define a list of distinct colors for each temperature
    color_list = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink']
    color_map = {T: color_list[i % len(color_list)] for i, T in enumerate(Temperatures)}

    # Define a list of distinct markers for each mixing rule
    marker_list = ['o', 's', '^', 'D', 'p', '*', 'x', '+', 'v', '<', '>']
    marker_map = {rule: marker_list[i % len(marker_list)] for i, rule in enumerate(mixingRules)}

    marker_legend = {}

    for T in Temperatures:
        for mixingRule in mixingRules:
            dataMatrix = ARDViscIsotherm(T, mixingRule, composition)
            color = color_map[T]  # Assign a distinct color for each temperature
            markerSymbol = marker_map[mixingRule]  # Assign a distinct marker for each mixing rule
            label = f"{mixingRule} mixture model"  # Ensure correct legend label

            for row in dataMatrix:
                pressures = row[1]
                ARDs = row[5]

                # Plot data points with unfilled markers
                ax.scatter(pressures, ARDs, edgecolors=color, facecolors='none', marker=markerSymbol, label=label, s=40)

                # Store unique marker types for the legend
                if label not in marker_legend:  # Use full label
                    legend_handle = plt.Line2D([0], [0], marker=markerSymbol, color='black', linestyle='None',
                                               markerfacecolor='none', markeredgecolor='black', markersize=8, label=label)
                    marker_legend[label] = legend_handle  # Store with full label

    # Plot formatting
    ax.set_title(f"{composition[0]} H2 + {composition[1]} CH4", fontsize=20)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel("Pressure [MPa]", fontsize=16)
    ax.set_ylabel(r'$100 \cdot \left(\eta_{\mathrm{calc}} - \eta_{\mathrm{exp}}\right)/\eta_{\mathrm{exp}}$', fontsize=18)
    ax.set_ylim(-20, 20.01)
    ax.set_xlim(-0.5, 42)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)

    # Create legend for the mixing rule markers
    fig.legend(marker_legend.values(), marker_legend.keys(), loc='lower center', ncol=5, bbox_to_anchor=(0.48, -0.05),
               frameon=False, fontsize=10)

    # Save the plot
    fig.tight_layout()
    fig.savefig(f"{saveName}.png", dpi=300, bbox_inches='tight')
    plt.clf()


import matplotlib.patches as mpatches

def GroupedARDPlot(Temperatures, compositions, mixingRules, saveName):
    num_compositions = len(compositions)
    num_cols = 2  # Set number of columns
    num_rows = (num_compositions + 1) // num_cols  # Calculate number of rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 6 * num_rows))
    axes = axes.flatten() if num_compositions > 1 else [axes]

    # Define colors and markers
    color_list = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink']
    color_map = {T: color_list[i % len(color_list)] for i, T in enumerate(Temperatures)}

    marker_list = ['o', 's','^', 'x', '+', 'D', 'p', '*', 'v', '<', '>']
    marker_map = {rule: marker_list[i % len(marker_list)] for i, rule in enumerate(mixingRules)}

    # Define colors for each mixing rule/viscosity model
    colors = {
        "PFCT": "black",
        "PFCTMod": "red",
        "LBC": "blue",
        "Linear": "green",
        "LBCMod_methane": "orange",
    }

    legend_labels = {}

    for idx, composition in enumerate(compositions):
        ax = axes[idx]
        for T in Temperatures:
            for mixingRule in mixingRules:
                dataMatrix = ARDViscIsotherm(T, mixingRule, composition)
                markerSymbol = marker_map[mixingRule]

                for row in dataMatrix:
                    pressures = row[1]
                    ARDs = row[5]
                    markerSymbol = row[6]
                    facecolor = colors[mixingRule] if markerSymbol in ['x', '2', '+'] else 'none'
                    facecolor_black = "black" if markerSymbol in ['x', '2', '+'] else 'none'
                    label = row[3]

                    # Plot data points with unfilled markers or filled markers for 'x' and '+'
                    if label not in legend_labels:
                        ax.scatter(pressures, ARDs, edgecolors=colors[mixingRule], facecolors=facecolor, marker=markerSymbol, label=label, s=80)

                        legend_labels[label] = ax.scatter([], [], label=label, marker=markerSymbol, edgecolors="black", facecolors=facecolor_black, s=80)
                    else:
                        ax.scatter(pressures, ARDs, edgecolors=colors[mixingRule], facecolors=facecolor, marker=markerSymbol, label=label, s=80)
                    
        if composition[0] == 0 and composition[1] == 1:
            ax.set_title(f"Pure CH4", fontsize=20)
        elif composition[0] == 1 and composition[1] == 0:
            ax.set_title(f"Pure H2", fontsize=20)
        else:
            ax.set_title(f"{composition[0]} H2 + {composition[1]} CH4", fontsize=20)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_ylim(-5, 5.01)
        ax.set_xlim(-0.5, 42)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True)

        # Set x-label only for bottom row
        if idx >= len(compositions) - 2:  
            axes[idx].set_xlabel("Pressure [MPa]", fontsize=22)

    # Hide empty subplots if num_compositions is odd
    for i in range(num_compositions, num_rows * num_cols):
        fig.delaxes(axes[i])

    fig.legend(legend_labels.values(), legend_labels.keys(), loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.06),
               frameon=False, fontsize=24)
    fig.text(-0.02, 0.5, r'$100\cdot \left(\eta_{\mathrm{calc}} - \eta_{\mathrm{exp}}\right)/\eta_{\mathrm{exp}}$', 
             ha='center', va='center', rotation='vertical', fontsize=32)
    
    # Manually create legend entries for viscosity model colors
    color_legend_patches = [
        mpatches.Patch(color='black', label='PFCT'),
        mpatches.Patch(color='red', label='PFCT$_{mod}$'),
        mpatches.Patch(color='blue', label='LBC'),
        mpatches.Patch(color='green', label='Linear mixing rule')
    ]

    # Add second legend for colors
    fig.legend(handles=color_legend_patches, loc='upper center', ncol=4, 
            bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=17)

    fig.tight_layout(rect=[0, 0.1, 1, 0.95])
    fig.savefig(f"{saveName}.png", dpi=300, bbox_inches='tight')
    plt.clf()



def viscosityPlot(temperatures, compositions, saveName):
    num_plots = len(compositions)
    num_cols = 2  # Set number of columns for subplots
    num_rows = (num_plots + 1) // num_cols  # Calculate number of rows for subplots

    # Create subplots, one for each composition
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 6 * num_rows))
    axes = axes.flatten() if num_plots > 1 else [axes]

    # Get the viscosity data
    data_dict = viscData()
    
    # Define the colormap
    color_map = plt.cm.get_cmap('jet')  # Use the full colormap ('jet' in this case)
    
    # Normalize the colormap to span the full color range
    norm = mpl.colors.Normalize(vmin=min(temperatures)-4, vmax=max(temperatures))  # Normalize across the entire range of the colormap

    # Set up the colorbar on the right of the entire plot group
    sm = mpl.cm.ScalarMappable(cmap=color_map, norm=norm)
    sm.set_array([])  # empty array for colorbar

    # Create space for the colorbar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Position the colorbar on the right side of the plot

    # Add the colorbar to the plot
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', cax=cbar_ax)
    cbar.ax.set_title('Temperature [K]', fontsize=13, pad=10)  # Title above colorbar
    cbar.ax.set_yticklabels([f'{T}' for T in temperatures], fontsize=14)  # Show ticks for the full color range

    plot_index = 0
    marker_legend = {}
    
    for comp in compositions:
        ax = axes[plot_index]
        
        # Iterate over each dataset
        for dataset_name, dataset in data_dict.items():
            for entry in dataset:
                temp_data, pressure_data, viscosity_data, composition_data, marker = entry
                
                # Find and plot data points for matching composition and temperature range
                for temp in temperatures:
                    if np.allclose(composition_data, comp, atol=0.005) and np.any(np.abs(temp_data.values - temp) <= 0.6):
                        temp_mask = np.abs(temp_data.values - temp) <= 0.35
                        
                        # Get the color for this temperature based on the full colormap
                        color = color_map(norm(temp))

                        # If the marker is '+' (invisible when unfilled), use a filled marker
                        if marker == '+':
                            sc = ax.scatter(pressure_data[temp_mask], viscosity_data[temp_mask], 
                                            marker=marker, facecolors=color, edgecolors=color, linewidth=1.5, s=50)
                        else:
                            # For all other markers, keep them unfilled with colored edges
                            sc = ax.scatter(pressure_data[temp_mask], viscosity_data[temp_mask], 
                                            marker=marker, facecolors='none', edgecolors=color, linewidth=1.5, s=50)
                            
                        # Store unique marker types for legend
                        if (marker, dataset_name) not in marker_legend:
                            # For the legend, ensure that markers are black and unfilled
                            if marker in ['2', '+', 'x']:
                                marker_legend[(marker, dataset_name)] = plt.Line2D([0], [0], marker=marker, color='black', markerfacecolor='none', linestyle='None', markersize=10)
                            else:
                                marker_legend[(marker, dataset_name)] = plt.Line2D([0], [0], marker=marker, color='black', markerfacecolor='none', linestyle='None', markersize=10)

        ax.set_title(f"xH$_2$={comp[0]}", fontsize=20)
        ax.set_xlabel("Pressure [MPa]", fontsize=18)
        ax.set_ylabel("Viscosity [µPa$\cdot$s]", fontsize=18)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlim(-0.5, 52)
        ax.grid(True)
        
        # Hide x-axis labels for all but the bottom row
        if plot_index < num_plots - num_cols:
            ax.set_xlabel('')

        # Hide y-axis labels for all but the left column
        if plot_index % num_cols != 0:
            ax.set_ylabel('')
        
        plot_index += 1
    
    # Adjust layout to ensure colorbar is not inside the plot area
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the colorbar on the right

    # Add the legend below the entire plot group
    legend_labels = [label for (_, label) in marker_legend.keys()]
    fig.legend(marker_legend.values(), legend_labels, loc='lower center', ncol=7, frameon=False, fontsize=17, bbox_to_anchor=(0.43, -0.03))

    # Save the figure
    plt.savefig(f"{saveName}.png", dpi=300, bbox_inches='tight')
    plt.clf()

