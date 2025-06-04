import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from Hydrogen.HydrogenDataReader import viscData





def ARDViscIsotherm(targetTemp, viscModel):
    from Hydrogen.HydrogenModels import getModelViscosity
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




def ARDMultipleModelsPlot(Temperatures, viscosityModels, saveName):
    num_models = len(viscosityModels)  # Number of viscosity models
    rows = (num_models + 1) // 2  # Determine number of rows dynamically for two columns
    cols = 2  # Always two columns
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 9))
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Define discrete color map
    color_map = plt.cm.get_cmap('jet', len(Temperatures))
    
    # Calculate boundaries for discrete bins
    boundaries = np.convolve(Temperatures, [0.5, 0.5], 'valid')
    boundaries = np.concatenate((
        [Temperatures[0] - (boundaries[0] - Temperatures[0])],
        boundaries,
        [Temperatures[-1] + (Temperatures[-1] - boundaries[-1])]
    ))

    # Create BoundaryNorm with new boundaries
    norm = mpl.colors.BoundaryNorm(boundaries, ncolors=len(Temperatures))

    marker_legend = {}
    
    for idx, (ax, viscosityModel) in enumerate(zip(axes, viscosityModels)):
        for i, T in enumerate(Temperatures):
            dataMatrix = ARDViscIsotherm(T, viscosityModel) 
            
            for row in dataMatrix:
                pressures = row[1]
                ARDs = row[5]
                markerSymbol = row[6]
                label = f"{row[3]}"  # Only display the author
                
                # Get color for temperature and plot markers
                color = color_map(norm(T))
                facecolor = 'none' if markerSymbol not in ['2', '+', 'x'] else color  # Ensure visibility for unfilled markers
                sc = ax.scatter(pressures, ARDs, edgecolors=[color], facecolors=facecolor, marker=markerSymbol, label=f'{T:.2f} K', s=50)
                
                # Store unique marker types for legend
                if (markerSymbol, label) not in marker_legend:
                    # For the legend, ensure that markers are black and unfilled
                    if markerSymbol in ['2', '+', 'x']:
                        marker_legend[(markerSymbol, label)] = plt.Line2D([0], [0], marker=markerSymbol, color='black', markerfacecolor='none', linestyle='None', markersize=10)
                    else:
                        marker_legend[(markerSymbol, label)] = plt.Line2D([0], [0], marker=markerSymbol, color='black', markerfacecolor='none', linestyle='None', markersize=10)
        
        # Plot formatting for each subplot
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel("Pressure [MPa]", fontsize=18)
        ax.set_ylim(-30, 30.01)
        ax.set_xlim(-0.5, 32)
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_title(f'{viscosityModel}', fontsize=20)
    
    # Hide x-labels for the top row (the first row)
    for i in range(cols):
        axes[i].set_xlabel('')  # Remove x-label for the top row

    # Hide any unused subplots if odd number of models
    for i in range(len(viscosityModels), len(axes)):
        fig.delaxes(axes[i])

    fig.text(-0.02, 0.5, r'$100\cdot \left(\eta_{\mathrm{calc}} - \eta_{\mathrm{exp}}\right)/\eta_{\mathrm{exp}}$', 
             ha='center', va='center', rotation='vertical', fontsize=24)
    
    # Adjust layout to prevent colorbar overlap
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Create colorbar in separate axis to avoid overlapping
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # Positioning outside the main plot area
    sm = mpl.cm.ScalarMappable(cmap=color_map, norm=norm)
    sm.set_array([])  # Empty array for colorbar
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=Temperatures)  # Now ticks at the centers of the color bands
    cbar.ax.set_title('Temperature [K]', fontsize=12, pad=10)
    cbar.ax.set_yticklabels([f'{temp:.2f}' for temp in Temperatures])
    
    # Create legend for marker symbols below all plots
    legend_labels = [label for (_, label) in marker_legend.keys()]
    fig.legend(marker_legend.values(), legend_labels, loc='lower center', ncol=5, frameon=False, fontsize=18, bbox_to_anchor=(0.43, -0.1))
    
    fig.savefig(f"{saveName}.png", dpi=300, bbox_inches='tight')
    plt.clf()



def SingleARDPlot(Temperatures, viscosityModel, saveName):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define discrete color map and norm
    color_map = plt.cm.get_cmap('jet', len(Temperatures))
    norm = mpl.colors.BoundaryNorm(boundaries=np.append(Temperatures, Temperatures[-1] + 1), ncolors=len(Temperatures))

    marker_legend = {}

    for idx, T in enumerate(Temperatures):
        dataMatrix = ARDViscIsotherm(T, viscosityModel)
        for row in dataMatrix:
            pressures = row[1]
            ARDs = row[5]
            markerSymbol = row[6]
            label = f"{row[3]}"  # Only display the author

            # Get color for temperature and plot hollow markers
            color = color_map(norm(T))  # Unique color
            
            # If the marker is "2" or "+", make it filled
            if markerSymbol in ['2', '+', 'x']:
                facecolor = color  # Fill the marker with the temperature color
                edgecolor = 'black'  # Edge color for filled markers
            else:
                facecolor = 'none'  # Keep markers "2" and "+" unfilled
                edgecolor = color  # Use color from temperature for the edge color

            # Plot the scatter points
            sc = ax.scatter(pressures, ARDs, edgecolors=[edgecolor], facecolors=facecolor, marker=markerSymbol, label=f'{T:.2f} K', s = 40)

            # Store unique marker types for the legend with the author name
            if (markerSymbol, label) not in marker_legend:
                # For the legend, make markers black
                legend_handle = plt.Line2D([0], [0], marker=markerSymbol, color='black', markerfacecolor='none', linestyle='None', markersize=8)
                marker_legend[(markerSymbol, label)] = legend_handle

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
    cbar.ax.set_yticklabels([f'{temp:.2f}' for temp in Temperatures])  # Optional: Format tick labels

    # Plot formatting
    ax.set_title(f"{viscosityModel}", fontsize = 20)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel("Pressure [MPa]", fontsize=16)
    ax.set_ylabel(r'$100 \cdot \left(\eta_{\mathrm{calc}} - \eta_{\mathrm{exp}}\right)/\eta_{\mathrm{exp}}$', fontsize=18)
    ax.set_ylim(-2, 2.01)
    ax.set_xlim(-0.5, 32)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True)

    # Create legend for the author name
    legend_labels = [label for (_, label) in marker_legend.keys()]
    fig.legend(marker_legend.values(), legend_labels, loc='lower center', ncol=5, bbox_to_anchor=(0.48, -0.08), frameon=False, fontsize=10)

    # Save the plot
    fig.tight_layout()
    fig.savefig(f"{saveName}.png", dpi=300, bbox_inches='tight')
    plt.clf()


def ViscPlot(Temperatures, viscosityModel, saveName):
    plt.figure(figsize=(14,7))
    
    # Define colormap
    color_map = plt.cm.get_cmap('turbo')
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
    plt.ylim(8, 14)

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





#SingleARDPlot([273.15, 298.15, 323.15, 348.15, 373.15, 398.15, 423.15, 523.15], "Muzny", "HYDROGEN_Muzny_(-4, 4)")
