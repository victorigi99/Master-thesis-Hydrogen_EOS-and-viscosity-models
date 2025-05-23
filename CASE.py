#!/usr/bin/env python3
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from neqsim.thermo import fluid, TPflash
from neqsim.process import stream, runProcess, compressor, heater, clearProcess
from neqsim import jneqsim


# Lower heating values [kJ/kg]
LHV = {
    'methane': 50020.0,
    'hydrogen': 119950.0
}

def energy_demand(flow_rate_msm3_day, EOS, p_out, T_cool):
    if EOS == 'GERG2008' or EOS == 'Leachman':
        f = jneqsim.thermo.system.SystemGERG2008Eos()
    else:
        f = fluid(EOS)

    f.setTemperature(T_cool, 'C')
    f.setPressure(p_out, 'bara')
    f.addComponent('methane', 1.0)
    if EOS != 'GERG2008' or EOS != 'Leachman': f.setMixingRule('classic')
    f.init(0)
    f.initProperties()
    TPflash(f)

    s = stream("stream", f)
    s.setFlowRate(flow_rate_msm3_day, 'MSm3/day')
    s.setTemperature(T_cool, 'C')
    s.setPressure(p_out, 'bara')
    s.run()

    m_dot = s.getFlowRate('kg/sec')
    return m_dot * LHV['methane'] * 1e-6  # GW


def compr_pipe_system(s, case_name, results, comp_specs: dict, pipe_specs: dict):
    # Unpack compressor specs
    clearProcess()
    eta_poly = comp_specs['eta_poly']
    p_mid    = comp_specs['p_mid']
    p_out    = comp_specs['p_out']
    T_cool   = comp_specs['T_cool']

    SOS1 = s.getFluid().getSoundSpeed()
    vel1 = s.getFlowRate('kg/sec') / (s.getFluid().getDensity() * np.pi * (pipe_specs['diameter']/2)**2)
    

    # Stage 1 compressor
    comp1 = compressor("compressor1", s)
    comp1.setPolytropicEfficiency(eta_poly)
    comp1.setUsePolytropicCalc(True)
    comp1.setPolytropicMethod('schultz')
    comp1.setOutletPressure(p_mid, 'bara')
    comp1.run()

    density_comp1_inlet = comp1.getInletStream().getFluid().getDensity()
    density_comp1_outlet = comp1.getOutletStream().getFluid().getDensity()
    SOS_comp1_inlet = comp1.getInletStream().getFluid().getSoundSpeed()
    SOS_comp1_outlet = comp1.getOutletStream().getFluid().getSoundSpeed()
    #Mach_comp1_inlet = comp1.getInletStream().getSuperficialVelocity() / SOS_comp1_inlet
    #Mach_comp1_outlet = comp1.getOutletStream().getSuperficialVelocity() / SOS_comp1_outlet
    # Intercooler
    cool1 = heater("cooler1", comp1.getOutletStream())
    cool1.setOutTemperature(T_cool, 'C')
    cool1.run()

    # Stage 2 compressor
    comp2 = compressor("compressor2", cool1.getOutletStream())
    comp2.setPolytropicEfficiency(eta_poly)
    comp2.setUsePolytropicCalc(True)
    comp2.setPolytropicMethod('schultz')
    comp2.setOutletPressure(p_out, 'bara')
    comp2.run()

    density_comp2_inlet = comp2.getInletStream().getFluid().getDensity()
    density_comp2_outlet = comp2.getOutletStream().getFluid().getDensity()
    SOS_comp2_inlet = comp2.getInletStream().getFluid().getSoundSpeed()
    SOS_comp2_outlet = comp2.getOutletStream().getFluid().getSoundSpeed()
    #Mach_comp2_inlet = comp2.getInletStream().getSuperficialVelocity() / SOS_comp2_inlet
    #Mach_comp2_outlet = comp2.getOutletStream().getSuperficialVelocity() / SOS_comp2_outlet

    # Aftercooler
    cool2 = heater("cooler2", comp2.getOutletStream())
    cool2.setOutTemperature(T_cool, 'C')
    cool2.run()

    # Unpack pipeline specs
    length     = pipe_specs['length']
    diameter   = pipe_specs['diameter']
    roughness  = pipe_specs['roughness']
    increments = pipe_specs['increments']
    isothermal = pipe_specs['isothermal']
    U          = pipe_specs['U']
    T_wall     = pipe_specs['T_wall']

    # Pipeline
    p = jneqsim.process.equipment.pipeline.PipeBeggsAndBrills('flow1', cool2.getOutletStream())
    p.setLength(length)
    p.setElevation(0.0)
    p.setDiameter(diameter)
    p.setPipeWallRoughness(roughness)
    p.setNumberOfIncrements(increments)
    p.setRunIsothermal(isothermal)
    p.setHeatTransferCoefficient(U)
    p.setConstantSurfaceTemperature(T_wall, 'C')
    p.run()


    fluid2 = s.getFluid()
    fluid2.setTemperature(T_cool, 'C')
    fluid2.setPressure(p_out, 'bara')
    fluid2.setMixingRule('classic')
    fluid2.init(0)
    fluid2.initProperties()
    TPflash(fluid2)

    '''

    lengths     = [x for x in p.getLengthProfile()]
    pressures   = [x for x in p.getPressureProfile()]
    temperatures_K = [t for t in p.getTemperatureProfile()]
    vel = [x for x in p.getGasSuperficialVelocityProfile()]
    dens = [x for x in p.getMixtureDensityProfile()]
    pipe_visc = [x for x in p.getMixtureViscosityProfile()]

# convert to °C
    temperatures_C = [t - 273.15 for t in temperatures_K]


    # — Plot pressure & temperature as before —
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(lengths, pressures, '-o')
    axes[0].set_ylabel('Pressure [bar]')
    axes[0].grid(True)

    axes[1].plot(lengths, temperatures_C, '-o')
    axes[1].set_ylabel('Temperature [°C]')
    axes[1].set_xlabel('Length [m]')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # — Now plot the three additional profiles —
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(lengths, vel, '-o')
    axes[0].set_ylabel('Gas superficial velocity [m/s]')
    axes[0].grid(True)

    axes[1].plot(lengths, dens, '-o')
    axes[1].set_ylabel('Mixture density [kg/m³]')
    axes[1].grid(True)

    axes[2].plot(lengths, pipe_visc, '-o')
    axes[2].set_ylabel('Mixture viscosity [cP]')
    axes[2].set_xlabel('Length [m]')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
'''

    # Collect metrics
    m_dot = s.getFlowRate('kg/sec')
    std_v = s.getFlowRate('MSm3/day')
    dp_m  = p.getPressureDrop() / length
    P1    = comp1.getPower('MW')
    P2    = comp2.getPower('MW')
    H1    = comp1.getPolytropicHead('kJ/kg')
    H2    = comp2.getPolytropicHead('kJ/kg')
    rho2  = cool2.getOutletStream().getFluid().getDensity()
    vel = m_dot / (rho2 * np.pi * (diameter/2)**2)
    Re = rho2 * vel * diameter / cool2.getOutletStream().getFluid().getPhase('gas').getPhysicalProperties().getViscosity()
    friction_factor = (1.0 / (-1.8 * np.log10(((roughness/diameter)/3.7)**1.11 + 6.9/Re)))**2
    density_pipe = p.getOutletStream().getFluid().getDensity()
    vel_pipe_out = m_dot / (density_pipe * np.pi * (diameter/2)**2)
    SOS_pipe_out = p.getOutletStream().getFluid().getSoundSpeed()
    MW = cool2.getOutletStream().getFluid().getPhase('gas').getMolarMass()

    print("Density: ", cool2.getOutletStream().getFluid().getDensity())
    print("Velocity: ", s.getFlowRate('kg/sec') / (cool2.getOutletStream().getFluid().getDensity() * np.pi * (diameter/2)**2))

    results.append({
        'Case':          case_name,
        'massFlow':      m_dot,
        'stdVolFlow':    std_v,
        'dp_per_m':      dp_m*1e3,
        'power1':        P1,
        'power2':        P2,
        'head1':         H1,
        'head2':         H2,
        'density':       rho2,
        'SOS1':          SOS1,
        'vel':          vel,
        'SOS1_out':      comp1.getOutletStream().getFluid().getSoundSpeed(),
        'SOS2':          cool1.getOutletStream().getFluid().getSoundSpeed(),
        'SOS2_out':      comp2.getOutletStream().getFluid().getSoundSpeed(),
        'Mach2_out':  vel / comp2.getOutletStream().getFluid().getSoundSpeed(),
        'viscosity':     cool2.getOutletStream().getFluid().getPhase('gas').getPhysicalProperties().getViscosity()*1e6,
        'T_out_pipe':    p.getOutletStream().getTemperature('C'),
        'p_out_pipe':    p.getOutletStream().getPressure('bara'),
        'T_out_comp1':   comp1.getOutletStream().getTemperature('C'),
        'T_out_comp2':   comp2.getOutletStream().getTemperature('C'),
        'Re':            Re,
        'friction_factor': friction_factor,
        'density_pipe':  density_pipe,
        'vel_pipe_out':  vel_pipe_out,
        'SOS_pipe_out':  SOS_pipe_out,
        'MW':            MW,
        'mach1':  vel1 / SOS1,
        'density_comp1_inlet': density_comp1_inlet,
        'density_comp1_outlet': density_comp1_outlet,
        'density_comp2_inlet': density_comp2_inlet,
        'density_comp2_outlet': density_comp2_outlet,
        'Comp1_density_ratio': density_comp1_outlet / density_comp1_inlet,
        'Comp2_density_ratio': density_comp2_outlet / density_comp2_inlet,
        'density_cool2': cool2.getOutletStream().getFluid().getDensity(),
        'SOS_comp1_inlet': SOS_comp1_inlet,
        'SOS_comp1_outlet': SOS_comp1_outlet,
        'SOS_comp2_inlet': SOS_comp2_inlet,
        'SOS_comp2_outlet': SOS_comp2_outlet,
    })


def base_case(results, EOS, vis_model, comp_specs, pipe_specs):
    flow = 60.0
    if EOS == 'GERG2008':
        f = jneqsim.thermo.system.SystemGERG2008Eos()
    else:
        f = fluid(EOS)
    f.addComponent('methane', 1.0)
    #f.setTemperature(50.15, 'C')
    #f.setPressure(90.0, 'bara')
    f.getPhase('gas').getPhysicalProperties().setViscosityModel(vis_model)
    if EOS != 'GERG2008': f.setMixingRule('classic')
    f.initPhysicalProperties()
    f.init(0)
    f.initProperties()
    TPflash(f)
    #print(vis_model, f.getPhase('gas').getPhysicalProperties().getViscosity())



    s = stream("fluid1", f)
    s.setFlowRate(flow, 'MSm3/day')
    s.setTemperature(50.15, 'C')
    s.setPressure(90.0, 'bara')
    s.run()

    

    #print("stream: ", vis_model, s.getFluid().getPhase('gas').getPhysicalProperties().getViscosity())
    compr_pipe_system(s, 'BaseCase: Methane', results, comp_specs, pipe_specs)


def hydrogen_case(results, EOS, vis_model, flow, comp_specs, pipe_specs):
    E   = energy_demand(flow, EOS, comp_specs['p_out'], comp_specs['T_cool'])
    mH2 = E * 1e6 / LHV['hydrogen']  # kg/sec

    if EOS == 'Leachman':
        f = jneqsim.thermo.system.SystemLeachmanEos()
    else:
        f = fluid(EOS)
        #f.setMixingRule('classic')

    f.addComponent('hydrogen', mH2, 'kg/sec')
    #if EOS != 'GERG2008'or EOS != 'Leachman': f.setMixingRule('classic')
    f.getPhase('gas').getPhysicalProperties().setViscosityModel(vis_model)

    f.initPhysicalProperties()
    f.init(0)
    f.initProperties()
    TPflash(f)
    f.initThermoProperties()
    #print(vis_model, f.getPhase('gas').getPhysicalProperties().getViscosity())


    s = stream("hydrogen_s", f)
    s.setFlowRate(mH2, 'kg/sec')
    s.setTemperature(50.15, 'C')
    s.setPressure(90.0, 'bara')
    s.run()


    compr_pipe_system(s, 'Hydrogen Case', results, comp_specs, pipe_specs)
    #print("stream: ", vis_model, s.getFluid().getPhase('gas').getPhysicalProperties().getViscosity())


def helium_case(results, EOS, vis_model, flow, comp_specs, pipe_specs):
    E   = energy_demand(flow, EOS, comp_specs['p_out'], comp_specs['T_cool'])
    mH2 = E * 1e6 / LHV['hydrogen']  # kg/sec

    if EOS == 'Vega':
        f = jneqsim.thermo.system.SystemVegaEos()
    else:
        f = fluid(EOS)
        #f.setMixingRule('classic')

    f.addComponent('helium', mH2, 'kg/sec')
    #if EOS != 'GERG2008'or EOS != 'Leachman': f.setMixingRule('classic')
    f.getPhase('gas').getPhysicalProperties().setViscosityModel(vis_model)

    f.initPhysicalProperties()
    f.init(0)
    f.initProperties()
    TPflash(f)
    f.initThermoProperties()
    #print(vis_model, f.getPhase('gas').getPhysicalProperties().getViscosity())


    s = stream("helium_s", f)
    s.setFlowRate(mH2, 'kg/sec')
    s.setTemperature(50.15, 'C')
    s.setPressure(90.0, 'bara')
    s.run()
    compr_pipe_system(s, 'Helium Case', results, comp_specs, pipe_specs)
    #print("stream: ", vis_model, s.getFluid().getPhase('gas').getPhysicalProperties().getViscosity())


def mix_cases(results, EOS, vis_model, flow, comp_specs, pipe_specs):
    fractions = list(np.linspace(0.1, 0.9, 9))
    for x in fractions:
        E    = energy_demand(flow, EOS, comp_specs['p_out'], comp_specs['T_cool'])
        MW   = {'methane':16.04, 'hydrogen':2.016}
        mix_MW = (1-x)*MW['methane'] + x*MW['hydrogen']
        w_ch4  = (1-x)*MW['methane']/mix_MW
        w_h2   = x*MW['hydrogen']/mix_MW
        m_dot  = E * 1e6 / (w_ch4*LHV['methane'] + w_h2*LHV['hydrogen'])


        if EOS == 'GERG2008':
            f = jneqsim.thermo.system.SystemGERG2008Eos()
        else:
            f = fluid(EOS)
            #f.setMixingRule('classic')
        f.addComponent('methane', 1-x, 'mol/sec')
        f.addComponent('hydrogen', x, 'mol/sec')
        #if EOS != 'GERG2008': f.setMixingRule('classic')
        if EOS != 'GERG2008' or EOS != 'Leachman': f.setMixingRule('classic')
        f.getPhase('gas').getPhysicalProperties().setViscosityModel(vis_model)

        f.initPhysicalProperties()
        f.init(0)
        f.initProperties()
        TPflash(f)

        s = stream("stream2", f)
        s.setFlowRate(m_dot, 'kg/sec')
        s.setTemperature(50.15, 'C')
        s.setPressure(90.0, 'bara')
        s.run()
        
        label = f"Mix ({1-x:.2f}+{x:.2f})"
        compr_pipe_system(s, label, results, comp_specs, pipe_specs)


def extract_h2_frac(label: str) -> float:
    if "Hydrogen Case" in label and "Methane" not in label: return 1.0
    if "Methane" in label and "Hydrogen" not in label: return 0.0
    m = re.search(r"\(([^)]+)\)", label)
    if m:
        parts = [p.strip() for p in m.group(1).split("+")]
        return float(parts[1]) if len(parts)==2 else float('nan')
    return float('nan')


import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_results_EOS(data: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    def save_fig(fig, name):
        fig.savefig(os.path.join(output_dir, f"{name}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
    # Build sorted DataFrames per scenario
    dfs = {name: pd.DataFrame(results) for name, results in data.items()}
    for name, df in dfs.items():
        df['h2'] = df['Case'].apply(extract_h2_frac)
        df['total_power'] = df['power1'] + df['power2']
        dfs[name] = df.sort_values('h2')

    # Plot styles
    style = {
        'Optimal': {'marker':'o', 'color':'red'},
        'PR': {'marker':'s', 'color':'blue'},
        'SRK': {'marker':'^', 'color':'green'}
    }

    # Helper to save and close
    def save_fig(fig, name):
        path = os.path.join(output_dir, f"{name}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Constants
    fig, ax1 = plt.subplots(figsize=(7,5))
    ax2 = ax1.twinx()  # right‐hand axis

    # Plot your head curves
    for name, df in dfs.items():
        x = df['h2']
        ax1.plot(
            x, df['head1'], '-', linewidth=0.5,
            label=f"{name} Comp1", **style[name],
            mfc='none', markeredgewidth=1.5
        )
        ax1.plot(
            x, df['head2'], '--', linewidth=0.5,
            label=f"{name} Comp2", **style[name],
            mfc='none', markeredgewidth=1.5
        )

    # Now plot the MW curve once, in red
    # Pick any df to get the common x and MW
    any_df = next(iter(dfs.values()))
    x_common = any_df['h2']
    mw_common = any_df['MW']  # or compute: x_common*MW_H2 + (1-x_common)*MW_other

    ax2.plot(
        x_common, mw_common*1e3, ':', linewidth=1.5,
        label="Mixture MW", color='red'
    )

    # Label axes
    ax1.set_xlabel("H₂ Molar Fraction")
    ax1.set_ylabel("Polytropic Head (kJ/kg)")
    ax2.set_ylabel("Mixture Molar Weight (kg/kmol)", color='red')

    # Color the right ticks & spine
    ax2.tick_params(axis='y', colors='red')
    ax2.spines['right'].set_edgecolor('red')

    # Legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1, l1, title="Case/Stage", loc="upper left", fontsize="small")
    ax2.legend(h2, l2, title=None, loc="upper right", fontsize="small")

    ax1.grid(True)
    fig.tight_layout()
    save_fig(fig, 'polytropic_head_with_mw')



    # 2) Pressure Drop
    fig = plt.figure(figsize=(7,5))
    for name, df in dfs.items():
        plt.plot(df['h2'], df['dp_per_m']*1e3, linestyle='-', linewidth=0.5,
                 marker=style[name]['marker'], color=style[name]['color'], mfc='none', markeredgewidth=1.5)
    plt.xlabel("H₂ Molar Fraction")
    plt.ylabel("Pressure Drop per Meter (mbara/m)")
    #plt.title("Pressure Drop vs. H₂ Fraction")
    plt.grid(True)
    plt.legend(dfs.keys(), title="Case")
    save_fig(fig, 'pressure_drop')

    # 3) Density
    fig = plt.figure(figsize=(7,5))
    for name, df in dfs.items():
        plt.plot(df['h2'], df['density'], linestyle='-', linewidth=0.5,
                 marker=style[name]['marker'], color=style[name]['color'], mfc='none', markeredgewidth=1.5)
    plt.xlabel("H₂ Molar Fraction")
    plt.ylabel("Density (kg/m³)")
    #plt.title("Density vs. H₂ Fraction")
    plt.grid(True)
    plt.legend(dfs.keys(), title="Case")
    save_fig(fig, 'density')

    # 4) Total Power
    fig, ax = plt.subplots(figsize=(7,5))

    # 1) Plot & label every line
    for name, df in dfs.items():
        x = df['h2']
        ax.plot(
            x, df['total_power'], '-',
            label=f"{name} – Total",
            **style[name], mfc='none', markeredgewidth=1.5
        )
        ax.plot(
            x, df['power1'], ':',
            label=f"{name} – Comp1",
            **style[name], mfc='none', markeredgewidth=1.5
        )
        ax.plot(
            x, df['power2'], '-.',
            label=f"{name} – Comp2",
            **style[name], mfc='none', markeredgewidth=1.5
        )

    ax.set_xlabel("H₂ Molar Fraction")
    ax.set_ylabel("Power (MW)")
    ax.grid(True)

    # 2) Pull back exactly the handles you want
    handles, labels = ax.get_legend_handles_labels()

    # 3) Draw the legend
    ax.legend(
        handles, labels,
        title="Case / Stream",
        fontsize="small",
        ncol=2,
        loc="upper left"
    )

    fig.tight_layout()
    save_fig(fig, 'power')

    # 5) Mass & Volume Flow
    from matplotlib.lines import Line2D

    fig, ax1 = plt.subplots(figsize=(7,5))
    ax2 = ax1.twinx()

    # plot both flows
    for name, df in dfs.items():
        # mass flow: solid black + marker
        ax1.plot(df['h2'], df['massFlow'],
                 linestyle='-', linewidth=0.8,
                 marker=style[name]['marker'],
                 color='black', mfc='none', markeredgewidth=1.5)
        # std vol flow: dotted red + same marker
        ax2.plot(df['h2'], df['stdVolFlow'],
                 linestyle=':', linewidth=0.8,
                 marker=style[name]['marker'],
                 color='red', mfc='none', markeredgewidth=1.5)

    # axis labels & grid
    ax1.set_xlabel("H₂ Molar Fraction")
    ax1.set_ylabel("Mass flow (kg/s)")
    ax2.set_ylabel("Std. vol. flow (MSm³/day)", color='red')
    ax1.spines['right'].set_visible(False)
    ax2.spines ['right'].set_color('red')
    ax2.tick_params(axis='y', colors='red')
    ax2.yaxis.label.set_color('red')          
    ax1.grid(True)

    # build a legend for just the CASE (marker shapes)
    case_handles = [
        Line2D([0], [0],
               marker=style[name]['marker'],
               color='black', linestyle='',
               mfc='none', markeredgewidth=1.5,
               label=name)
        for name in dfs.keys()
    ]
    legend_cases = ax1.legend(handles=case_handles,
                              title="Case",
                              loc="upper right",
                              frameon=True)
    ax1.add_artist(legend_cases)

    # build a legend for just the FLOW TYPE (line style + color)
    flow_handles = [
        Line2D([0], [0], color='black', linestyle='-', label='Mass flow'),
        Line2D([0], [0], color='red',   linestyle=':', label='Std. vol. flow')
    ]
    #plt.title("Mass & Volumetric Flow vs. H₂ Fraction")
    save_fig(fig, 'mass_volume_flow')
    import numpy as np

    # 6) Velocity Squared + Density (dual‑axis)
    area = np.pi * (0.98/2)**2

    # create figure and twin axes
    fig, ax1 = plt.subplots(figsize=(7,5))
    ax2 = ax1.twinx()

    vel_lines = []
    dens_lines = []
    for name, df in dfs.items():
        # compute velocity squared
        Q  = df['massFlow'] / df['density']
        u2 = (Q / area)**2

        # plot velocity² on left axis
        l1, = ax1.plot(
            df['h2'], u2,
            linestyle='-',
            marker=style[name]['marker'],
            color=style[name]['color'],
            mfc='none',
            markeredgewidth=1.5,
            label=name
        )
        vel_lines.append(l1)

        # plot density on right axis
        l2, = ax2.plot(
            df['h2'], df['density'],
            linestyle='--',
            marker=style[name]['marker'],
            color=style[name]['color'],
            mfc='none',
            markeredgewidth=1.5,
            label=name
        )
        dens_lines.append(l2)

    # labels
    ax1.set_xlabel("H₂ Molar Fraction")
    ax1.set_ylabel("Velocity² (m²/s²)")
    ax2.set_ylabel("Density (kg/m³)")

    # grid on primary axis
    ax1.grid(True)

    # separate legends for clarity
    leg1 = ax1.legend(handles=vel_lines, title="Case (Velocity²)", loc="upper center", fontsize="small")
    leg2 = ax2.legend(handles=dens_lines, title="Case (Density)", loc="center left", fontsize="small")
    ax1.add_artist(leg1)

    # save
    save_fig(fig, 'eos_velocity_squared_with_density')

    # 7) Relative deviations vs Optimal
    opt = dfs['Optimal']
    rel = {}
    for name in ['PR', 'SRK']:
        dfm = opt.merge(dfs[name], on='h2', suffixes=('_opt', f'_{name.lower()}'))
        rel[name] = dfm.assign(
            dens_rel   = 100*(dfm[f'density_{name.lower()}']   - dfm['density_opt'])   / dfm['density_opt'],
            power_rel  = 100*(dfm[f'total_power_{name.lower()}']- dfm['total_power_opt'])/ dfm['total_power_opt'],
            power1_rel = 100*(dfm[f'power1_{name.lower()}']    - dfm['power1_opt'])    / dfm['power1_opt'],
            power2_rel = 100*(dfm[f'power2_{name.lower()}']    - dfm['power2_opt'])    / dfm['power2_opt'],
            dp_rel     = 100*(dfm[f'dp_per_m_{name.lower()}']  - dfm['dp_per_m_opt'])  / dfm['dp_per_m_opt'],
            head1_rel  = 100*(dfm[f'head1_{name.lower()}']     - dfm['head1_opt'])     / dfm['head1_opt'],
            head2_rel  = 100*(dfm[f'head2_{name.lower()}']     - dfm['head2_opt'])     / dfm['head2_opt']
        )

    rel_styles = {'PR':{'marker':'s','linestyle':'-'}, 'SRK':{'marker':'^','linestyle':'-'}}
    rel_metrics = [
        ('dens_rel',
        r'$100\,(\rho_{\mathrm{model}} - \rho_{\mathrm{Optimal}})/\rho_{\mathrm{Optimal}}\,(\%)$',
        'Density Rel Dev vs Optimal'),
        ('power_rel',
        r'$100\,(P_{\mathrm{model}} - P_{\mathrm{Optimal}})/P_{\mathrm{Optimal}}\,(\%)$',
        'Power Rel Dev vs Optimal'),
        ('dp_rel',
        r'$100\,(\Delta p_{\mathrm{model}} - \Delta p_{\mathrm{Optimal}})/\Delta p_{\mathrm{Optimal}}\,(\%)$',
        'Pressure Drop Rel Dev vs Optimal'),
        ('head1_rel',
        r'$100\,(H_{1,\mathrm{model}} - H_{1,\mathrm{Optimal}})/H_{1,\mathrm{Optimal}}\,(\%)$',
        'Comp1 Head Rel Dev vs Optimal'),
        ('head2_rel',
        r'$100\,(H_{2,\mathrm{model}} - H_{2,\mathrm{Optimal}})/H_{2,\mathrm{Optimal}}\,(\%)$',
        'Comp2 Head Rel Dev vs Optimal')
    ]

    for key, yl, title in rel_metrics:
        fig = plt.figure(figsize=(6,4))
        plt.axhline(0, color='k', linestyle='--')
        for name in ['PR', 'SRK']:
            plt.plot(rel[name]['h2'], rel[name][key],
                     label=name,
                     marker=rel_styles[name]['marker'],
                     linestyle=rel_styles[name]['linestyle'],
                     color=style[name]['color'],
                     mfc='none', markeredgewidth=1.5)
        plt.xlabel('H₂ Fraction')
        plt.ylabel(yl)
        #plt.title(title)
        plt.grid(True)
        plt.legend()
        save_fig(fig, f"rel_dev_{key}")

    fig = plt.figure(figsize=(6,4))
    plt.axhline(0, color='k', linestyle='--')

    for name in ['PR', 'SRK']:
        # Comp1
        plt.plot(
            rel[name]['h2'], rel[name]['head1_rel'],
            label=f"{name} Comp1",
            marker=rel_styles[name]['marker'],
            linestyle=rel_styles[name]['linestyle'],
            color=style[name]['color'],
            mfc='none', markeredgewidth=1.5
        )
        # Comp2 (use dashed line to distinguish)
        plt.plot(
            rel[name]['h2'], rel[name]['head2_rel'],
            label=f"{name} Comp2",
            marker=rel_styles[name]['marker'],
            linestyle='--',
            color=style[name]['color'],
            mfc='none', markeredgewidth=1.5
        )

    plt.xlabel('H₂ Fraction')
    plt.ylabel(
        r'$100\,(H_{i,\mathrm{model}} - H_{i,\mathrm{Optimal}})/H_{i,\mathrm{Optimal}}\,(\%)$'
    )
    # (Optional) plt.title('Head Relative Deviation vs Optimal')
    plt.grid(True)
    plt.legend(title="Model / Compressor")
    fig.tight_layout()
    save_fig(fig, 'rel_dev_heads_combined')

    fig, ax = plt.subplots(figsize=(6,4))

    # zero‐line
    ax.axhline(0, color='k', linestyle='--', linewidth=1)

    for name in ['PR','SRK']:
        col = style[name]['color']
        m   = rel_styles[name]['marker']
        # Comp1
        ax.plot(
            rel[name]['h2'], rel[name]['power_rel'],
            label=f"{name} Total",
            marker=m, linestyle='-',
            color=col, mfc='none', markeredgewidth=1.5
        )
        # Comp2
        ax.plot(
            rel[name]['h2'], rel[name]['power1_rel'],
            label=f"{name} Comp1",
            marker=m, linestyle=':',
            color=col, mfc='none', markeredgewidth=1.5
        )
        # Combined
        ax.plot(
            rel[name]['h2'], rel[name]['power2_rel'],
            label=f"{name} Comp2",
            marker=m, linestyle='-.',
            color=col, mfc='none', markeredgewidth=1.5
        )

    ax.set_xlabel('H₂ Fraction')
    ax.set_ylabel(
        r'$100\,(P_{\mathrm{model},i} - P_{\mathrm{Optimal},i})/P_{\mathrm{Optimal},i}\,(\%)$'
    )
    ax.grid(True)
    ax.legend(title="Model / Stream", loc='best', fontsize='small')
    fig.tight_layout()
    save_fig(fig, 'rel_dev_power_combined')


    # 8) Speed of Sound (after) & Mach Number
    from matplotlib.lines import Line2D
    import numpy as np

    fig, ax1 = plt.subplots(figsize=(7,5))
    ax2 = ax1.twinx()

    # pipe cross-sectional area (m²)
    diameter = 0.98
    area     = np.pi * (diameter/2)**2

    for name, df in dfs.items():
        x = df['h2']

        # SOS after compressors (solid)
        ax1.plot(x, df['SOS2'],
                 linestyle='-',
                 marker=style[name]['marker'],
                 color=style[name]['color'],
                 mfc='none',
                 markeredgewidth=1.5,
                 label='_nolegend_')

        # Mach number after compressors (dotted)
        u    = df['massFlow'] / df['density'] / area
        Mach = u / df['SOS2']
        ax2.plot(x, Mach,
                 linestyle=':',
                 marker=style[name]['marker'],
                 color=style[name]['color'],
                 mfc='none',
                 markeredgewidth=1.5,
                 label='_nolegend_')

    # axis labels & styling
    ax1.set_xlabel("H₂ Molar Fraction")
    ax1.set_ylabel("Speed of Sound after compressors (m/s)")
    ax2.set_ylabel("Mach Number")
    ax1.grid(True)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_color('black')
    ax2.tick_params(axis='y', colors='black')
    ax2.yaxis.label.set_color('black')

    # Legend 1: marker shapes = model
    model_handles = [
        Line2D([0], [0],
               marker=style[name]['marker'],
               color=style[name]['color'],
               linestyle='',
               mfc='none',
               markeredgewidth=1.5,
               label=name)
        for name in dfs.keys()
    ]
    ax1.legend(handles=model_handles,
               title="Case / Model",
               loc="upper left",
               frameon=True)

    # Legend 2: line styles = series
    series_handles = [
        Line2D([0], [0], color='black', linestyle='-',  label='Speed of Sound'),
        Line2D([0], [0], color='black', linestyle=':',  label='Mach')
    ]
    ax2.legend(handles=series_handles,
               title="Series",
               loc="best",
               frameon=True)

    #plt.title("Post-Compression Speed of Sound & Mach vs. H₂ Fraction")
    save_fig(fig, 'speed_and_mach_after_compressor')



    fig, ax1 = plt.subplots(figsize=(7,5))
    ax2 = ax1.twinx()

    # pipe cross-sectional area (m²)
    diameter = 0.98
    area     = np.pi * (diameter/2)**2

    for name, df in dfs.items():
        x = df['h2']

        # SOS after compressors (solid)
        ax1.plot(x, df['SOS_pipe_out'],
                 linestyle='-',
                 marker=style[name]['marker'],
                 color=style[name]['color'],
                 mfc='none',
                 markeredgewidth=1.5,
                 label='_nolegend_')

        ax2.plot(x, df['vel_pipe_out']/df['SOS_pipe_out'],
                 linestyle=':',
                 marker=style[name]['marker'],
                 color=style[name]['color'],
                 mfc='none',
                 markeredgewidth=1.5,
                 label='_nolegend_')

    # axis labels & styling
    ax1.set_xlabel("H₂ Molar Fraction")
    ax1.set_ylabel("Speed of sound after pipe segment (m/s)")
    ax2.set_ylabel("Mach Number")
    ax1.grid(True)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_color('black')
    ax2.tick_params(axis='y', colors='black')
    ax2.yaxis.label.set_color('black')

    # Legend 1: marker shapes = model
    model_handles = [
        Line2D([0], [0],
               marker=style[name]['marker'],
               color=style[name]['color'],
               linestyle='',
               mfc='none',
               markeredgewidth=1.5,
               label=name)
        for name in dfs.keys()
    ]
    ax1.legend(handles=model_handles,
               title="Case / Model",
               loc="upper left",
               frameon=True)

    # Legend 2: line styles = series
    series_handles = [
        Line2D([0], [0], color='black', linestyle='-',  label='Speed of Sound'),
        Line2D([0], [0], color='black', linestyle=':',  label='Mach')
    ]
    ax2.legend(handles=series_handles,
               title="Series",
               loc="lower right",
               frameon=True)

    #plt.title("Post-Compression Speed of Sound & Mach vs. H₂ Fraction")
    save_fig(fig, 'SOS_and_mach_after_pipe')


    # 11) Viscosity vs. H₂ Fraction
    fig = plt.figure(figsize=(7,5))
    for name, df in dfs.items():
        plt.plot(df['h2'], df['viscosity'],
                 linestyle='-', linewidth=0.8,
                 marker=style[name]['marker'],
                 color=style[name]['color'], mfc='none', markeredgewidth=1.5,
                 label=name)
    plt.xlabel("H₂ Molar Fraction")
    plt.ylabel("Viscosity (Pa·s)")
    plt.grid(True)
    plt.legend(title="Model")
    save_fig(fig, 'viscosity')

      # 12) Outlet temperature of the pipe
    fig = plt.figure(figsize=(7,5))
    for name, df in dfs.items():
        plt.plot(df['h2'], df['p_out_pipe'],
                 linestyle='-', linewidth=0.8,
                 marker=style[name]['marker'],
                 color=style[name]['color'], mfc='none', markeredgewidth=1.5,
                 label=name)
    plt.xlabel("H₂ Molar Fraction")
    plt.ylabel("Pressure out of the pipe segment (bar)")
    plt.grid(True)
    plt.legend(title="Model")
    save_fig(fig, 'p_out_pipe')

    # 12) Outlet temperature of the pipe
    fig = plt.figure(figsize=(7,5))
    for name, df in dfs.items():
        plt.plot(df['h2'], df['T_out_pipe'],
                 linestyle='-', linewidth=0.8,
                 marker=style[name]['marker'],
                 color=style[name]['color'], mfc='none', markeredgewidth=1.5,
                 label=name)
    plt.xlabel("H₂ Molar Fraction")
    plt.ylabel("Temperature out of the pipe segment (°C)")
    plt.grid(True)
    plt.legend(title="Model")
    save_fig(fig, 'T_out_pipe')

    # Combined Outlet Temperatures for Comp1 & Comp2
    fig, ax = plt.subplots(figsize=(7,5))

    for name, df in dfs.items():
        ax.plot(
            df['h2'], df['T_out_comp1'],
            linestyle='-', linewidth=0.8,
            marker=style[name]['marker'],
            color=style[name]['color'], mfc='none', markeredgewidth=1.5,
            label=f"{name} – Comp1"
        )
        ax.plot(
            df['h2'], df['T_out_comp2'],
            linestyle='--', linewidth=0.8,
            marker=style[name]['marker'],
            color=style[name]['color'], mfc='none', markeredgewidth=1.5,
            label=f"{name} – Comp2"
        )

    ax.set_xlabel("H₂ Molar Fraction")
    ax.set_ylabel("Outlet Temperature (°C)")
    ax.grid(True)
    ax.legend(title="Model / Compressor", fontsize="small", ncol=2)
    fig.tight_layout()
    save_fig(fig, 'T_out_compr_combined')


def plot_results_viscosity(data: dict, output_dir: str):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(output_dir, exist_ok=True)
    def save_fig(fig, name):
        fig.savefig(os.path.join(output_dir, f"{name}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    # prepare DataFrames
    dfs = {name: pd.DataFrame(results) for name, results in data.items()}
    for name, df in dfs.items():
        df['h2'] = df['Case'].apply(extract_h2_frac)
        df['total_power'] = df['power1'] + df['power2']
        dfs[name] = df.sort_values('h2')

    # styling
    style = {
        'Optimal Modified': {'marker':'o','color':'red'},
        'Optimal':          {'marker':'s','color':'blue'},
        'LBC':              {'marker':'^','color':'green'},
        'PFCT':             {'marker':'v','color':'purple'},
        'FT':               {'marker':'d','color':'orange'}
    }

    # 1) Polytropic Head
    fig = plt.figure(figsize=(7,5))
    for name, df in dfs.items():
        x = df['h2']
        plt.plot(x, df['head1'],
                 linestyle='-',
                 marker=style[name]['marker'],
                 color=style[name]['color'],
                 mfc='none',
                 label=f"{name} Comp1")
        plt.plot(x, df['head2'],
                 linestyle='--',
                 marker=style[name]['marker'],
                 color=style[name]['color'],
                 mfc='none',
                 label=f"{name} Comp2")
    plt.xlabel("H₂ Molar Fraction")
    plt.ylabel("Polytropic Head (kJ/kg)")
    plt.grid(True)
    plt.legend(title="Case/Stage")
    save_fig(fig, 'visc_polytropic_head')

    # 2) Pressure Drop
    fig = plt.figure(figsize=(7,5))
    for name, df in dfs.items():
        plt.plot(df['h2'], df['dp_per_m']*1e3,
                 linestyle='-',
                 marker=style[name]['marker'],
                 color=style[name]['color'],
                 mfc='none',
                 label=name)
    plt.xlabel("H₂ Molar Fraction")
    plt.ylabel("Pressure Drop per Meter (mbara/m)")
    plt.grid(True)
    plt.legend(title="Case")
    save_fig(fig, 'visc_pressure_drop')

    # 3) Density
    fig = plt.figure(figsize=(7,5))
    for name, df in dfs.items():
        plt.plot(df['h2'], df['density'],
                 linestyle='-',
                 marker=style[name]['marker'],
                 color=style[name]['color'],
                 mfc='none',
                 label=name)
    plt.xlabel("H₂ Molar Fraction")
    plt.ylabel("Density (kg/m³)")
    plt.grid(True)
    plt.legend(title="Case")
    save_fig(fig, 'visc_density')

    # 4) Total Power
    fig = plt.figure(figsize=(7,5))
    for name, df in dfs.items():
        plt.plot(df['h2'], df['total_power'],
                 linestyle='-',
                 marker=style[name]['marker'],
                 color=style[name]['color'],
                 mfc='none',
                 label=name)
    plt.xlabel("H₂ Molar Fraction")
    plt.ylabel("Total Power (MW)")
    plt.grid(True)
    plt.legend(title="Case")
    save_fig(fig, 'visc_total_power')

    # 5) Mass & Volume Flow (dual‐axis)
    fig, ax1 = plt.subplots(figsize=(7,5))
    ax2 = ax1.twinx()
    for name, df in dfs.items():
        ax1.plot(df['h2'], df['massFlow'],
                 linestyle='-',
                 marker=style[name]['marker'],
                 color='black',
                 mfc='none')
        ax2.plot(df['h2'], df['stdVolFlow'],
                 linestyle=':',
                 marker=style[name]['marker'],
                 color='red',
                 mfc='none')
    ax1.set_xlabel("H₂ Molar Fraction")
    ax1.set_ylabel("Mass flow (kg/s)")
    ax2.set_ylabel("Std. vol. flow (MSm³/day)", color='red')
    ax2.tick_params(axis='y', colors='red')
    ax2.spines['right'].set_color('red')
    ax1.grid(True)

    # legends
    model_handles = [
        Line2D([0],[0], marker=style[n]['marker'],
               color='black', linestyle='', mfc='none', label=n)
        for n in dfs.keys()
    ]
    ax1.legend(handles=model_handles, title="Case", loc="upper right")
    flow_handles = [
        Line2D([0],[0], color='black', linestyle='-', label='Mass flow'),
        Line2D([0],[0], color='red',   linestyle=':', label='Std. vol. flow')
    ]
    ax2.legend(handles=flow_handles, title="Flow type", loc="upper left")
    save_fig(fig, 'visc_mass_volume_flow')

    # 6) Velocity Squared
    area = np.pi * (0.98/2)**2
    fig = plt.figure(figsize=(7,5))
    for name, df in dfs.items():
        Q  = df['massFlow']/df['density']
        u2 = (Q/area)**2
        plt.plot(df['h2'], u2,
                 linestyle='-',
                 marker=style[name]['marker'],
                 color=style[name]['color'],
                 mfc='none',
                 label=name)
    plt.xlabel("H₂ Molar Fraction")
    plt.ylabel("Velocity² (m²/s²)")
    plt.grid(True)
    plt.legend(title="Case")
    save_fig(fig, 'visc_velocity_squared')

        # 7) Relative deviations vs Optimal Modified (with EOS‐style y-labels)
    opt = dfs['Optimal Modified']
    rel = {}
    for name in dfs:
        if name == 'Optimal Modified': continue
        dfm = opt.merge(dfs[name], on='h2', suffixes=('_opt', f'_{name.lower()}'))
        rel[name] = dfm.assign(
            dens_rel  = 100*(dfm[f'density_{name.lower()}']   - dfm['density_opt'])  / dfm['density_opt'],
            power_rel = 100*(dfm[f'total_power_{name.lower()}']- dfm['total_power_opt'])/ dfm['total_power_opt'],
            dp_rel    = 100*((dfm[f'dp_per_m_{name.lower()}']  - dfm['dp_per_m_opt'])  / dfm['dp_per_m_opt'])
        )

    rel_styles = {
        'Optimal': {'marker':'s','linestyle':'-'},
        'LBC':     {'marker':'^','linestyle':'-'},
        'PFCT':    {'marker':'v','linestyle':'-'},
        'FT':      {'marker':'d','linestyle':'-'}
    }

    rel_metrics = [
        ('dens_rel',
         r'$100\,(\,\rho_{\mathrm{model}} - \rho_{\text{Optimal Modified}}\,)\,/\,\rho_{\text{Optimal Modified}}\,\%$'),
        ('power_rel',
         r'$100\,(P_{\mathrm{model}} - P_{\text{Optimal Modified}})\,/\,P_{\text{Optimal Modified}}\,\%$'),
        ('dp_rel',
         r'$100\,(\Delta P_{\mathrm{model}} - \Delta P_{\text{Optimal Modified}})\,/\,\Delta P_{\text{Optimal Modified}}\,\%$')
    ]


    for key, ylab in rel_metrics:
        fig = plt.figure(figsize=(6,4))
        plt.axhline(0, color='k', linestyle='--')
        for name in rel:
            plt.plot(rel[name]['h2'], rel[name][key],
                     label=name,
                     marker=rel_styles[name]['marker'],
                     linestyle=rel_styles[name]['linestyle'],
                     color=style[name]['color'],
                     mfc='none')
        plt.xlabel("H₂ Molar Fraction")
        plt.ylabel(ylab)
        plt.grid(True)
        plt.legend(title="Case")
        save_fig(fig, f"visc_rel_dev_{key}")


        # 8) Gas‐Phase Viscosity
    fig = plt.figure(figsize=(7,5))
    for name, df in dfs.items():
        plt.plot(df['h2'], df['viscosity']*10**(6),
                 linestyle='-',
                 marker=style[name]['marker'],
                 color=style[name]['color'],
                 mfc='none',
                 markeredgewidth=1.5,
                 label=name)
    plt.xlabel("H₂ Molar Fraction")
    plt.ylabel("Viscosity (μPa·s)")
    plt.grid(True)
    plt.legend(title="Case")
    save_fig(fig, 'visc_viscosity')

    #--- physical constants
    D              = 0.98      # pipe diameter [m]
    area           = np.pi * (D/2)**2
    epsilon        = 5e-6      # absolute roughness [m]
    rel_roughness  = epsilon / D

    # 1) Plot Reynolds number vs H2
    fig_Re = plt.figure(figsize=(7,5))
    for name, df in dfs.items():
        u  = df['massFlow'] / (df['density'] * area)
        Re = df['density'] * u * D / df['viscosity']
        plt.plot(df['h2'], Re,
                linestyle='-',
                marker=style[name]['marker'],
                color=style[name]['color'],
                mfc='none',
                markeredgewidth=1.5,
                label=name)

    plt.xlabel("H₂ Molar Fraction")
    plt.ylabel("Reynolds number, Re")
    plt.grid(True)
    plt.legend(title="Case")
    plt.tight_layout()
    save_fig(fig_Re, 'visc_Reynolds')


    # 2) Plot friction factor vs H2
    fig_f = plt.figure(figsize=(7,5))
    for name, df in dfs.items():
        # recompute Re for this case
        u  = df['massFlow'] / (df['density'] * area)
        Re = df['density'] * u * D / df['viscosity']

        # explicit Colebrook–White approximation
        f = (1.0 / (
            -1.8 * np.log10(rel_roughness/3.7 + 6.9/Re)
            )
            )**2

        plt.plot(df['h2'], f,
                linestyle='-',
                marker=style[name]['marker'],
                color=style[name]['color'],
                mfc='none',
                markeredgewidth=1.5,
                label=name)

    plt.xlabel("H₂ Molar Fraction")
    plt.ylabel("Friction factor, f")
    plt.ylim(0.001, 0.01)
    plt.grid(True)
    plt.legend(title="Case")
    plt.tight_layout()
    save_fig(fig_f, 'visc_friction_factor')


        # 9) Pipeline Outlet Temperature
    fig = plt.figure(figsize=(7,5))
    for name, df in dfs.items():
        plt.plot(df['h2'], df['T_out_pipe'],
                 linestyle='-',
                 marker=style[name]['marker'],
                 color=style[name]['color'],
                 mfc='none',
                 markeredgewidth=1.5,
                 label=name)
    plt.xlabel("H₂ Molar Fraction")
    plt.ylabel("Pipeline Outlet T (°C)")
    plt.grid(True)
    plt.legend(title="Case")
    save_fig(fig, 'visc_pipe_outlet_temperature')

    # 10) Pipeline Outlet Pressure
    fig = plt.figure(figsize=(7,5))
    for name, df in dfs.items():
        plt.plot(df['h2'], df['p_out_pipe'],
                 linestyle='-',
                 marker=style[name]['marker'],
                 color=style[name]['color'],
                 mfc='none',
                 markeredgewidth=1.5,
                 label=name)
    plt.xlabel("H₂ Molar Fraction")
    plt.ylabel("Pipeline Outlet p (bara)")
    plt.grid(True)
    plt.legend(title="Case")
    save_fig(fig, 'visc_pipe_outlet_pressure')

    # 11) Velocity & Mach vs H₂ Molar Fraction
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    for name, df in dfs.items():
        x = df['h2']
        vel = df['vel_pipe_out']        # outlet velocity [m/s]
        mach = vel / df['SOS_pipe_out']  # speed divided by speed of sound

        ax1.plot(x, vel,
                 linestyle='-',
                 marker=style[name]['marker'],
                 color=style[name]['color'],
                 mfc='none',
                 label=f"{name} Velocity")
        ax2.plot(x, mach,
                 linestyle='--',
                 marker=style[name]['marker'],
                 color=style[name]['color'],
                 mfc='none',
                 label=f"{name} Mach")

    # labels
    ax1.set_xlabel("H₂ Molar Fraction")
    ax1.set_ylabel("Outlet Velocity (m/s)")
    ax2.set_ylabel("Mach Number")

    # combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title("Outlet Velocity and Mach Number vs H₂ Fraction")
    plt.grid(True)
    plt.tight_layout()
    save_fig(fig, 'vel_mach_vs_h2')


    #plot density*vel^2 vs H2
    fig = plt.figure(figsize=(7,5))
    for name, df in dfs.items():
        e_kin = df['density'] * df['vel']**2
        plt.plot(df['h2'], e_kin,
                 linestyle='-',
                 marker=style[name]['marker'],
                 color=style[name]['color'],
                 mfc='none',
                 markeredgewidth=1.5,
                 label=name)
    plt.xlabel("H₂ Molar Fraction")
    plt.ylabel("Density * Velocity² (kg/m/s²)")
    plt.grid(True)
    plt.legend(title="Case")
    save_fig(fig, 'visc_kinetic_energy')


        #plot density*vel^2 vs H2
    fig = plt.figure(figsize=(7,5))
    for name, df in dfs.items():
        e_kin = df['density_pipe'] * df['vel_pipe_out']**2
        plt.plot(df['h2'], e_kin,
                 linestyle='-',
                 marker=style[name]['marker'],
                 color=style[name]['color'],
                 mfc='none',
                 markeredgewidth=1.5,
                 label=name)
    plt.xlabel("H₂ Molar Fraction")
    plt.ylabel("Density * Velocity² (kg/m/s²)")
    plt.grid(True)
    plt.legend(title="Case")
    save_fig(fig, 'visc_kinetic_energy_pipeout')





def main():
    comp_specs = {
        'eta_poly': 0.85,
        'p_mid':    127.279,
        'p_out':    180,
        'T_cool':   15.0
    }
    pipe_specs = {
        'length':     100e3,
        'diameter':   0.98,
        'roughness':  5e-6,
        'increments': 50,
        'isothermal': False,
        'U':          10,
        'T_wall':     5.0
    }
    flow = 60.0

    # --------------------------------------------------
    # Old model looping and plotting code commented out
    # --------------------------------------------------
    viscosity_models = ["Muzny_mod", ]
    '''
    models = {
        'Optimal Modified': (
            ['Leachman', 'GERG2008', 'GERG2008'],
            ['Muzny_mod','PFCT'     , 'MethaneModel']
        ),
        'Optimal': (
            ['Leachman', 'GERG2008', 'GERG2008'],
            ['Muzny'    ,'PFCT'     , 'LBC']
        ),
        'LBC': (
            ['Leachman', 'GERG2008', 'GERG2008'],
           ['LBC','LBC'   , 'LBC']
        ),
        'PFCT': (
            ['Leachman', 'GERG2008', 'GERG2008'],
            ['PFCT','PFCT', 'PFCT']
        ),
        'FT': (
            ['Leachman', 'GERG2008', 'GERG2008'],
            ['friction theory','friction theory'   , 'friction theory']
        )
    }
    '''
    '''
    models = {
        'Optimal':(
            ['Leachman', 'GERG2008', 'GERG2008'],
            ['Muzny'    ,'PFCT'     , 'LBC']
        ),
        'SRK':(
            ['srk'    , 'srk'    , 'srk'    ],
            ['Muzny'    ,'PFCT'     , 'LBC']
        ),
        'PR':(
            ['pr'    , 'pr'    , 'pr'    ],
            ['Muzny'    ,'PFCT'     , 'LBC']
        )
    }

    data = {}
    for case_name, (eos_list, vis_list) in models.items():
         results = []
         base_case(results, eos_list[2], vis_list[2], comp_specs, pipe_specs)
         mix_cases(results, eos_list[1], vis_list[1], flow, comp_specs, pipe_specs)
         hydrogen_case(results, eos_list[0], vis_list[0], flow, comp_specs, pipe_specs)
         data[case_name] = results

    plot_results_EOS(data, 'CASE_EOS_plots')
    '''
    #plot_results_viscosity(data, 'CASE_visc_plots')

    # Print pure-hydrogen gas viscosity by case (no longer needed)
    # print("Pure-hydrogen gas viscosity (Pa·s) by case:")
    # for case, results in data.items():
    #     entry = next((r for r in results if extract_h2_frac(r['Case']) == 1.0), None)
    #     if entry:
    #         print(f"  {case:20s}: {entry['viscosity']:.3e}")
    # --------------------------------------------------

    
    
    # Run only Hydrogen and Helium single-component cases
    # Hydrogen case
    results_h2 = []
    hydrogen_case(results_h2, 'Leachman', 'Muzny_mod', flow, comp_specs, pipe_specs)
    df_h2 = pd.DataFrame(results_h2)


    # Helium case
    results_he = []
    helium_case(results_he, 'Vega', 'KTA_mod', flow, comp_specs, pipe_specs)
    df_he = pd.DataFrame(results_he)

    # pick out the first (and only) row from each
    h2 = df_h2.iloc[0].drop('Case')
    he = df_he.iloc[0].drop('Case')

    # build a new DataFrame with metrics as the index, and one column per gas
    summary = pd.DataFrame({
        'Hydrogen': h2,
        'Helium':   he
    })

    # round off for readability
    pd.options.display.float_format = '{:.2f}'.format

    summary['He/H2_ratio'] = (
        100 * summary['Helium'] / summary['Hydrogen']
    ).round(2)

    #print it out
    print("\n---- Compressor + Pipeline Metrics ----")
    print(summary)
   

    #result_base = []
    #base_case(result_base, 'GERG2008', 'MethaneModel', comp_specs, pipe_specs)
    #df_base = pd.DataFrame(result_base)



    #pick out the first (and only) row from each
    #base = df_base.iloc[0].drop('Case')

    # build a new DataFrame with metrics as the index, and one column per gas
    #summary_base = pd.DataFrame({
     #   'Nase': base
    #})

    # round off for readability
    #summary_base = summary_base.round(3)

    # print it out
    #print("\n---- Compressor + Pipeline Metrics ----")
    #print(summary_base)
    




if __name__ == '__main__':
    main()
