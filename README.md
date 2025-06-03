# Master Thesis – NeqSim Hydrogen and Helium Modeling

This repository is linked to our master thesis, which focused on testing and validating equations of state (EOS) and viscosity models for hydrogen systems. The main objective was to improve NeqSim’s capabilities for modeling hydrogen-rich gas mixtures, and to support helium systems for cases where helium is used as a surrogate gas for hydrogen.

## What is Implemented

- The **Leachman** and **Vega** EOS are implemented, which can be used as accurate property generators for hydrogen and helium, respectively. These were further extended with system implementations in NeqSim. A similar system was started for **GERG2008**, but not completed.
- The viscosity models added include:
  - **New Muzny** and **KTA** models for hydrogen and helium.
  - **Modified** versions of the **LBC**, **Muzny**, and **KTA** models for pure methane, hydrogen, and helium, respectively.

## Repository Contents

The repository includes:

- Scripts implemented in NeqSim
- Streamlit property generator for hydrogen and helium
- Scripts for running simulations and case studies related to the thesis
- Data processing scripts
- Our preliminary project thesis and final master thesis document

## License

This repository is provided for academic and research use.

---

Developed as part of our MSc thesis at NTNU, 2025.
