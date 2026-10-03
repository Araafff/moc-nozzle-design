# MOC Nozzle Design Solver

## Project Overview

This project designs minimum-length two-dimensional supersonic nozzles using the Method of Characteristics (MOC). The solver computes the geometry and flow properties of supersonic nozzles for specified exit Mach numbers.

## What It Does

The solver uses characteristic theory to calculate nozzle wall contours and flow properties. It finds the optimal nozzle shape that produces a uniform supersonic flow at the exit while keeping the nozzle length as short as possible.

## Physical Assumptions

- The flow is steady and two-dimensional.
- The flow is isentropic (no entropy generation).
- The gas follows ideal gas behavior with constant specific heat ratio.
- No viscous or heat transfer effects are considered.
- The flow starts from a throat and expands through a centered expansion fan.

## Initial Conditions

- Throat exit Mach number: M = 1.0 (sonic condition).
- Specific heat ratio: γ = 1.4 (air at room temperature).
- The nozzle expands from a throat with height y_throat = 1.0 unit.

## Design Parameters

- **Exit Mach numbers**: The solver designs nozzles for Me = 1.5, 3.0, and 6.0.
- **Number of expansion waves**: N = 40 waves are used to discretize the expansion fan.
- **Maximum turning angle**: θ_max = ν(Me) / 2, where ν is the Prandtl-Meyer function.

## Key Physics Relationships

- **Prandtl-Meyer function**: ν(M) relates Mach number to flow deflection angle.
- **Mach angle**: μ = arcsin(1/M) defines the characteristic line orientation.
- **Characteristic relations**: K_minus = θ + ν and K_plus = θ - ν are preserved along characteristic curves.
- **Area-Mach relation**: The quasi-1D isentropic relation connects flow area and Mach number.

## Program Output

The solver creates:

- **Figures**: Characteristic networks, Mach number contours, wall profiles, and comparison plots.
- **Data files**: CSV files containing coordinates and flow properties at all grid points, centerline, and wall locations.
- **Validation**: Exit area ratios from both MOC and quasi-1D theory are compared.

## How to Run

```bash
python moc_nozzle_design.py
```

The program will generate subdirectories `figures/` and `data/` containing all results.

## Dependencies

- Python 3.6 or higher
- numpy
- pandas
- matplotlib

Install dependencies:

```bash
pip install numpy pandas matplotlib
```

## Output Files

- `figures/`: Contains PNG plots for each design
- `data/`: Contains CSV files with MOC grid points, centerline flow, and wall coordinates
- `validation_summary.csv`: Comparison of exit area ratios from different methods

## Notes

This is a real 2-D grid-marching MOC solver that computes characteristic lines and flow properties using compatibility relations and intersection methods. The nozzle designs are optimal for minimum length while maintaining uniform supersonic flow at the exit.
