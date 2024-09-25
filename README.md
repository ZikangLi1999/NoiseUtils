# NoiseUtils

## Overview
Utility module helping processing neutron noise calculation based on morphy or vitas.

## Example
Here is a minimal example.
```python
import pathlib
import noiseutils


param = noiseutils.comparison.ProblemParameter(
    n_energy_group = 2,
    n_cell = 324,
    n_frequency = 1,
    perturbation_frequency = 1.0,
    n_step = 100,
    simulation_time = 1.0,
    path = pathlib.Path('mox-2d')
)

model = noiseutils.comparison.get_result_from_vtk(param)
model.save('mox-2d/noise_post.vtk')
```

> Written by LI Zikang at Sep. 25th, 2024.
