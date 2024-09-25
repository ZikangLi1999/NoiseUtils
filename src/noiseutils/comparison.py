"""
Tool for domain-wise comparison
@Author: LI Zikang
@Date: 2024-09-18
"""
import dataclasses
import pathlib

import numpy as np
import pyvista as pv

from .time_domain_dataset import TimeDomainDataset
from .frequency_domain_dataset import FrequencyDomainDataset


@dataclasses.dataclass
class ProblemParameter:
    n_energy_group        : int
    n_cell                : int
    n_frequency           : int
    perturbation_frequency: float
    n_step                : int
    simulation_time       : float
    path                  : pathlib.Path


def get_result_from_vtk(parameter: ProblemParameter) -> pv.UnstructuredGrid:
    """
    Get a PyVista UnstructuredGrid containing neutron noise related data from both domains
    """
    # parameter pre-processing
    time_step = parameter.simulation_time / parameter.n_step

    # get time-domain data
    dataset_td = TimeDomainDataset(
        n_energy_group = parameter.n_energy_group,
        n_cell = parameter.n_cell,
        n_step = parameter.n_step
    )
    dataset_td.read(path = parameter.path / 'transient' / 'output' / 'Flux')
    dataset_td.to_frequency_domain()

    # get freqeuncy-domain data
    dataset_fd = FrequencyDomainDataset(
        n_energy_group = parameter.n_energy_group,
        n_cell = parameter.n_cell,
        n_frequency = parameter.n_frequency
    )
    dataset_fd.read(path = parameter.path / 'noise' / 'output' / 'noise' / 'noise.vtk')

    # find the DFT frequency nearest to the actual perturbation frequency
    dft_freqs = np.fft.fftfreq(
        n = parameter.n_step,
        d = time_step
    )
    nearest_index = np.argmin(np.abs(dft_freqs - parameter.perturbation_frequency))

    # prepare to merge data from two domains into one
    model = dataset_fd.model.copy()

    # Check the consistency of static flux in solvers of two domains
    for g in range(parameter.n_energy_group):
        model.cell_data[f"td-flux(G{g+1})"] = dataset_td.dataset[:, g, 0]
        model.cell_data[f"fd-flux(G{g+1})"] = dataset_fd.static_flux[:, g]
        model.cell_data[f"relerr-flux(G{g+1})"] = \
            (model.cell_data[f"td-flux(G{g+1})"] - model.cell_data[f"fd-flux(G{g+1})"]) \
            / model.cell_data[f"td-flux(G{g+1})"]

    # Rename noise-related fields from frequency-domain with prefix "fd-"
    PREFIX_TO_RENAME = ("amplitude", "phase")
    for field_name in model.cell_data.keys():
        # Jump over noise-irrelated fields
        if not np.any([field_name.startswith(prefix) for prefix in PREFIX_TO_RENAME]):
            continue

        # Rename field name with domain as prefix
        new_name = f"fd-{field_name}"
        if field_name.startswith("amplitude"):
            model.cell_data[new_name] = model.cell_data[field_name]
        elif field_name.startswith("phase"):
            model.cell_data[new_name] = model.cell_data[field_name]
        else:
            raise ValueError(f"Unrecgonized field name: {field_name}")
        del model.cell_data[field_name]

    # Add noise-related fields from time-domain with prefix "td-"
    SELECTED_FREQUENCY_TD = (nearest_index, ) # select the frequency points manually
    for f in SELECTED_FREQUENCY_TD:
        for g in range(parameter.n_energy_group):
            amp_name = f"td-amplitude(G{g+1},F{f+1})"
            model.cell_data[amp_name] = dataset_td.amplitude[:, g, f]
            phs_name = f"td-phase(G{g+1},F{f+1})"
            model.cell_data[phs_name] = dataset_td.phase[:, g, f]

    # The correspondence of frequency point between two domain should be set manually
    for g in range(parameter.n_energy_group):
        for q in ("amplitude", "phase"):
            err = f"relerr-{q}(G{g+1},F1)"
            fd  = f"fd-{q}(G{g+1},F1)"
            td  = f"td-{q}(G{g+1},F2)"
            model.cell_data[err] = \
                (model.cell_data[fd] - model.cell_data[td]) / model.cell_data[td]

    return model
