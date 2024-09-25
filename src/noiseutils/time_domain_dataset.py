"""
FFT on time-domain data
"""
import os

import numpy as np
import pyvista as pv

class TimeDomainDataset:

    _FIELD_NAME_TO_KEEP = ("Material", "RelativePower", "ActualPower")

    def __init__(self, n_energy_group: int, n_cell: int, n_step: int) -> None:
        """
        Input
        -----
        n_energy_group: number of energy groups
        n_cell: number of cells in the model
        n_step: number of time-steps
        """
        self.n_energy_group = n_energy_group
        self.n_cell = n_cell
        self.n_step = n_step + 1 # add the static state (step 0)
        self.dataset = np.zeros(shape=(
            self.n_cell, 
            self.n_energy_group, 
            self.n_step
        ))
    
    def read(self, path: str) -> None:
        self.path = path
        for t in range(self.n_step):
            fpath = os.path.join(self.path, "TSTEP{:0>4d}.vtk".format(t))
            model = pv.read(filename=fpath)
            for g in range(self.n_energy_group):
                self.dataset[:, g, t] = model.get_array(name="flux(G{:d})".format(g + 1))
            del model
    
    def to_frequency_domain(self) -> None:
        self.spectra = np.zeros_like(self.dataset, dtype=np.complex128)
        for c in range(self.n_cell):
            for g in range(self.n_energy_group):
                self.spectra[c, g, :] = np.fft.fft(
                    np.complex128(self.dataset[c, g, :] - np.mean(self.dataset[c, g, :])), 
                    n    = self.n_step,
                    norm = "forward"
                )
        
        self.amplitude = np.abs(self.spectra) # the modulus of forward fft spectrum is half of actual amplitude
        self.phase = np.angle(self.spectra, deg=True) # (-90deg, +90deg]

        half = self.n_step // 2 + 1
        self.amplitude = 2 * self.amplitude[:half]
        self.phase = self.phase[:half]

    def to_vtk(self, file_name: str = "noise_td.vtk") -> None:
        fpath = os.path.join(self.path, "TSTEP{:0>4d}.vtk".format(0))
        model: pv.UnstructuredGrid = pv.read(filename=fpath)
        field_name_to_remove = [
            field_name
            for field_name in model.array_names
            if field_name not in self._FIELD_NAME_TO_KEEP
        ]
        for field_name in field_name_to_remove:
            del model.cell_data[field_name]

        # The value of frequency should calculated with FFT theory further
        for f in range(self.n_step):
            for g in range(self.n_energy_group):
                model.cell_data[
                    "amplitude(F{:d},G{:d})".format(f+1, g+1)
                ] = self.amplitude[:, g, f]
        
        for f in range(self.n_step):
            for g in range(self.n_energy_group):
                model.cell_data[
                    "phase(F{:d},G{:d})".format(f+1, g+1)
                ] = self.phase[:, g, f]

        noise_filename = os.path.join(self.path, file_name)
        model.save(filename=noise_filename)


if __name__ == "__main__":
    dataset = TimeDomainDataset(
        n_energy_group = 2,
        n_cell = 4624,
        n_step = 49
    )
    dataset.read(path = "time-domain/output/Flux")
    dataset.to_frequency_domain()
    dataset.to_vtk()
