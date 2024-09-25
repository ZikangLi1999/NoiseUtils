"""
IFFT on frequency domain dataset
@Author: LI Zikang
"""
import os
import numpy as np
import pyvista as pv

class FrequencyDomainDataset:

    _FIELD_NAME_TO_KEEP = ("material", )

    def __init__(self, n_energy_group: int, n_cell: int, n_frequency: int) -> None:
        """
        Input
        -----
        n_energy_group: number of energy groups
        n_cell: number of cells in the model
        n_step: number of time-steps
        deg: phase angle in unit degree or not (radian)
        """
        self.n_energy_group = n_energy_group
        self.n_cell = n_cell
        self.n_frequency = n_frequency
    
    def read(self, path: str) -> None:
        self.path = path
        self.model: pv.UnstructuredGrid = pv.read(self.path)

        self.static_flux = np.zeros(shape=(
            self.n_cell, self.n_energy_group
        ))
        self.amplitude = np.zeros(shape=(
            self.n_cell, self.n_energy_group, self.n_frequency
        ))
        self.phase = np.zeros_like(self.amplitude)

        for g in range(self.n_energy_group):

            flux_name = "flux(G{:d})".format(g+1)
            self.static_flux[:, g] = self.model.get_array(flux_name)

            for f in range(self.n_frequency):
                amplitude_name = "amplitude(G{:d},F{:d})".format(g+1, f+1)
                self.amplitude[:, g, f] = self.model.get_array(name=amplitude_name)

                phase_name = "phase(G{:d},F{:d})".format(g+1, f+1)
                self.phase[:, g, f] = self.model.get_array(name=phase_name)
        
        self.noise = np.empty(
            shape = (self.n_cell, self.n_energy_group, self.n_frequency),
            dtype = np.complex128
        )
        self.noise.real = self.amplitude * np.cos(self.phase)
        self.noise.imag = self.amplitude * np.sin(self.phase)
    
    def to_time_domain(self) -> None:
        self.dflux = np.zeros(
            shape = (self.n_cell, self.n_energy_group, self.n_frequency),
            dtype = np.complex128
        )

        for c in range(self.n_cell):
            for g in range(self.n_energy_group):
                self.dflux[c, g, :] = np.fft.ifft(
                    self.noise[c, g, :],
                    n = self.n_frequency
                )
        
        self.flux = self.static_flux + np.sum(self.dflux, axis=2)
    
    def to_vtk(self, file_name = None) -> None:
        if file_name is None:
            file_name = "noise_td.vtk"
        
        model: pv.UnstructuredGrid = pv.read(self.path)

        field_name_to_remove = [
            field_name
            for field_name in model.array_names
            if field_name not in self._FIELD_NAME_TO_KEEP
        ]
        for field_name in field_name_to_remove:
            del model.cell_data[field_name]

        for t in range(self.n_frequency):
            fpath = os.path.join(
                os.path.dirname(self.path),
                file_name.replace('.', "{:d}.".format(t))
            )

            model_ = model.copy()
            for g in range(self.n_energy_group):
                flux_name = "flux(G{:d})".format(g+1)
                model_.cell_data[flux_name] = self.flux[:, g]
        
            model_.save(filename=fpath)
            del model_

if __name__ == "__main__":
    dataset = FrequencyDomainDataset(
        n_energy_group = 2,
        n_cell = 400,
        n_frequency = 1
    )
    dataset.read(path="noise/output/noise/noise.vtk")
    dataset.to_time_domain()
    dataset.to_vtk()
