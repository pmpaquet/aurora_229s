from typing import Callable, List, Union, Tuple
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from aurora import Batch, Metadata

class InferenceBatcher:
    def __init__(self, base_date_list: List[str], data_path: Path) -> None:
        self.base_date_list = base_date_list[:]
        self.day = self.base_date_list.pop(0)
        self.data_path = data_path
        self.static_vars_ds = xr.open_dataset(data_path / "static.nc", engine="netcdf4")
        self.surf_vars_ds: xr.Dataset
        self.atmos_vars_ds: xr.Dataset
        self._load_date_files()

        # Variable names
        self.surf_vars_names = [
            ('2t', '2m_temperature'),
            ('10u', '10m_u_component_of_wind'),
            ('10v', '10m_v_component_of_wind'),
            ('msl', 'mean_sea_level_pressure'),
        ]
        self.static_vars_names = [
            ('z', 'z'),
            ('slt', 'slt'),
            ('lsm', 'lsm')
        ]
        self.atmos_vars_names = [
            ('t', 'temperature'),
            ('u', 'u_component_of_wind'),
            ('v', 'v_component_of_wind'),
            ('q', 'specific_humidity'),
            ('z', 'geopotential')

        ]

        self.time_idx: int
        self.features: Batch
        self.labels: Batch
        self._set_initial_feature_labels()


    def _load_date_files(self) -> None:
        self.surf_vars_ds = xr.open_dataset(
            self.data_path / self.day / f"{self.day}-surface-level.nc",
            engine="netcdf4"
        )
        self.atmos_vars_ds = xr.open_dataset(
            self.data_path / self.day / f"{self.day}-atmospheric.nc",
            engine="netcdf4"
        )

    def _set_initial_feature_labels(self) -> None:
        self.time_idx = 0 # Initialized to 0
        self.features = self._make_batch()
        self.time_idx += 1
        self.labels = self._make_batch()
        self.time_idx += 1 # Finish at 2 --> this is the next index to pull from

    def _increment_day(self) -> None:
        # quick and dirty
        days_in_month = {2:28} # 2022 not a leap year
        for i in [9, 4, 6, 11]:
            days_in_month[i] = 30
        # all else is 31

        y, m, d = [int(x) for x in self.day.split('-')]
        d += 1
        if d > days_in_month.get(m, 31):
            m += 1
            d = 1
        assert m <= 12, f'Month is greater than 12: {m}'

        self.day = f'{y}-{m:02}-{d:02}'

    def _update_internal_state(self) -> bool:
        # First, check if time_index (i) is valid
        if self.time_idx > 3:
            # need to reload new date
            self._increment_day()

            # check whether the directory exists
            if (self.data_path / self.day).is_dir():
                # If next day directory exists, load from there
                self.time_idx = 0
                self._load_date_files()
            elif len(self.base_date_list):
                # If next day not found, need to jump to new base date
                self.day = self.base_date_list.pop(0)
                self._load_date_files()
                # Need to initialize new states for features and labels
                self._set_initial_feature_labels()
            else:
                return False

        return True
    
    def _update_features_and_labels(self) -> None:
        '''Updates internal state of features and labels'''
        # sh = short-hand, lh = long-hand
        self.features = Batch(
            surf_vars={
                sh:torch.concat((self.features.surf_vars[sh][:,[-1]], self.labels.surf_vars[sh][:, [-1]]), dim=1)
                for sh,_ in self.surf_vars_names
            },
            static_vars=self.labels.static_vars,
            atmos_vars={
                sh:torch.concat((self.features.atmos_vars[sh][:,[-1]], self.labels.atmos_vars[sh][:, [-1]]), dim=1)
                for sh,_ in self.atmos_vars_names
            },
            metadata=self.labels.metadata,
        )
        self.labels = self._make_batch()

    def _make_batch(self) -> Batch:
        def _prepare(x: np.ndarray) -> torch.Tensor:
            """Prepare a variable.

            This does the following things:
            * Select time indices `i` and `i - 1`.
            * Insert an empty batch dimension with `[None]`.
            * Flip along the latitude axis to ensure that the latitudes are decreasing.
            * Copy the data, because the data must be contiguous when converting to PyTorch.
            * Convert to PyTorch.
            """
            return torch.from_numpy(x[[self.time_idx]][None][..., ::-1, :].copy())

        return Batch(
            surf_vars={sh:_prepare(self.surf_vars_ds[lh].values) for sh,lh in self.surf_vars_names},
            static_vars={sh: torch.from_numpy(self.static_vars_ds[lh].values[0]) for sh,lh in self.static_vars_names},
            atmos_vars={sh: _prepare(self.atmos_vars_ds[lh].values) for sh,lh in self.atmos_vars_names},
            metadata=Metadata(
                # Flip the latitudes! We need to copy because converting to PyTorch, because the
                # data must be contiguous.
                lat=torch.from_numpy(self.surf_vars_ds.latitude.values[::-1].copy()),
                lon=torch.from_numpy(self.surf_vars_ds.longitude.values),
                # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
                # `datetime.datetime`s. Note that this needs to be a tuple of length one:
                # one value for every batch element.
                time=(self.surf_vars_ds.time.values.astype("datetime64[s]").tolist()[self.time_idx],),
                atmos_levels=tuple(int(level) for level in self.atmos_vars_ds.level.values),
            ),
        )

    def get_batch(self):
        is_valid_batch: bool = self._update_internal_state()

        if not is_valid_batch:
            return None, None
        else:
            self._update_features_and_labels()
            self.time_idx += 1
            return self.features, self.labels