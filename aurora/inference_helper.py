import dataclasses
from typing import Callable, List, Union, Tuple
from pathlib import Path
from datetime import timedelta
from copy import deepcopy

from matplotlib.style import use
import numpy as np
import torch
import torch.utils.checkpoint
import xarray as xr

from aurora import Aurora, Batch, Metadata
from aurora.model.decoder import Perceiver3DDecoder
from aurora.model.encoder import Perceiver3DEncoder
from aurora.model.fourier import lead_time_expansion

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
            return torch.from_numpy(x[[self.time_idx]][None][..., ::-1, :].copy()).to(torch.float16)

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


def preprocess_batch(model: Aurora, batch: Batch, device:str):
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.normalise(surf_stats=model.surf_stats)
    batch = batch.crop(patch_size=model.patch_size)
    batch = batch.to(device)
    return batch


def encoder_forward(model: Aurora, batch: Batch, device:str):
    '''forward pass of encoder'''

    # Move encoder to device
    encoder = model.encoder.to(device)

    H, W = batch.spatial_shape
    patch_res = (
        encoder.latent_levels,
        H // encoder.patch_size,
        W // encoder.patch_size,
    )

    # Insert batch and history dimension for static variables.
    B, T = next(iter(batch.surf_vars.values())).shape[:2]
    batch = dataclasses.replace(
        batch,
        static_vars={k: v[None, None].repeat(B, T, 1, 1) for k, v in batch.static_vars.items()},
    )

    x = encoder(
        batch,
        lead_time=timedelta(hours=6),
    )
    
    return x, patch_res


def backbone_encoder_layers_forward(model: Aurora, x: torch.Tensor, patch_res: tuple[int, int, int], rollout_step: int, device:str):
    lead_time = timedelta(hours=6)
    all_enc_res, padded_outs = model.backbone.get_encoder_specs(patch_res)

    lead_hours = lead_time / timedelta(hours=1)
    lead_times = lead_hours * torch.ones(x.shape[0], dtype=torch.float32, device=x.device)
    c = model.backbone.time_mlp.to(device)(lead_time_expansion(lead_times, model.backbone.embed_dim).to(dtype=x.dtype))

    skips = []
    for i, layer in enumerate(model.backbone.encoder_layers):
        x, x_unscaled = layer.to(device)(x, c, all_enc_res[i], rollout_step=rollout_step)
        skips.append(x_unscaled)

    return x, skips, c, all_enc_res, padded_outs


class BackboneDecoderLayers(torch.nn.Module):
    def __init__(self, decoder_layers, num_decoder_layers):
        super().__init__()
        self.num_decoder_layers = num_decoder_layers
        self.decoder_layers = torch.nn.ModuleList()
        for layer in decoder_layers:
            self.decoder_layers.append(deepcopy(layer))

    def forward(self, x, skips, c, all_enc_res, padded_outs, rollout_step):
        for i, layer in enumerate(self.decoder_layers):
            index = self.num_decoder_layers - i - 1
            x, _ = layer(
                x,
                c,
                all_enc_res[index],
                padded_outs[index - 1],
                rollout_step=rollout_step,
            )
            # x = torch.utils.checkpoint.checkpoint(layer.forward, x, c, all_enc_res[index], padded_outs[index - 1], rollout_step=roll, use_reentrant=False)

            if 0 < i < self.num_decoder_layers - 1:
                # For the intermediate stages, we use additive skip connections.
                x = x + skips[index - 1]
            elif i == self.num_decoder_layers - 1:
                # For the last stage, we perform concatentation like in Pangu.
                x = torch.cat([x, skips[0]], dim=-1)
        return x


def decoder_forward(decoder: Perceiver3DDecoder, x :torch.Tensor, batch: Batch, patch_res: tuple[int, int, int], surf_stats):
    # x = decoder.forward(
    #     x,
    #     batch,
    #     lead_time=timedelta(hours=6),
    #     patch_res=patch_res,
    # )
    x = torch.utils.checkpoint.checkpoint(decoder.forward, x, batch, patch_res, timedelta(hours=6), use_reentrant=False)

    x = dataclasses.replace(
        x,
        static_vars={k: v[0, 0] for k, v in batch.static_vars.items()},
    )

    # Insert history dimension in prediction. The time should already be right.
    x = dataclasses.replace(
        x,
        surf_vars={k: v[:, None] for k, v in x.surf_vars.items()},
        atmos_vars={k: v[:, None] for k, v in x.atmos_vars.items()},
    )

    return x.unnormalise(surf_stats=surf_stats)
