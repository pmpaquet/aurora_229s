import dataclasses
from typing import Callable, List, Union, Tuple
from pathlib import Path
from datetime import timedelta
from copy import deepcopy
import shutil

from matplotlib.style import use
import numpy as np
import torch
import torch.utils.checkpoint
import xarray as xr

from aurora import Aurora, Batch, Metadata
from aurora.model.decoder import Perceiver3DDecoder
from aurora.model.encoder import Perceiver3DEncoder
from aurora.model.fourier import lead_time_expansion
from aurora.download_data import download_for_day


def increment_day(day:str) -> str:
    # quick and dirty
    days_in_month = {2:28} # 2022 not a leap year
    for i in [9, 4, 6, 11]:
        days_in_month[i] = 30
    # all else is 31

    y, m, d = [int(x) for x in day.split('-')]
    d += 1
    if d > days_in_month.get(m, 31):
        m += 1
        d = 1
    assert m <= 12, f'Month is greater than 12: {m}'

    return f'{y}-{m:02}-{d:02}'


class InferenceBatcher:
    def __init__(self, base_date_list: List[str], data_path: Path, max_n_days: int) -> None:
        self.base_date_list = base_date_list[:]
        self.day = self.base_date_list.pop(0)
        self.data_path = data_path
        self.static_vars_ds = xr.open_dataset(data_path / "static.nc", engine="netcdf4")
        self.surf_vars_ds: xr.Dataset
        self.atmos_vars_ds: xr.Dataset
        self.max_n_days = max_n_days
        self.n_days = 0
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

        if not (self.data_path / self.day / f"{self.day}-atmospheric.nc").is_file():
            download_for_day(self.day, self.data_path)

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
        self.day = increment_day(self.day)
        self.n_days += 1

    def _update_internal_state(self) -> bool:
        # First, check if time_index (i) is valid
        if self.time_idx > 3:
            # need to reload new date
            shutil.rmtree(str(self.data_path / self.day))
            self._increment_day()

            # check whether the directory exists
            if self.n_days < self.max_n_days:
                # If next day directory exists, load from there
                self.time_idx = 0
                self._load_date_files()
            elif len(self.base_date_list):
                # If next day not found, need to jump to new base date
                self.day = self.base_date_list.pop(0)
                self._load_date_files()
                # Need to initialize new states for features and labels
                self._set_initial_feature_labels()
                self.n_days = 0
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
            return torch.from_numpy(x[[self.time_idx]][None][..., ::-1, :].copy())#.to(torch.float16)

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


def preprocess_batch(model: Aurora, batch: Batch, device:str, norm:bool):
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    if norm:
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


def backbone_prep(backbone, x, patch_res, device):
    lead_time = timedelta(hours=6)
    all_enc_res, padded_outs = backbone.get_encoder_specs(patch_res)

    lead_hours = lead_time / timedelta(hours=1)
    lead_times = lead_hours * torch.ones(x.shape[0], dtype=torch.float32, device=x.device)
    c = backbone.time_mlp.to(device)(lead_time_expansion(lead_times, backbone.embed_dim).to(dtype=x.dtype))
    return c, all_enc_res, padded_outs


class BackboneEncoderLayers(torch.nn.Module):
    def __init__(self, encoder_layers):
        super().__init__()
        self.encoder_layers = torch.nn.ModuleList()
        for layer in encoder_layers:
            self.encoder_layers.append(deepcopy(layer))

    def forward(self, x, c, all_enc_res, rollout_step):
        skips = []
        for i, layer in enumerate(self.encoder_layers):
            x, x_unscaled = layer(x, c, all_enc_res[i], rollout_step=rollout_step)
            skips.append(x_unscaled)
        return x, skips

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


# ------------------------------------------------------------
# Evaluation inference helpers
# ------------------------------------------------------------

def get_vars_names_wts():
    surf_vars_names_wts = [
        ('2t', '2m_temperature', 3.0),
        ('10u', '10m_u_component_of_wind', 0.77),
        ('10v', '10m_v_component_of_wind', 0.66),
        ('msl', 'mean_sea_level_pressure', 1.5),
    ]
    atmos_vars_names_wts = [
        ('t', 'temperature', 1.7),
        ('u', 'u_component_of_wind', 0.87),
        ('v', 'v_component_of_wind', 0.6),
        ('q', 'specific_humidity', 0.78),
        ('z', 'geopotential', 2.8)
    ]
    return surf_vars_names_wts, atmos_vars_names_wts


class RolloutInferenceBatcher(InferenceBatcher):
    def __init__(self, start_day: str, data_path: Path, max_n_days: int) -> None:
        self.day = start_day
        self.data_path = data_path
        # SUPER HACKY
        self.base_date_list = [] # ALREADY NOTHING LEFT!!!
        self.static_vars_ds = xr.open_dataset(data_path / "static.nc", engine="netcdf4")
        self.surf_vars_ds: xr.Dataset
        self.atmos_vars_ds: xr.Dataset
        self.max_n_days = max_n_days
        self.n_days = 0
        self._load_date_files()

        # Variable names
        self.static_vars_names = [
            ('z', 'z'),
            ('slt', 'slt'),
            ('lsm', 'lsm')
        ]
        self.surf_vars_names_wts = [
            ('2t', '2m_temperature', 3.0),
            ('10u', '10m_u_component_of_wind', 0.77),
            ('10v', '10m_v_component_of_wind', 0.66),
            ('msl', 'mean_sea_level_pressure', 1.5),
        ]
        self.atmos_vars_names_wts = [
            ('t', 'temperature', 1.7),
            ('u', 'u_component_of_wind', 0.87),
            ('v', 'v_component_of_wind', 0.6),
            ('q', 'specific_humidity', 0.78),
            ('z', 'geopotential', 2.8)
        ]

        # HACKY
        self.surf_vars_names = [
            ('2t', '2m_temperature'),
            ('10u', '10m_u_component_of_wind'),
            ('10v', '10m_v_component_of_wind'),
            ('msl', 'mean_sea_level_pressure'),
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
        self._update_features_and_labels() # CALL IN BEGINNING


    def rollout_update_features_and_labels(self, pred:Batch) -> None:
        '''Updates internal state of features and labels'''
        # sh = short-hand, lh = long-hand
        # self.features = Batch(
        #     surf_vars={
        #         sh:torch.concat((self.features.surf_vars[sh][:,[-1]], self.labels.surf_vars[sh][:, [-1]]), dim=1)
        #         for sh,_ in self.surf_vars_names
        #     },
        #     static_vars=self.labels.static_vars,
        #     atmos_vars={
        #         sh:torch.concat((self.features.atmos_vars[sh][:,[-1]], self.labels.atmos_vars[sh][:, [-1]]), dim=1)
        #         for sh,_ in self.atmos_vars_names
        #     },
        #     metadata=self.labels.metadata,
        # )
        # Add the appropriate history so the model can be run on the prediction.
        # for k,v in pred.surf_vars.items():
        #     print(k, v.shape, self.features.surf_vars[k].shape)
        # print(pred.shape)
        self.features = dataclasses.replace(
            pred,
            surf_vars={
                # k: torch.cat([self.features.surf_vars[k][:, 1:], v], dim=1)
                k: torch.cat([self.features.surf_vars[k][:, :, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                # k: torch.cat([self.features.atmos_vars[k][:, 1:], v], dim=1)
                k: torch.cat([self.features.atmos_vars[k][:, :, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )
        self.labels = self._make_batch()

    def get_batch(self):
        is_valid_batch: bool = self._update_internal_state()

        if not is_valid_batch:
            return None, None
        else:
            self.time_idx += 1
            return self.features, self.labels
        # User needs to call rollout update function
