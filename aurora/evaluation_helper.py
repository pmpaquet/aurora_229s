from cProfile import label
from pathlib import Path
import shutil

import torch
import numpy as np
import pandas as pd
import xarray as xr

from aurora.model import aurora
from aurora import inference_helper, Batch, download_data, Metadata, rollout


def cleanup_download_dir(download_path: Path) -> None:
    for item in download_path.iterdir():
        if item.stem.startswith('2022-'):
            shutil.rmtree(str(item))


def np_mae(x: np.ndarray, y: np.ndarray):
    return np.mean(np.abs(x - y))


def sameday_batch_helper(model, day: str, download_path: Path, i: int) -> Batch:
    if not (download_path / day / f"{day}-atmospheric.nc").is_file():
        download_data.download_for_day(day=day, download_path=download_path)

    static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
    surf_vars_ds = xr.open_dataset(download_path / day / f"{day}-surface-level.nc", engine="netcdf4")
    atmos_vars_ds = xr.open_dataset(download_path / day / f"{day}-atmospheric.nc", engine="netcdf4")

    def _prepare(x: np.ndarray) -> torch.Tensor:
        """Prepare a variable.

        This does the following things:
        * Select time indices `i` and `i - 1`.
        * Insert an empty batch dimension with `[None]`.
        * Flip along the latitude axis to ensure that the latitudes are decreasing.
        * Copy the data, because the data must be contiguous when converting to PyTorch.
        * Convert to PyTorch.
        """
        return torch.from_numpy(x[[i - 1, i]][None][..., ::-1, :].copy())


    batch = Batch(
        surf_vars={
            "2t": _prepare(surf_vars_ds["2m_temperature"].values),
            "10u": _prepare(surf_vars_ds["10m_u_component_of_wind"].values),
            "10v": _prepare(surf_vars_ds["10m_v_component_of_wind"].values),
            "msl": _prepare(surf_vars_ds["mean_sea_level_pressure"].values),
        },
        static_vars={
            # The static variables are constant, so we just get them for the first time. They
            # don't need to be flipped along the latitude dimension, because they are from
            # ERA5.
            "z": torch.from_numpy(static_vars_ds["z"].values[0]),
            "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
        },
        atmos_vars={
            "t": _prepare(atmos_vars_ds["temperature"].values),
            "u": _prepare(atmos_vars_ds["u_component_of_wind"].values),
            "v": _prepare(atmos_vars_ds["v_component_of_wind"].values),
            "q": _prepare(atmos_vars_ds["specific_humidity"].values),
            "z": _prepare(atmos_vars_ds["geopotential"].values),
        },
        metadata=Metadata(
            # Flip the latitudes! We need to copy because converting to PyTorch, because the
            # data must be contiguous.
            lat=torch.from_numpy(surf_vars_ds.latitude.values[::-1].copy()),
            lon=torch.from_numpy(surf_vars_ds.longitude.values),
            # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
            # `datetime.datetime`s. Note that this needs to be a tuple of length one:
            # one value for every batch element.
            time=(surf_vars_ds.time.values.astype("datetime64[s]").tolist()[i],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.level.values),
        ),
    )
    return batch


def same_day_eval(model, day: str, download_path: Path, device:str) -> pd.DataFrame:
    # 'i' is index of last timestep --> 1 is training, 3 is testing (for 2 rollout steps)
    trn_batch = sameday_batch_helper(model=model, day=day, download_path=download_path, i=1)

    # ----------------------------------------------------------------------------
    # Inference
    model.eval()
    model = model.to(device)

    with torch.inference_mode():
        preds = [pred.to("cpu") for pred in rollout(model, trn_batch, steps=2)]
    model = model.to("cpu")

    # Evaluation
    tst_batch = sameday_batch_helper(model=model, day=day, download_path=download_path, i=3)
    tst_batch = inference_helper.preprocess_batch(model=model, batch=tst_batch, device='cpu', norm=False)
    surf_vars_names_wts, atmos_vars_names_wts = inference_helper.get_vars_names_wts()
    results = {var:[] for var,_,_ in surf_vars_names_wts+atmos_vars_names_wts}

    for i in range(2):
        # print(preds[i].surf_vars['2t'].shape)
        # print(tst_batch.surf_vars['2t'].shape)
        for sh,lh,wt in surf_vars_names_wts:
            results[sh].append(
                wt * np_mae(
                    preds[i].surf_vars[sh][0, 0].numpy(),
                    # batch.atmos_vars[sh][2 + i, 0].numpy(),
                    tst_batch.surf_vars[sh][0, i].numpy(),
                )
            )
        for sh,lh,wt in atmos_vars_names_wts:
            results[sh].append(
                wt * np_mae(
                    preds[i].atmos_vars[sh][0, 0].numpy(),
                    tst_batch.atmos_vars[sh][0, i].numpy(),
                    # batch.atmos_vars[sh][2 + i, 0].numpy(),
                )
            )

    results_df = pd.DataFrame(results)
    results_df["multitask"] = np.sum(results_df.values, axis=1)
    results_df["Day"] = [day, day]
    results_df["Time"] = ['12:00:00', '18:00:00']

    cleanup_download_dir(download_path=download_path)

    return results_df


def multi_day_eval(model, day: str, download_path: Path, max_n_days:int, device:str, verbose:bool) -> pd.DataFrame:

    batcher = inference_helper.RolloutInferenceBatcher(start_day=day, data_path=download_path, max_n_days=max_n_days)
    
    model.eval()
    model = model.to(device)
    
    surf_vars_names_wts, atmos_vars_names_wts = inference_helper.get_vars_names_wts()
    results = {var:[] for var,_,_ in surf_vars_names_wts+atmos_vars_names_wts}
    results['TimeIndex'] = []
    results['Day'] = []

    while True:
        results['Day'].append(batcher.day)
        results['TimeIndex'].append(batcher.time_idx)
        try:
            batch, labels = batcher.get_batch()
        except Exception as e:
            print('\n', e, '\n')
            break
        if batch is None or labels is None:
            break
        if verbose:
            print(batcher.day, batcher.time_idx - 1)

        batch = inference_helper.preprocess_batch(model=model, batch=batch, device=device, norm=True)
        torch.cuda.empty_cache()

        p = next(model.parameters())
        labels = labels.type(p.dtype)
        labels = labels.crop(model.patch_size)

        with torch.inference_mode():
            preds = model.forward(batch).to('cpu')
        torch.cuda.empty_cache()

        # Append loss
        is_loss_nan = False
        for sh,_,wt in surf_vars_names_wts:
            loss = wt * np_mae(
                preds.surf_vars[sh][0, 0].numpy(),
                labels.surf_vars[sh][0, 0].numpy(),
            )
            is_loss_nan = is_loss_nan or bool(np.isnan(loss))
            results[sh].append(loss)
        for sh,_,wt in atmos_vars_names_wts:
            loss = wt * np_mae(
                preds.atmos_vars[sh][0, 0].numpy(),
                labels.atmos_vars[sh][0, 0].numpy(),
            )
            is_loss_nan = is_loss_nan or bool(np.isnan(loss))
            results[sh].append(loss)

        # Fix if isnan
        if is_loss_nan:
            for sh,_,_ in surf_vars_names_wts:
                preds.surf_vars[sh] = torch.where(torch.isnan(preds.surf_vars[sh]), preds.surf_vars[sh], labels.surf_vars[sh])
            for sh,_,_ in atmos_vars_names_wts:
                preds.atmos_vars[sh] = torch.where(torch.isnan(preds.atmos_vars[sh]), preds.atmos_vars[sh], labels.atmos_vars[sh])

        # UPDATE BATCHER
        batcher.rollout_update_features_and_labels(preds)
        cleanup_download_dir(download_path=download_path)

    results_df = pd.DataFrame(results)
    results_df['multitask'] = np.sum(results_df[[c for c in results_df.columns if not c in ['Day', 'TimeIndex']]].values, axis=1)
    cleanup_download_dir(download_path=download_path)
    return results_df

def multi_day_eval_v2(model, day: str, download_path: Path, max_n_days:int, device:str, verbose:bool) -> pd.DataFrame:

    batcher = inference_helper.RolloutInferenceBatcher(start_day=day, data_path=download_path, max_n_days=max_n_days)
    
    model.eval()
    model = model.to(device)
    
    surf_vars_names_wts, atmos_vars_names_wts = inference_helper.get_vars_names_wts()
    results = {var:[] for var,_,_ in surf_vars_names_wts+atmos_vars_names_wts}
    results['TimeIndex'] = []
    results['Day'] = []

    batch, labels = batcher.get_batch()
    n_steps = (max_n_days * 4) - 2
    with torch.inference_mode():
        for preds in rollout(model, batch, steps=n_steps):
            preds = preds.to('cpu')
            labels = inference_helper.preprocess_batch(model=model, batch=labels, device='cpu', norm=False)
            
            results['Day'].append(batcher.day)
            results['TimeIndex'].append(batcher.time_idx)

            # Append loss
            is_loss_nan = False
            for sh,_,wt in surf_vars_names_wts:
                loss = wt * np_mae(
                    preds.surf_vars[sh][0, 0].numpy(),
                    labels.surf_vars[sh][0, 0].numpy(),
                )
                is_loss_nan = is_loss_nan or bool(np.isnan(loss))
                results[sh].append(loss)
            for sh,_,wt in atmos_vars_names_wts:
                loss = wt * np_mae(
                    preds.atmos_vars[sh][0, 0].numpy(),
                    labels.atmos_vars[sh][0, 0].numpy(),
                )
                is_loss_nan = is_loss_nan or bool(np.isnan(loss))
                results[sh].append(loss)

            # Fix if isnan
            if is_loss_nan:
                for sh,_,_ in surf_vars_names_wts:
                    preds.surf_vars[sh] = torch.where(torch.isnan(preds.surf_vars[sh]), preds.surf_vars[sh], labels.surf_vars[sh])
                for sh,_,_ in atmos_vars_names_wts:
                    preds.atmos_vars[sh] = torch.where(torch.isnan(preds.atmos_vars[sh]), preds.atmos_vars[sh], labels.atmos_vars[sh])

            # UPDATE BATCHER
            _, labels = batcher.get_batch()

    results_df = pd.DataFrame(results)
    results_df['multitask'] = np.sum(results_df[[c for c in results_df.columns if not c in ['Day', 'TimeIndex']]].values, axis=1)
    try:
        cleanup_download_dir(download_path=download_path)
    except:
        pass
    return results_df


def label_batch_helper(day: str, download_path: Path) -> Batch:
    if not (download_path / day / f"{day}-atmospheric.nc").is_file():
        download_data.download_for_day(day=day, download_path=download_path)

    static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
    surf_vars_ds = xr.open_dataset(download_path / day / f"{day}-surface-level.nc", engine="netcdf4")
    atmos_vars_ds = xr.open_dataset(download_path / day / f"{day}-atmospheric.nc", engine="netcdf4")

    def _prepare(x: np.ndarray) -> torch.Tensor:
        """Prepare a variable.

        This does the following things:
        * Select time indices `i` and `i - 1`.
        * Insert an empty batch dimension with `[None]`.
        * Flip along the latitude axis to ensure that the latitudes are decreasing.
        * Copy the data, because the data must be contiguous when converting to PyTorch.
        * Convert to PyTorch.
        """
        return torch.from_numpy(x[[0, 1, 2, 3]][None][..., ::-1, :].copy())


    batch = Batch(
        surf_vars={
            "2t": _prepare(surf_vars_ds["2m_temperature"].values),
            "10u": _prepare(surf_vars_ds["10m_u_component_of_wind"].values),
            "10v": _prepare(surf_vars_ds["10m_v_component_of_wind"].values),
            "msl": _prepare(surf_vars_ds["mean_sea_level_pressure"].values),
        },
        static_vars={
            # The static variables are constant, so we just get them for the first time. They
            # don't need to be flipped along the latitude dimension, because they are from
            # ERA5.
            "z": torch.from_numpy(static_vars_ds["z"].values[0]),
            "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
        },
        atmos_vars={
            "t": _prepare(atmos_vars_ds["temperature"].values),
            "u": _prepare(atmos_vars_ds["u_component_of_wind"].values),
            "v": _prepare(atmos_vars_ds["v_component_of_wind"].values),
            "q": _prepare(atmos_vars_ds["specific_humidity"].values),
            "z": _prepare(atmos_vars_ds["geopotential"].values),
        },
        metadata=Metadata(
            # Flip the latitudes! We need to copy because converting to PyTorch, because the
            # data must be contiguous.
            lat=torch.from_numpy(surf_vars_ds.latitude.values[::-1].copy()),
            lon=torch.from_numpy(surf_vars_ds.longitude.values),
            # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
            # `datetime.datetime`s. Note that this needs to be a tuple of length one:
            # one value for every batch element.
            time=(surf_vars_ds.time.values.astype("datetime64[s]").tolist()[3],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.level.values),
        ),
    )
    return batch


def multi_day_eval_v3(model, day: str, download_path: Path, max_n_days:int, device:str, verbose:bool) -> pd.DataFrame:

    model.eval()
    model = model.to(device)
    
    surf_vars_names_wts, atmos_vars_names_wts = inference_helper.get_vars_names_wts()
    results = {var:[] for var,_,_ in surf_vars_names_wts+atmos_vars_names_wts}
    results['TimeIndex'] = []
    results['Day'] = []

    n_steps = (max_n_days * 4) - 2

    curr_day = day
    curr_idx = 2
    labels = label_batch_helper(day=curr_day, download_path=download_path)
    labels = inference_helper.preprocess_batch(model=model, batch=labels, device='cpu', norm=False)

    batch = sameday_batch_helper(model=model, day=day, download_path=download_path, i=1)

    with torch.inference_mode():
        for preds in rollout(model, batch, steps=n_steps):
            preds = preds.to('cpu')
            
            results['Day'].append(curr_day)
            results['TimeIndex'].append(curr_idx)

            # Append loss
            is_loss_nan = False
            for sh,_,wt in surf_vars_names_wts:
                loss = wt * np_mae(
                    preds.surf_vars[sh][0, 0].numpy(),
                    labels.surf_vars[sh][0, curr_idx].numpy(),
                )
                is_loss_nan = is_loss_nan or bool(np.isnan(loss))
                results[sh].append(loss)
            for sh,_,wt in atmos_vars_names_wts:
                loss = wt * np_mae(
                    preds.atmos_vars[sh][0, 0].numpy(),
                    labels.atmos_vars[sh][0, curr_idx].numpy(),
                )
                is_loss_nan = is_loss_nan or bool(np.isnan(loss))
                results[sh].append(loss)

            # Fix if isnan
            if is_loss_nan:
                for sh,_,_ in surf_vars_names_wts:
                    preds.surf_vars[sh] = torch.where(torch.isnan(preds.surf_vars[sh]), preds.surf_vars[sh], labels.surf_vars[sh])
                for sh,_,_ in atmos_vars_names_wts:
                    preds.atmos_vars[sh] = torch.where(torch.isnan(preds.atmos_vars[sh]), preds.atmos_vars[sh], labels.atmos_vars[sh])

            # UPDATE BATCHER
            curr_idx += 1
            if curr_idx >= 4:
                curr_idx = 0
                curr_day = inference_helper.increment_day(day)
                labels = label_batch_helper(day=curr_day, download_path=download_path)
                labels = inference_helper.preprocess_batch(model=model, batch=labels, device='cpu', norm=False)
            try:
                cleanup_download_dir(download_path=download_path)
            except:
                pass

    results_df = pd.DataFrame(results)
    results_df['multitask'] = np.sum(results_df[[c for c in results_df.columns if not c in ['Day', 'TimeIndex']]].values, axis=1)
    try:
        cleanup_download_dir(download_path=download_path)
    except:
        pass
    return results_df
