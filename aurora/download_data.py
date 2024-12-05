from pathlib import Path

import fsspec
import xarray as xr

def download_for_day(day:str, download_path: Path) -> None:
    # Data will be downloaded here.
    # download_path = Path("/workspace/data")
    download_path = download_path / day

    download_path = download_path.expanduser()
    download_path.mkdir(parents=True, exist_ok=True)

    # We will download from Google Cloud.
    url = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
    ds = xr.open_zarr(fsspec.get_mapper(url), chunks=None)

    # Download the surface-level variables. We write the downloaded data to another file to cache.
    if not (download_path / f"{day}-surface-level.nc").exists():
        surface_vars = [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure",
        ]
        ds_surf = ds[surface_vars].sel(time=day).compute()
        ds_surf.to_netcdf(str(download_path / f"{day}-surface-level.nc"))
    # print("\tSurface-level variables downloaded!")

    # Download the atmospheric variables. We write the downloaded data to another file to cache.
    if not (download_path / f"{day}-atmospheric.nc").exists():
        atmos_vars = [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "specific_humidity",
            "geopotential",
        ]
        ds_atmos = ds[atmos_vars].sel(time=day).compute()
        ds_atmos.to_netcdf(str(download_path / f"{day}-atmospheric.nc"))
    # print("\tAtmos-level variables downloaded!")
    