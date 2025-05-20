import xarray as xr
import zarr

# 1) 원격 Zarr 스토어 래핑 (GCS anonymous access)
store = zarr.storage.FsspecStore.from_url(
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr",
    storage_options={"token": "anon"}
)

# 2) Lazy load
ds = xr.open_dataset(store, engine="zarr", consolidated=True)

# 3) 하루치(2020-01-01 00:00~23:00), 850 hPa, 세 변수만 선택
vars_to_use = ['u_component_of_wind', 'v_component_of_wind', 'temperature']
day   = "2020-01-01"
level = 850  # hPa 단위
ds_day = (
    ds[vars_to_use]
      .sel(
          level=level,
          time=slice(f"{day}T00:00", f"{day}T12:00"),
          method="nearest"
      )
      .load()
)

# 4) 저장 시 압축 해제 및 Zarr v2 메타데이터 포맷 지정
encoding = {var: {"compressor": None} for var in ds_day.data_vars}
out_path = f"./era5_850hpa_{day}.zarr"
ds_day.to_zarr(
    out_path,
    mode="w",
    encoding=encoding,
    zarr_format="2",       # v2 메타데이터 포맷
    consolidated=True      # 메타데이터 통합
)

print(f"✅ {day} 850 hPa u/v/temperature 데이터 저장 완료 → {out_path}")
