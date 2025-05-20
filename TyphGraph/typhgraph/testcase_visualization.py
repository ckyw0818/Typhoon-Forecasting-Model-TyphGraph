import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset(
    "./data/graphcast_dataset/source-era5_date-2022-01-01_res-0.25_levels-13_steps-01.nc",
    decode_timedelta=True  # FutureWarning 없애려면 명시해두세요
)
print(ds.data_vars)  # 변수 이름 확인

vars_to_plot = [
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential_at_surface",
    "specific_humidity",
]

for var in vars_to_plot:
    data = ds[var]
    for t in range(data.sizes["time"]):
        # level 차원은 중간값 선택
        sel = data.isel(time=t, level=(len(data.level)//2) if "level" in data.dims else 0)
        plt.figure(figsize=(6,4))
        sel.plot(cmap="RdBu_r" if sel.ndim==2 else "viridis")
        plt.title(f"{var} — time index {t}")
        plt.tight_layout()
        plt.show()
