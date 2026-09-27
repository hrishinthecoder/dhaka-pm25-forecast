# Data Dictionary — Dhaka Air Quality Data Collection

Every variable preserved by this pipeline, grouped by source. Units mirror what
the upstream API returns; we do not convert or normalise them at collection
time.

> Time convention: every tabular file carries `datetime_utc` (UTC, ISO 8601)
> plus a `datetime_local` column in `Asia/Dhaka` (UTC+6, no DST).

---

## 1. OpenAQ v3 — `data/raw/openaq/`

Surface in-situ measurements from monitoring stations within
`station_radius_km` of the Dhaka centre.

| Column | Description | Units |
|---|---|---|
| `datetime_utc` | Period start (UTC) | ISO timestamp |
| `datetimeTo_utc` | Period end (UTC) | ISO timestamp |
| `value` | Measurement value | per `units` |
| `parameter` | Pollutant name | pm25 / pm10 / o3 / no2 / so2 / co |
| `units` | Reporting unit string | e.g. µg/m³, ppm |
| `coverage_expected_minutes` | Expected sample interval | minutes |
| `coverage_observed_count` | Samples in the aggregation | count |
| `location_id`, `location_name`, `sensor_id` | OpenAQ identifiers | int / str |
| `latitude`, `longitude` | Sensor coordinates | degrees |

Auxiliary files: `_locations.json` (full discovery dump), `_sensors.csv` (the
sensor index used).

## 2. Open-Meteo Historical Weather (ERA5) — `data/raw/open_meteo_weather/`

Hourly reanalysis at the city centre. ERA5 archive begins 1940-01-01.

| Variable | Description | Units |
|---|---|---|
| `temperature_2m` | 2 m air temperature | °C |
| `relative_humidity_2m` | 2 m relative humidity | % |
| `dew_point_2m` | 2 m dew point temperature | °C |
| `wind_speed_10m` | 10 m wind speed | m/s |
| `wind_gusts_10m` | 10 m wind gusts | m/s |
| `wind_direction_10m` | 10 m wind direction | degrees from N |
| `surface_pressure` | Surface pressure | hPa |
| `precipitation` | Total precipitation (rain + snow) | mm |
| `rain` | Liquid precipitation | mm |
| `cloud_cover` | Total cloud cover | % |
| `boundary_layer_height` | Planetary boundary layer height | m |
| `shortwave_radiation` | Downward shortwave radiation | W/m² |
| `et0_fao_evapotranspiration` | FAO reference ET₀ | mm |
| `vapour_pressure_deficit` | VPD | kPa |

## 3. Open-Meteo Air Quality (CAMS) — `data/raw/open_meteo_air_quality/`

Hourly CAMS global reanalysis surface concentrations at the city centre.

| Variable | Description | Units |
|---|---|---|
| `pm2_5` | PM₂.₅ mass concentration | µg/m³ |
| `pm10` | PM₁₀ mass concentration | µg/m³ |
| `carbon_monoxide` | Surface CO | µg/m³ |
| `nitrogen_dioxide` | Surface NO₂ | µg/m³ |
| `sulphur_dioxide` | Surface SO₂ | µg/m³ |
| `ozone` | Surface O₃ | µg/m³ |
| `dust` | Surface dust | µg/m³ |
| `aerosol_optical_depth` | AOD 550 nm | unitless |
| `ammonia` | Surface NH₃ | µg/m³ |
| `uv_index` | UV index | unitless |

## 4. NASA POWER — `data/raw/nasa_power/`

Independent hourly meteorology for cross-checking ERA5.

| Variable | Description | Units |
|---|---|---|
| `T2M` | 2 m temperature | °C |
| `RH2M` | 2 m relative humidity | % |
| `WS10M` | 10 m wind speed | m/s |
| `WD10M` | 10 m wind direction | degrees |
| `PS` | Surface pressure | kPa |
| `PRECTOTCORR` | Corrected total precipitation | mm/hour |
| `ALLSKY_SFC_SW_DWN` | All-sky downward SW radiation | W/m² |

Sentinel value `-999` is converted to NaN. No further imputation.

## 5. NASA FIRMS — `data/raw/nasa_firms/`

Active-fire / thermal anomaly detections inside the configured bounding box,
one parquet per sensor (MODIS Terra/Aqua, VIIRS SNPP, VIIRS NOAA-20).

| Column | Description |
|---|---|
| `latitude`, `longitude` | Detection centroid (deg) |
| `brightness` | Brightness temperature (K) — MODIS |
| `bright_ti4`, `bright_ti5` | VIIRS brightness temperatures (K) |
| `scan`, `track` | Pixel size along scan/track (km) |
| `acq_date` | Acquisition date (UTC) |
| `acq_time` | Acquisition time HHMM (UTC) |
| `satellite` | Platform code |
| `confidence` | Detection confidence |
| `version` | Algorithm version |
| `frp` | Fire radiative power (MW) |
| `daynight` | D / N flag |
| `sensor` | Sensor stream (added by this pipeline) |

## 6. CAMS EAC4 — `data/raw/cams_eac4/`

Per-year NetCDF for the Dhaka bbox + a centre-point CSV per year.

Variables: `particulate_matter_2.5um`, `particulate_matter_10um`,
`total_aerosol_optical_depth_550nm`, surface `nitrogen_dioxide`,
`sulphur_dioxide`, `carbon_monoxide`, `ozone`. Units are mass mixing ratios
(kg/kg) or AOD (unitless); see the NetCDF attributes for the authoritative
unit string per variable.

## 7. MERRA-2 M2T1NXAER — `data/raw/merra2/`

Monthly NetCDF granules + a monthly centre-point CSV.

| Variable | Description | Units |
|---|---|---|
| `DUSMASS25` | Dust surface mass concentration (PM₂.₅ fraction) | kg/m³ |
| `OCSMASS` | Organic carbon surface mass concentration | kg/m³ |
| `BCSMASS` | Black carbon surface mass concentration | kg/m³ |
| `SO4SMASS` | Sulfate surface mass concentration | kg/m³ |
| `SSSMASS25` | Sea-salt surface mass concentration (PM₂.₅) | kg/m³ |
| `TOTEXTTAU` | Total aerosol extinction AOT [550 nm] | unitless |

## 8. Google Earth Engine — `data/raw/gee/`

Per-scene reductions over the Dhaka bbox + city-centre point.

Sentinel-5P (start 2018-07): tropospheric `NO2`, `SO2`, `CO`, `O3`
column number densities (mol/m²) and the Absorbing Aerosol Index (unitless).

MAIAC AOD (MODIS/061/MCD19A2_GRANULES, start 2000): `Optical_Depth_047`,
`Optical_Depth_055` (unitless, scale factor 0.001 — see collection docs).

Each column appears twice: `<band>__bbox_mean` and `<band>__point`.

## 9. ACAG surface PM₂.₅ — `data/raw/acag/`

Satellite-derived annual and monthly surface PM₂.₅ NetCDF grids
(van Donkelaar et al., Washington University ACAG). Variables and units follow
the published product (`GWRPM25`, µg/m³). Each download is subset to the
Dhaka bbox and a centre-point CSV is exported.

---

## Provenance

Every file written here has a corresponding entry in `data/manifest.json`
recording endpoint, query parameters, identifiers, row count and the file path.
The per-variable coverage summary lives in `reports/coverage_report.csv`.
