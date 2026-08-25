"""Reference workload timing and peak-RSS measurements for the audit."""

from __future__ import annotations

import json
import resource
import time
import warnings

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg

from pyskylumos.sky_models import (
    AsymmetricQuartic,
    Berry,
    DepolarizedRayleigh,
    Pan,
    QuEEN,
    Rayleigh,
)

LOCATION = EarthLocation(lat=53.4 * deg, lon=-2.96 * deg, height=50)


def benchmark(model_class, rows: int, columns: int, time_count: int) -> dict[str, object]:
    times = Time([f"2026-06-21T{hour % 24:02d}:00:00" for hour in range(time_count)])
    azimuths = np.tile(np.linspace(-180.0, 180.0, columns, dtype=np.float32), (rows, 1))
    altitudes = np.tile(np.linspace(0.1, 89.9, rows, dtype=np.float32)[:, None], (1, columns))
    sun = SkyCoord(
        az=np.linspace(100.0, 260.0, time_count) * deg,
        alt=np.linspace(10.0, 60.0, time_count) * deg,
        frame=AltAz(obstime=times, location=LOCATION),
    )
    kwargs = {"out_of_range": "ignore"} if model_class in (AsymmetricQuartic, Pan, QuEEN) else {}
    model = model_class(
        times=times,
        observation_location=LOCATION,
        azimuths=azimuths,
        altitudes=altitudes,
        **kwargs,
    )
    before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = model.simulate_sky(cie_sky_type=4, sun_position=sun)
    elapsed = time.perf_counter() - started
    after_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "seconds": elapsed,
        "peak_rss_before_bytes": int(before_rss),
        "peak_rss_after_bytes": int(after_rss),
        "output_shape": list(output[0].shape),
        "finite_dop": int(np.isfinite(output[0]).sum()),
        "dop_min": float(np.nanmin(output[0])),
        "dop_max": float(np.nanmax(output[0])),
    }


def main() -> None:
    workloads = ((64, 64, 1), (512, 512, 1), (128, 128, 24))
    results = {}
    for rows, columns, time_count in workloads:
        workload = f"{rows}x{columns}x{time_count}"
        results[workload] = {}
        for model_class in (
            Rayleigh,
            DepolarizedRayleigh,
            AsymmetricQuartic,
            Berry,
            Pan,
            QuEEN,
        ):
            results[workload][model_class.__name__] = benchmark(
                model_class, rows, columns, time_count
            )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
