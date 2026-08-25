"""Compare the released 0.0.6 wheel directly with the frozen regression arrays."""

from __future__ import annotations

import json
import sys

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg

from pyskylumos.sky_models import Berry, Pan, Rayleigh

try:
    from pyskylumos.sky_models import AsymmetricQuartic, DepolarizedRayleigh
except ImportError:  # The released 0.0.6 wheel predates this class.
    AsymmetricQuartic = None  # type: ignore[misc, assignment]
    DepolarizedRayleigh = None  # type: ignore[misc, assignment]


def main() -> None:
    golden = np.load(sys.argv[1])
    times = Time(["2026-06-21T12:00:00"])
    location = EarthLocation(lat=53.4 * deg, lon=-2.96 * deg, height=50)
    metrics = {}
    for golden_name, model_class in (("rayleigh", Rayleigh), ("berry", Berry), ("queen", Pan)):
        maxima = {"dop": 0.0, "aop": 0.0, "radiance": 0.0}
        for elevation in (5.0, 15.0, 27.0, 30.0, 45.0, 60.0):
            sun = SkyCoord(
                az=[137.0] * deg,
                alt=[elevation] * deg,
                frame=AltAz(obstime=times, location=location),
            )
            model = model_class(
                times=times,
                observation_location=location,
                azimuths=golden["azimuths"],
                altitudes=golden["altitudes"],
            )
            result = model.simulate_sky(cie_sky_type=4, sun_position=sun, altitude_min_clip=0.0)
            for quantity, index in (("dop", 0), ("aop", 1), ("radiance", 2)):
                delta = np.abs(
                    np.asarray(result[index]) - golden[f"{golden_name}_{elevation:g}_{quantity}"]
                )
                maxima[quantity] = max(maxima[quantity], float(np.nanmax(delta)))
        metrics[f"0.0.6_{model_class.__name__}_vs_{golden_name}_golden"] = maxima

    if DepolarizedRayleigh is not None:
        maxima = {"dop": 0.0, "aop": 0.0, "radiance": 0.0}
        for elevation in (5.0, 15.0, 27.0, 30.0, 45.0, 60.0):
            sun = SkyCoord(
                az=[137.0] * deg,
                alt=[elevation] * deg,
                frame=AltAz(obstime=times, location=location),
            )
            model = DepolarizedRayleigh(
                times=times,
                observation_location=location,
                azimuths=golden["azimuths"],
                altitudes=golden["altitudes"],
                depolarization_ratio=0.0,
            )
            result = model.simulate_sky(cie_sky_type=4, sun_position=sun, altitude_min_clip=0.0)
            for quantity, index in (("dop", 0), ("aop", 1), ("radiance", 2)):
                delta = np.abs(
                    np.asarray(result[index]) - golden[f"rayleigh_{elevation:g}_{quantity}"]
                )
                maxima[quantity] = max(maxima[quantity], float(np.nanmax(delta)))
        metrics["DepolarizedRayleigh_delta_zero_vs_0.0.6_rayleigh_golden"] = maxima

    if AsymmetricQuartic is not None:
        maxima = {"dop": 0.0, "aop": 0.0, "radiance": 0.0}
        for elevation in (5.0, 15.0, 27.0, 30.0, 45.0, 60.0):
            sun = SkyCoord(
                az=[137.0] * deg,
                alt=[elevation] * deg,
                frame=AltAz(obstime=times, location=location),
            )
            model = AsymmetricQuartic(
                times=times,
                observation_location=location,
                azimuths=golden["azimuths"],
                altitudes=golden["altitudes"],
                arago_offset="brewster",
                fourth_offset="babinet",
                normalisation="berry",
                out_of_range="ignore",
            )
            result = model.simulate_sky(cie_sky_type=4, sun_position=sun, altitude_min_clip=0.0)
            for quantity, index in (("dop", 0), ("aop", 1), ("radiance", 2)):
                delta = np.abs(
                    np.asarray(result[index]) - golden[f"queen_{elevation:g}_{quantity}"]
                )
                maxima[quantity] = max(maxima[quantity], float(np.nanmax(delta)))
        metrics["AsymmetricQuartic_paired_vs_0.0.6_pan_golden"] = maxima
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
