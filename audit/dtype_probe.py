"""Record public output dtypes for every Engine model path."""

from __future__ import annotations

import json
import warnings

from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg
from clean_install_smoke import make_engine


def main() -> None:
    time = Time(["2026-06-21T12:00:00"])
    location = EarthLocation(lat=53.4 * deg, lon=-2.96 * deg, height=50)
    sun = SkyCoord(az=[137] * deg, alt=[33] * deg, frame=AltAz(obstime=time, location=location))
    results = {}
    for model in (
        "RAYLEIGH",
        "DEPOLARIZED_RAYLEIGH",
        "ASYMMETRIC",
        "BERRY",
        "PAN",
        "QUEEN",
    ):
        engine = make_engine()
        azimuths, altitudes = engine.get_initial_azimuth_altitude(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values, names = engine.simulate_sky_polarization(
                model,
                location,
                time,
                4,
                altitudes,
                azimuths,
                sun_position=sun,
            )
        results[model] = {
            name: str(value.dtype) for name, value in zip(names, values, strict=False)
        }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
