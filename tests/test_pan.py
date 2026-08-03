import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import deg

from pyskylumos.sky_models.Pan import Pan


def test_anti_solar_singularity_metadata_uses_antipodal_root_distances():
    times = Time(["2026-01-01T12:00:00"])
    location = EarthLocation(lat=0 * deg, lon=0 * deg, height=0)
    simulator = Pan(
        times=times,
        observation_location=location,
        azimuths=np.array([[0.0]], dtype=np.float32),
        altitudes=np.array([[45.0]], dtype=np.float32),
    )
    sun_position = SkyCoord(
        az=[120.0] * deg,
        alt=[20.0] * deg,
        frame=AltAz(obstime=times, location=location),
    )

    result = simulator.simulate_sky(cie_sky_type=1, sun_position=sun_position)

    anti_sun_position = sun_position.directional_offset_by(
        position_angle=0 * deg,
        separation=180 * deg,
    )
    angle_between_sun_babinet = 42.53 * deg - 0.56 * sun_position.alt
    angle_between_sun_brewster = 37.34 * deg + 0.49 * sun_position.alt
    expected_above_anti_sun = anti_sun_position.directional_offset_by(
        position_angle=0 * deg,
        separation=angle_between_sun_brewster,
    )
    expected_below_anti_sun = anti_sun_position.directional_offset_by(
        position_angle=0 * deg,
        separation=-angle_between_sun_babinet,
    )

    np.testing.assert_allclose(result[12], expected_above_anti_sun.az.radian[..., None, None])
    np.testing.assert_allclose(result[13], expected_above_anti_sun.alt.radian[..., None, None])
    np.testing.assert_allclose(result[14], expected_below_anti_sun.az.radian[..., None, None])
    np.testing.assert_allclose(result[15], expected_below_anti_sun.alt.radian[..., None, None])
