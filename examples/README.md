# Polarization model examples

These scripts compare each model's theoretical degree of polarization (DOP), angle of
polarization (AOP), and relative CIE radiance with the DOP and AOP reconstructed by the
simulated micro-polarizer sensor.

Install PySkyLumos with the plotting dependency from the repository root:

```console
python -m pip install -e ".[examples]"
```

Run any model independently:

```console
python examples/rayleigh.py
python examples/depolarized_rayleigh.py
python examples/asymmetric.py
python examples/berry.py
python examples/pan.py
python examples/queen.py
python examples/rotated_pose.py
```

The wrappers use the package defaults: ideal Rayleigh, molecularly depolarized Rayleigh,
default peak-normalized AsymmetricQuartic, Berry, published Pan, and QuEEN respectively.
`asymmetric.py` can take longer than the others because its default normalization performs a
deterministic full-sphere peak search.

Each script saves a PNG under `examples/output/` and then opens the figure. For a headless
run or a custom destination, use:

```console
MPLBACKEND=Agg python examples/queen.py --no-show --output /tmp/queen.png
```

All six model examples use the same deterministic setup: an **untilted**, upward-looking
128×128 equi-angle fisheye view, CIE sky type 4, a sun at 137° azimuth and 33° altitude,
and a seeded 12-bit noisy sensor. The theoretical AOP is therefore already aligned with the analyzer
frame. The 2×2 analyzer mosaic reconstructs 64×64 measured DOP and AOP fields. Radiance
appears only on the theoretical side because the normalized, auto-exposed sensor is not
radiometrically calibrated.

`rotated_pose.py` is the exception: it points the camera away from vertical and renders the
untilted and rotated AOP fields side by side, with their difference in a third panel.

## Posing the camera

Both pose representations keep the optical grid **sensor-local** and are supplied to
`Engine.simulate_sky_polarization`. They are mutually exclusive.

For a general three-dimensional orientation, pass a sensor-to-world rotation matrix. This is
the only way to express a twist about the optical axis, which the tilt arguments cannot
describe:

```python
values, names = engine.simulate_sky_polarization(
    # ...the same model, location, time, and sky arguments...
    azimuths=azimuths,
    altitudes=altitudes,
    sensor_to_world_rotation_matrix=rotation,  # (3, 3), maps sensor vectors to world
)
```

For the simpler case of a tilt about a horizontal axis, the two tilt arguments remain
available and are not deprecated:

```python
values, names = engine.simulate_sky_polarization(
    # ...the same model, location, time, and sky arguments...
    azimuths=azimuths,
    altitudes=altitudes,
    sensor_azimuthal_tilt_radians=0.0,
    sensor_tilt_angle_radians=0.35,
)
```

Either way, Engine rotates the rays into world coordinates and transports AOP back into the
rotated analyzer frame. The correction varies per pixel, so a uniform
`azimuth_rotation_angle` is not a substitute for it. The standalone `Engine.tilt_sensor`
helper rotates directions only and is not a substitute for this integrated path either.

For model derivations, provenance, Pan errata, the Pan–QuEEN relationship, and the complete
AsymmetricQuartic assumptions, tilted-AOP derivation and limitations, see the
[mathematical reference](../README.md#mathematical-reference) in the main README.
