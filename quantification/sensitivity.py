"""Sensitivity and uncertainty analysis for the capture model comparison.

``quantify_models`` reports one ranking under one set of declared assumptions:
an empirical degree-of-polarization scale, a usable-image-circle radius, a solar
exclusion radius and a minimum measured degree of polarization for angle
statistics.  A ranking that only holds at those four values is not a result.

This script answers two questions the single ranking cannot:

* **Sensitivity.** Does the rank order survive moving each declared assumption
  across a defensible range?
* **Uncertainty.** Pixels inside one frame are spatially correlated, so the
  hundreds of thousands of analyzer tiles are not independent observations.  A
  moving-block bootstrap over the residual fields turns the point scores into
  intervals and gives paired comparisons between models.

Both reuse the scoring functions from :mod:`quantify_models` unchanged, so the
numbers here and there are computed by the same code.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from pyskylumos import __version__

if TYPE_CHECKING or __package__:
    from quantification.quantify_models import (
        CANONICAL_MODELS,
        POSITIONS,
        CaptureConfig,
        EvaluationMasks,
        MeasuredData,
        _build_engine,
        _load_measurements,
        _select_cie_sky_type,
        _simulate,
        _sun_position,
        axial_difference_radians,
        axial_error_metrics,
        build_masks,
        load_config,
        scalar_error_metrics,
        verify_checksums,
    )
else:  # Direct ``python quantification/sensitivity.py`` invocation.
    from quantify_models import (  # type: ignore[import-not-found]
        CANONICAL_MODELS,
        POSITIONS,
        CaptureConfig,
        EvaluationMasks,
        MeasuredData,
        _build_engine,
        _load_measurements,
        _select_cie_sky_type,
        _simulate,
        _sun_position,
        axial_difference_radians,
        axial_error_metrics,
        build_masks,
        load_config,
        scalar_error_metrics,
        verify_checksums,
    )

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]

DEFAULT_BLOCK_TILES = 32
DEFAULT_RESAMPLES = 1000
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class ModelFields:
    """One model's tile-resolution fields, cached across sweep settings."""

    unscaled_dop: FloatArray
    aop: FloatArray


def simulate_model_fields(
    config: CaptureConfig,
    model: str,
    cie_sky_type: int,
    world_azimuths: FloatArray,
    altitudes: FloatArray,
) -> ModelFields:
    """Simulate one model and pool it onto the analyzer-tile grid.

    The degree of polarization is returned *before* the empirical scale, because
    that scale is a pure multiplier and sweeping it must not require resimulating.

    Args:
        config: Capture manifest.
        model: Canonical model name.
        cie_sky_type: CIE relative-radiance sky type.
        world_azimuths: Native-resolution world azimuth grid in degrees.
        altitudes: Native-resolution altitude grid in degrees.

    Returns:
        The unscaled tile degree of polarization and the tile angle of polarization.
    """
    engine = _build_engine(config)
    times, location, _, _ = _sun_position(config)
    tile_shape = (config.image_height // 2, config.image_width // 2)
    dop_sum = np.zeros(tile_shape, dtype=np.float64)
    axial_sum = np.zeros(tile_shape, dtype=np.complex128)
    for row, column in POSITIONS:
        dop, aop, _ = _simulate(
            engine,
            config,
            times,
            location,
            model,
            cie_sky_type,
            world_azimuths[row::2, column::2],
            altitudes[row::2, column::2],
        )
        dop_sum += dop
        axial_sum += np.exp(2j * aop)
    return ModelFields(
        unscaled_dop=np.asarray(dop_sum / 4.0, dtype=np.float64),
        aop=np.asarray(0.5 * np.angle(axial_sum), dtype=np.float64),
    )


def score_model(
    fields: ModelFields,
    measured: MeasuredData,
    masks: EvaluationMasks,
    dop_scale: float,
) -> dict[str, float]:
    """Score one cached model under one set of masks and one scale."""
    scaled = np.asarray(dop_scale * fields.unscaled_dop, dtype=np.float64)
    dop_metrics = scalar_error_metrics(scaled, measured.tile_dop, masks.tile, dop_scale)
    aop_metrics = axial_error_metrics(fields.aop, measured.tile_aop, masks.aop_tile)
    return {
        "dop_nrmse": dop_metrics.nrmse,
        "dop_rmse": dop_metrics.rmse,
        "dop_mae": dop_metrics.mae,
        "dop_count": float(dop_metrics.count),
        "aop_nrmse": aop_metrics.nrmse,
        "aop_rmse_deg": aop_metrics.rmse,
        "aop_mae_deg": aop_metrics.mae,
        "aop_count": float(aop_metrics.count),
        "polarization_score": float(np.sqrt((dop_metrics.nrmse**2 + aop_metrics.nrmse**2) / 2.0)),
    }


def _geometry(config: CaptureConfig) -> tuple[FloatArray, FloatArray, float, float]:
    engine = _build_engine(config)
    azimuths, altitudes = engine.get_initial_azimuth_altitude(altitude_min_clip=None)
    world_azimuths = np.asarray(engine.rotate_sensor(azimuths, config.yaw_deg), dtype=np.float64)
    _, _, sun_azimuth, sun_altitude = _sun_position(config)
    return (
        world_azimuths,
        np.asarray(altitudes, dtype=np.float64),
        sun_azimuth,
        sun_altitude,
    )


def _sweep_settings(quick: bool) -> list[tuple[str, str, float]]:
    """Enumerate ``(parameter, label, value)`` triples for the sensitivity sweep."""
    if quick:
        return [
            ("dop_scale", "0.60", 0.60),
            ("dop_scale", "0.70", 0.70),
            ("usable_image_radius_pixels", "875", 875.0),
            ("sun_exclusion_deg", "15", 15.0),
        ]
    settings: list[tuple[str, str, float]] = []
    for value in (0.50, 0.60, 0.70, 0.80, 0.90):
        settings.append(("dop_scale", f"{value:.2f}", value))
    # 875 px reaches into the camera rim on purpose: the sweep has to show what
    # the excluded annulus does to the ranking, not hide it.
    for value in (730.0, 765.0, 800.0, 845.0, 875.0):
        settings.append(("usable_image_radius_pixels", f"{value:g}", value))
    for value in (5.0, 10.0, 15.0, 20.0):
        settings.append(("sun_exclusion_deg", f"{value:g}", value))
    for value in (0.02, 0.05, 0.10, 0.15):
        settings.append(("aop_min_measured_dop", f"{value:.2f}", value))
    return settings


def run_sensitivity(
    config: CaptureConfig,
    measured: MeasuredData,
    *,
    quick: bool,
    verbose: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Recompute the ranking at every swept assumption value.

    Returns:
        One row per model and setting, and the CIE type selected per setting.
    """
    world_azimuths, altitudes, sun_azimuth, sun_altitude = _geometry(config)
    cache: dict[tuple[str, int], ModelFields] = {}
    rows: list[dict[str, Any]] = []
    cie_by_setting: dict[str, int] = {}

    settings = [("baseline", "as published", float("nan")), *_sweep_settings(quick)]
    for parameter, label, value in settings:
        variant = config
        if parameter != "baseline":
            # Annotated Any so one loop can override any swept field by name.
            override: dict[str, Any] = {parameter: value}
            variant = replace(config, **override)
        masks = build_masks(variant, measured, world_azimuths, altitudes, sun_azimuth, sun_altitude)
        times, location, _, _ = _sun_position(variant)
        cie_sky_type, _ = _select_cie_sky_type(variant, measured, masks, times, location)
        cie_by_setting[f"{parameter}={label}"] = cie_sky_type

        scored: list[tuple[str, dict[str, float]]] = []
        for model in CANONICAL_MODELS:
            key = (model, cie_sky_type)
            if key not in cache:
                cache[key] = simulate_model_fields(
                    variant, model, cie_sky_type, world_azimuths, altitudes
                )
            scored.append((model, score_model(cache[key], measured, masks, variant.dop_scale)))

        order = sorted(scored, key=lambda item: item[1]["polarization_score"])
        ranks = {model: index + 1 for index, (model, _) in enumerate(order)}
        aop_order = sorted(scored, key=lambda item: item[1]["aop_rmse_deg"])
        aop_ranks = {model: index + 1 for index, (model, _) in enumerate(aop_order)}
        for model, metrics in scored:
            rows.append(
                {
                    "parameter": parameter,
                    "setting": label,
                    "value": value,
                    "cie_sky_type": cie_sky_type,
                    "model": model,
                    "rank": ranks[model],
                    "aop_rank": aop_ranks[model],
                    **metrics,
                }
            )
        if verbose:
            winner = order[0][0]
            print(
                f"{parameter:22s} {label:12s} cie={cie_sky_type:2d} "
                f"best={winner:20s} order={'>'.join(model for model, _ in order)}",
                flush=True,
            )
    return rows, cie_by_setting


def _block_index(shape: tuple[int, int], block_tiles: int) -> NDArray[np.int64]:
    """Label every tile with the index of the block it belongs to."""
    rows = np.arange(shape[0], dtype=np.int64)[:, None] // block_tiles
    columns = np.arange(shape[1], dtype=np.int64)[None, :] // block_tiles
    blocks_per_row = int(np.ceil(shape[1] / block_tiles))
    return np.asarray(rows * blocks_per_row + columns, dtype=np.int64)


def _block_sums(
    residuals: FloatArray, mask: BoolArray, blocks: NDArray[np.int64], block_count: int
) -> tuple[FloatArray, FloatArray]:
    """Sum squared residuals and counts within every block."""
    valid = mask & np.isfinite(residuals)
    labels = blocks[valid]
    squares = np.square(residuals[valid])
    sums = np.bincount(labels, weights=squares, minlength=block_count)
    counts = np.bincount(labels, minlength=block_count)
    return (
        np.asarray(sums, dtype=np.float64),
        np.asarray(counts, dtype=np.float64),
    )


def run_bootstrap(
    config: CaptureConfig,
    measured: MeasuredData,
    *,
    resamples: int,
    block_tiles: int,
    confidence: float,
    seed: int,
    verbose: bool,
) -> dict[str, Any]:
    """Moving-block bootstrap over the residual fields.

    Blocks of adjacent analyzer tiles are resampled with replacement.  Every
    model is evaluated on the *same* resampled blocks, so the pairwise score
    differences are paired and their intervals are meaningful.
    """
    world_azimuths, altitudes, sun_azimuth, sun_altitude = _geometry(config)
    masks = build_masks(config, measured, world_azimuths, altitudes, sun_azimuth, sun_altitude)
    times, location, _, _ = _sun_position(config)
    cie_sky_type, _ = _select_cie_sky_type(config, measured, masks, times, location)

    tile_shape = (config.image_height // 2, config.image_width // 2)
    blocks = _block_index(tile_shape, block_tiles)
    block_count = int(blocks.max()) + 1

    statistics: dict[str, dict[str, FloatArray]] = {}
    point: dict[str, float] = {}
    for model in CANONICAL_MODELS:
        fields = simulate_model_fields(config, model, cie_sky_type, world_azimuths, altitudes)
        dop_residual = np.asarray(
            config.dop_scale * fields.unscaled_dop - measured.tile_dop, dtype=np.float64
        )
        aop_residual = np.asarray(
            np.rad2deg(axial_difference_radians(fields.aop, measured.tile_aop)), dtype=np.float64
        )
        dop_sums, dop_counts = _block_sums(dop_residual, masks.tile, blocks, block_count)
        aop_sums, aop_counts = _block_sums(aop_residual, masks.aop_tile, blocks, block_count)
        statistics[model] = {
            "dop_sums": dop_sums,
            "dop_counts": dop_counts,
            "aop_sums": aop_sums,
            "aop_counts": aop_counts,
        }
        point[model] = float(
            np.sqrt(
                (
                    (np.sqrt(dop_sums.sum() / dop_counts.sum()) / config.dop_scale) ** 2
                    + (np.sqrt(aop_sums.sum() / aop_counts.sum()) / 90.0) ** 2
                )
                / 2.0
            )
        )
        if verbose:
            print(f"bootstrap prepared {model:22s} point score {point[model]:.6f}", flush=True)

    generator = np.random.default_rng(seed)
    draws = generator.integers(0, block_count, size=(resamples, block_count))
    scores = {model: np.zeros(resamples, dtype=np.float64) for model in CANONICAL_MODELS}
    for index in range(resamples):
        selection = draws[index]
        for model, arrays in statistics.items():
            dop_count = arrays["dop_counts"][selection].sum()
            aop_count = arrays["aop_counts"][selection].sum()
            if dop_count <= 0.0 or aop_count <= 0.0:
                scores[model][index] = np.nan
                continue
            dop_nrmse = np.sqrt(arrays["dop_sums"][selection].sum() / dop_count) / config.dop_scale
            aop_nrmse = np.sqrt(arrays["aop_sums"][selection].sum() / aop_count) / 90.0
            scores[model][index] = np.sqrt((dop_nrmse**2 + aop_nrmse**2) / 2.0)

    lower_quantile = (1.0 - confidence) / 2.0
    upper_quantile = 1.0 - lower_quantile
    intervals: list[dict[str, Any]] = []
    for model in CANONICAL_MODELS:
        sample = scores[model][np.isfinite(scores[model])]
        intervals.append(
            {
                "model": model,
                "point_score": point[model],
                "bootstrap_mean": float(np.mean(sample)),
                "lower": float(np.quantile(sample, lower_quantile)),
                "upper": float(np.quantile(sample, upper_quantile)),
            }
        )

    ordered = sorted(CANONICAL_MODELS, key=lambda model: point[model])
    pairs: list[dict[str, Any]] = []
    for first_index, first in enumerate(ordered):
        for second in ordered[first_index + 1 :]:
            difference = scores[first] - scores[second]
            finite = difference[np.isfinite(difference)]
            pairs.append(
                {
                    "better": first,
                    "worse": second,
                    "point_difference": point[first] - point[second],
                    "lower": float(np.quantile(finite, lower_quantile)),
                    "upper": float(np.quantile(finite, upper_quantile)),
                    "probability_better": float(np.mean(finite < 0.0)),
                    "separated": bool(float(np.quantile(finite, upper_quantile)) < 0.0),
                }
            )

    return {
        "resamples": resamples,
        "block_tiles": block_tiles,
        "block_native_pixels": block_tiles * 2,
        "blocks": block_count,
        "confidence": confidence,
        "seed": seed,
        "cie_sky_type": cie_sky_type,
        "intervals": intervals,
        "pairwise": pairs,
    }


def _plot(
    output_dir: Path,
    sensitivity_rows: list[dict[str, Any]],
    bootstrap: dict[str, Any],
    tight_field: dict[str, Any],
) -> None:
    """Render the sensitivity and uncertainty figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parameters = [
        parameter
        for parameter in dict.fromkeys(row["parameter"] for row in sensitivity_rows)
        if parameter != "baseline"
    ]
    figure, axes = plt.subplots(1, len(parameters), figsize=(4.0 * len(parameters), 4.0))
    if len(parameters) == 1:
        axes = np.asarray([axes])
    for axis, parameter in zip(axes, parameters, strict=True):
        for model in CANONICAL_MODELS:
            rows = [
                row
                for row in sensitivity_rows
                if row["parameter"] == parameter and row["model"] == model
            ]
            rows.sort(key=lambda row: row["value"])
            axis.plot(
                [row["value"] for row in rows],
                [row["rank"] for row in rows],
                "o-",
                label=model.replace("_", " ").title(),
            )
        axis.set_xlabel(parameter)
        axis.set_ylabel("combined rank")
        axis.invert_yaxis()
        axis.set_yticks(range(1, len(CANONICAL_MODELS) + 1))
        axis.set_title(parameter)
    axes[-1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    figure.tight_layout()
    figure.savefig(output_dir / "sensitivity_ranks.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    panels = (
        (bootstrap, "(a) as published, $r \\leq 800$ px", "#1f77b4"),
        (tight_field, "(b) tighter field, $r \\leq 730$ px", "#d62728"),
    )
    for axis, (result, title, colour) in zip(axes, panels, strict=True):
        order = sorted(result["intervals"], key=lambda item: item["point_score"])
        positions = np.arange(len(order), dtype=np.float64)
        axis.errorbar(
            positions,
            [item["point_score"] for item in order],
            yerr=np.asarray(
                [
                    [item["point_score"] - item["lower"] for item in order],
                    [item["upper"] - item["point_score"] for item in order],
                ]
            ),
            fmt="o",
            capsize=5,
            color=colour,
        )
        axis.set_xticks(positions)
        axis.set_xticklabels([item["model"].replace("_", "\n") for item in order], fontsize=7)
        axis.set_ylabel("combined normalized RMSE score")
        axis.set_title(title, fontsize=10)
    figure.suptitle(
        f"{int(bootstrap['confidence'] * 100)}% moving-block bootstrap "
        f"({bootstrap['blocks']} blocks of {bootstrap['block_native_pixels']} px)",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "bootstrap_scores.png", dpi=200)
    plt.close(figure)


def run(
    config_path: Path,
    output_dir: Path,
    *,
    resamples: int,
    block_tiles: int,
    confidence: float,
    seed: int,
    quick: bool,
    verbose: bool,
) -> int:
    """Run the sensitivity sweep and the bootstrap, then write every artifact."""
    config = load_config(config_path)
    verify_checksums(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    measured = _load_measurements(config)

    sensitivity_rows, cie_by_setting = run_sensitivity(
        config, measured, quick=quick, verbose=verbose
    )
    # The sweep shows the comparison radius decides the ordering, so the
    # bootstrap is run in both regimes: as published, and on a tighter field.
    bootstrap = run_bootstrap(
        config,
        measured,
        resamples=resamples,
        block_tiles=block_tiles,
        confidence=confidence,
        seed=seed,
        verbose=verbose,
    )
    tight_field = run_bootstrap(
        replace(config, usable_image_radius_pixels=730.0),
        measured,
        resamples=resamples,
        block_tiles=block_tiles,
        confidence=confidence,
        seed=seed,
        verbose=verbose,
    )

    with (output_dir / "sensitivity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sensitivity_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sensitivity_rows)

    baseline = {
        row["model"]: row["rank"] for row in sensitivity_rows if row["parameter"] == "baseline"
    }
    swept = [row for row in sensitivity_rows if row["parameter"] != "baseline"]
    rank_changes = [row for row in swept if row["rank"] != baseline[row["model"]]]
    winners = {
        f"{row['parameter']}={row['setting']}": row["model"] for row in swept if row["rank"] == 1
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pyskylumos_version": __version__,
        "config_path": str(config.config_path),
        "sensitivity": {
            "settings": len({(row["parameter"], row["setting"]) for row in swept}),
            "baseline_ranks": baseline,
            "rank_changes": len(rank_changes),
            "distinct_winners": sorted(set(winners.values())),
            "cie_sky_type_by_setting": cie_by_setting,
            "rows": sensitivity_rows,
        },
        "bootstrap": bootstrap,
        "bootstrap_usable_image_radius_730_px": tight_field,
        "assumptions": {
            "cie_sky_type_reselected_for_every_setting": True,
            "dop_scale_is_a_pure_multiplier_so_models_are_simulated_once": True,
            "blocks_are_contiguous_tiles_not_independent_observations": True,
            "single_capture_so_intervals_describe_this_frame_only": True,
        },
    }
    with (output_dir / "sensitivity.json").open("w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")

    _plot(output_dir, sensitivity_rows, bootstrap, tight_field)

    print(
        f"sensitivity: {len(rank_changes)} rank changes across "
        f"{report['sensitivity']['settings']} settings; "
        f"winners {report['sensitivity']['distinct_winners']}"
    )
    for label, result in (
        (f"r <= {config.usable_image_radius_pixels:g} px", bootstrap),
        ("r <= 730 px", tight_field),
    ):
        separated = sum(1 for pair in result["pairwise"] if pair["separated"])
        best = min(result["intervals"], key=lambda item: item["point_score"])["model"]
        print(
            f"bootstrap [{label}]: best={best}; {separated}/{len(result['pairwise'])} "
            f"pairs separated at {int(confidence * 100)}% confidence"
        )
    return 0


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parent / "capture.toml"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "output"
    )
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--block-tiles", type=int, default=DEFAULT_BLOCK_TILES)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="Reduced sweep for a smoke test.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Entry point."""
    parsed = parse_arguments(arguments)
    return run(
        parsed.config.expanduser().resolve(),
        parsed.output_dir.expanduser().resolve(),
        resamples=parsed.resamples,
        block_tiles=parsed.block_tiles,
        confidence=parsed.confidence,
        seed=parsed.seed,
        quick=parsed.quick,
        verbose=not parsed.quiet,
    )


if __name__ == "__main__":
    raise SystemExit(main())
