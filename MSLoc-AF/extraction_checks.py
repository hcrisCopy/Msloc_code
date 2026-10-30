"""Validation-only helpers; these never change sampling or feature computation."""
import math
import warnings

DEFAULT_FRAME_COUNT_TOLERANCE = 4

# The immediately preceding extractor, in LF and CRLF forms. The only compatible
# change is frame-count validation/auditing; all feature-producing fields remain
# equal. Unknown producer versions and changes to FPS/weights/etc. are rejected.
LEGACY_EXTRACTOR_HASHES = {
    "f8196079ce0872a7ab294caec15c792558758f38cf25ec8c2455ce76558f0adc",
    "7b9b5b7d0af568bfaa78e9247b65fc2e278cdce832fae5f620806453a9f7479e",
    # Previous count-tolerance release; this update only adds YAML/terminal UI.
    "faee78505339b928a7dc43b22df5fdeec50393073cea7d98a5480ea133680f44",
    "d4d326bf944b0e64ef36fcdf0f4191e9d665835df409c35eab3b7e273ecfb18c",
    # Last single-device extractor; multi-process execution keeps its per-video
    # preprocessing, sampling, hook tensors and aggregation unchanged.
    "8b229afd0700e23a11ad492fbb9c5c467a2d1f2795e5306ab41667d24ad8f95b",
}


def check_frame_count(video, duration, fps, observed, tolerance):
    if (not math.isfinite(duration) or not math.isfinite(fps) or duration <= 0
            or fps <= 0 or observed <= 0 or tolerance < 0):
        raise ValueError("Invalid duration, FPS, observed frame count or count tolerance")
    expected = duration * fps
    error = observed - expected
    if abs(error) > tolerance + 1e-6:
        raise ValueError(
            f"FPS/count mismatch: {video}: {observed} frames, duration*FPS={expected:.3f}, "
            f"difference={error:+.3f} frames ({error/fps:+.3f}s), allowed={tolerance} frames; "
            "verify actual extraction FPS and annotation duration. "
            "--frame-count-tolerance changes this check only, not the sampling FPS."
        )
    if abs(error) > 2 + 1e-6:
        warnings.warn(
            f"Frame-count difference within tolerance: {video}: {error:+.3f} frames "
            f"({error/fps:+.3f}s), allowed={tolerance}; keeping configured FPS={fps:g} "
            "and annotation timestamps unchanged.", RuntimeWarning, stacklevel=2)
    return dict(expected_frames=expected, difference_frames=error,
                difference_seconds=error/fps, tolerance_frames=tolerance)


def require_compatible_cache(existing, requested):
    """Allow validation-only changes, preserving the existing feature signature."""
    if existing == requested:
        return
    old, new = dict(existing), dict(requested)
    old_hash = old.pop("extractor_sha256", None)
    new_hash = new.pop("extractor_sha256", None)
    old.pop("frame_count_tolerance", None)
    new.pop("frame_count_tolerance", None)
    different = sorted(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))
    if different:
        raise ValueError("Existing cache has a different extraction spec; use a new output directory. "
                         f"Changed fields: {different}")
    if not old_hash or old_hash != new_hash and old_hash not in LEGACY_EXTRACTOR_HASHES:
        raise ValueError("Unknown cached extractor version; cannot assume identical features. "
                         "Use a new output directory.")
