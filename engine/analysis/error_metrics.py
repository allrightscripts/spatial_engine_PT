"""Cell-volume weighted errors for one real spatial field."""
# grid volume is the measure of ONE cell, not the total domain volume.
# Omitting it makes L2 and mass depend incorrectly on grid resolution/length.
import numpy as np


# Compare equal-shaped real samples using cell-volume-weighted L2 and mass, plus Linf and
# regularized relative errors. Epsilon affects denominators, not the difference field.
def error_metrics(u_num, u_ref, grid, epsilon=1e-12):
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Metric epsilon must be finite and positive')
    if np.iscomplexobj(u_num) or np.iscomplexobj(u_ref):
        raise ValueError('Metrics expect real spatial fields')
    num, ref = np.asarray(u_num, float), np.asarray(u_ref, float)
    if num.shape != grid.shape or ref.shape != grid.shape:
        raise ValueError('Metric field/grid shape mismatch')
    if not np.isfinite(num).all() or not np.isfinite(ref).all():
        raise ValueError('Metric inputs must be finite')
    volume = float(grid.volume)
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError('Cell volume must be finite and positive')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        e = num - ref
        l2 = float(np.sqrt(np.sum(e * e) * volume))
        ref_l2 = float(np.sqrt(np.sum(ref * ref) * volume))
        mass_num = float(np.sum(num) * volume)
        mass_ref = float(np.sum(ref) * volume)
        mass_abs = abs(mass_num - mass_ref)
        result = {'L2': l2, 'rel_L2': l2 / (ref_l2 + epsilon),
                  'Linf': float(np.max(np.abs(e))), 'mass_abs': mass_abs,
                  'mass_rel': mass_abs / (abs(mass_ref) + epsilon)}
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite error metric')
    return result
