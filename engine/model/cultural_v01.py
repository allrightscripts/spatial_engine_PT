"""Researcher's equation file for the v0.9 cultural demo model.

Edit formulas here; numerical parameter values stay in config JSON.

U.shape == (number_of_current_fields, *grid.shape).
`fields[i]` contains family/model_type/parameters metadata for U[i].
The implementation deliberately never assumes that a particular array index
is T or P, so it remains valid when structural events change field count/order.
"""
import numpy as np

from .definitions import Model
from .mobility import ConstantFieldMobility


# ======================= FUNCTIONAL ========================================
# F[U] = 1/2 integral sum_i U_i^2 dx, hence mu_i = U_i.
class CulturalFunctional:
    # Quadratic free-energy derivative mu=U; returned array shares storage with U and must
    # be treated as read-only.
    def chemical_potential(self, U, fields, grid, t):
        return U

    # Integrate one half of sum_i U_i squared using grid.volume, the volume of one cell.
    # Return a scalar diagnostic.
    def value(self, U, fields, grid, t):
        return float(0.5 * np.sum(U * U) * grid.volume)


# ======================= ONSAGER / MOBILITY ================================
# Per-field diffusion and optional mobility_row metadata are handled by the
# existing ConstantFieldMobility implementation.
class CulturalMobility(ConstantFieldMobility):
    pass


# ======================= HELPERS ===========================================
def family_sum(U, fields, family):
    """Sum all current fields belonging to one family; shape == one field."""
    indices = [j for j, f in enumerate(fields) if f.get("family") == family]
    if not indices:
        return np.zeros_like(U[0])
    return U[indices].sum(axis=0)


def other_family_members(U, fields, i, family):
    """Sum a family while excluding U[i] when U[i] belongs to that family."""
    total = family_sum(U, fields, family)
    if fields[i].get("family") == family:
        return total - U[i]
    return total


def smooth_switch(x, threshold, width):
    """Stable logistic switch from 0 to 1 around `threshold`."""
    if width <= 0.0:
        return (x >= threshold).astype(float)
    z = np.clip((x - threshold) / width, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


# ======================= BACKWARD-COMPATIBLE DEFAULT =======================
def logistic_reaction(i, U, fields, grid, t, total):
    """Legacy shared logistic law kept for model_type='default'."""
    p = fields[i]["parameters"]
    capacity = max(float(p.get("capacity", 1.0)), 1e-12)
    return (
        p.get("growth", 0.0) * U[i] * (1.0 - U[i] / capacity)
        - p.get("competition", 0.0) * U[i] * (total - U[i])
        - p.get("decay", 0.0) * U[i]
    )


# ======================= T: TRADITION ======================================
def tradition_v1(i, U, fields, grid, t, total):
    """Slow self-limited T dynamics with competition between T lineages.

    R_Ti = r_i T_i (1 - T_i/K_i)
           - c_T,i T_i T_other
           - d_i T_i

    T does not need to receive a direct energetic benefit from suppressing P.
    Its 'despotic' effect is represented in progressive_v1 as a negative term
    acting on P.
    """
    p = fields[i]["parameters"]
    Ti = U[i]
    T_other = other_family_members(U, fields, i, "T")

    r = float(p.get("growth", 0.0))
    K = max(float(p.get("capacity", 1.0)), 1e-12)
    cT = float(p.get("competition_T", p.get("competition", 0.0)))
    decay = float(p.get("decay", 0.0))

    return r * Ti * (1.0 - Ti / K) - cT * Ti * T_other - decay * Ti


# ======================= P: PROGRESS / NOVELTY =============================
def progressive_v1(i, U, fields, grid, t, total):
    """P dynamics: self-growth + P-to-P acceleration + T suppression.

    Let
        P_other = sum of all other active P fields,
        T_total = sum of all active T fields,
        S(P_total) = logistic threshold switch.

    Then
        R_Pi = r_i P_i (1 - P_i/K_i)
               + a_i P_i P_other
               - b_i T_total P_i S(P_total)
               - d_i P_i.

    Interpretation used by the demo model:
      * progress_acceleration > 0: existing novelty accelerates other novelty;
      * T_suppression > 0: active T suppresses P;
      * the sigmoid makes suppression strongest after aggregate P becomes
        sufficiently visible/large.

    This reaction law is independent from stochastic P-birth hazard: the hazard
    controls WHEN a new dimension appears; this law controls what happens to an
    already-existing P field afterward.
    """
    p = fields[i]["parameters"]
    Pi = U[i]

    P_total = family_sum(U, fields, "P")
    P_other = other_family_members(U, fields, i, "P")
    T_total = family_sum(U, fields, "T")

    r = float(p.get("growth", 0.0))
    K = max(float(p.get("capacity", 1.0)), 1e-12)
    accel = float(p.get("progress_acceleration", p.get("help_from_P", 0.0)))
    suppress = float(p.get("T_suppression", p.get("suppression_by_T", 0.0)))
    threshold = float(p.get("suppression_threshold", p.get("threshold_suppression", 1.0)))
    width = float(p.get("suppression_width", p.get("threshold_width", 0.25)))
    decay = float(p.get("decay", 0.0))

    switch = smooth_switch(P_total, threshold, width)

    return (
        r * Pi * (1.0 - Pi / K)
        + accel * Pi * P_other
        - suppress * T_total * Pi * switch
        - decay * Pi
    )


REACTION_LAWS = {
    "default": logistic_reaction,
    "progressive_v1": progressive_v1,
    "tradition_v1": tradition_v1,
}


class CulturalReaction:
    # Require model_type to name an implemented REACTION_LAWS entry; family labels alone
    # do not select a reaction formula.
    @staticmethod
    def validate_type(model_type):
        if model_type not in REACTION_LAWS:
            raise ValueError(f"Unknown reaction model_type {model_type}")

    # Dispatch each current field to its model_type reaction and return an array shaped
    # like U. Use aligned field metadata rather than permanent numeric indices.
    def __call__(self, U, fields, grid, t):
        total = U.sum(axis=0)
        result = np.empty_like(U)
        for i, item in enumerate(fields):
            kind = item.get("model_type", "default")
            self.validate_type(kind)
            result[i] = REACTION_LAWS[kind](i, U, fields, grid, t, total)
        return result


def build_model(parameters=None):
    """Factory expected by the universal loader."""
    return Model(CulturalFunctional(), CulturalMobility(), CulturalReaction())
