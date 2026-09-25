"""A synthetic forest landscape with several owners, for studying cross-owner consequences.

The landscape consists of four lots (owners) arranged as a 2x2 block. Each lot is a grid of stands,
and each stand is managed with exactly one of a small set of management regimes. The per-hectare
values of each regime are synthetic, but their magnitudes are loosely calibrated against a
MELA-simulated Finnish holding. The numbers are not evidence about any real forest.
"""

import math

import numpy as np
from pydantic import BaseModel, Field

REGIMES: tuple[str, ...] = ("set_aside", "selection_cut", "thinning", "clearfell")
"""The management regimes available on every stand, in the order used by all regime-indexed values."""

LOTS: tuple[str, ...] = ("A", "B", "C", "D")
"""The lots of the landscape, placed as A B on the top row and C D on the bottom row."""

# Median per-hectare values of each regime, in the order of REGIMES.
_REGIME_MEDIANS: dict[str, tuple[float, ...]] = {
    "npv": (0.0, 4300.0, 4300.0, 7900.0),  # EUR/ha, discounted harvest revenue
    "carbon": (156.0, 119.0, 116.0, 107.0),  # tCO2/ha, stored at the end of the planning horizon
    "habitat": (0.80, 0.60, 0.45, 0.15),  # habitat suitability index, [0, 1]
    "bilberry": (6.4, 8.1, 5.9, 2.0),  # bilberry yield index
    "mushroom": (0.68, 0.46, 0.58, 0.21),  # marketed mushroom yield index
    "scenic": (6.6, 6.5, 5.9, 5.1),  # scenic value index
    "deadwood": (25.0, 12.0, 8.0, 4.0),  # m3/ha, deadwood from natural mortality
}


class Stand(BaseModel):
    """A single stand of the landscape."""

    id: int = Field(description="Index of the stand in the landscape.")
    lot: str = Field(description="The lot (owner) the stand belongs to.")
    row: int = Field(description="Row of the stand in the landscape grid.")
    col: int = Field(description="Column of the stand in the landscape grid.")
    area: float = Field(description="Area of the stand in hectares.")
    values: dict[str, list[float]] = Field(
        description="Per-hectare values of each quantity, one value per regime in the order of `REGIMES`."
    )


class ForestLandscape(BaseModel):
    """A forest landscape of several lots, each consisting of stands on a grid."""

    regimes: list[str] = Field(description="The management regimes available on every stand.")
    lots: list[str] = Field(description="The lots (owners) of the landscape.")
    stands: list[Stand] = Field(description="All stands of the landscape.")
    neighbours: dict[int, list[int]] = Field(
        description="Ids of the stands sharing an edge with each stand, regardless of the owner."
    )

    def lot_stands(self, lot: str) -> list[Stand]:
        """Return the stands belonging to `lot`."""
        return [stand for stand in self.stands if stand.lot == lot]

    def cross_owner_neighbours(self, stand_id: int) -> list[int]:
        """Return the ids of the neighbours of a stand that belong to another lot."""
        lot = self.stands[stand_id].lot
        return [n for n in self.neighbours[stand_id] if self.stands[n].lot != lot]


def _lot_grid_shape(stands_per_lot: int) -> tuple[int, int]:
    """Return the most square (rows, cols) grid holding exactly `stands_per_lot` stands."""
    rows = math.isqrt(stands_per_lot)
    while stands_per_lot % rows != 0:
        rows -= 1
    return rows, stands_per_lot // rows


def forest_landscape_data(stands_per_lot: int = 4, seed: int = 0) -> ForestLandscape:
    """Generate a synthetic forest landscape of four lots.

    Each lot is a grid of `stands_per_lot` stands, and the lots are placed as a 2x2 block, so
    that every lot shares a boundary with two other lots. Two stands are neighbours when they share
    an edge in the resulting landscape grid.

    The per-hectare value of each quantity on a stand is the median of its regime, scaled by a
    stand-specific factor and by a small regime-specific perturbation. The stand factor models
    site productivity for NPV and carbon, and site quality for the remaining quantities.

    Args:
        stands_per_lot (int, optional): the number of stands in each lot. Defaults to 4.
        seed (int, optional): seed for generating the areas and values. The same seed always
            gives the same landscape. Defaults to 0.

    Returns:
        ForestLandscape: the generated landscape.
    """
    if stands_per_lot < 2:  # noqa: PLR2004
        msg = f"A lot must have at least two stands, got {stands_per_lot}."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    lot_rows, lot_cols = _lot_grid_shape(stands_per_lot)

    stands: list[Stand] = []
    for lot_index, lot in enumerate(LOTS):
        lot_row, lot_col = divmod(lot_index, 2)
        for i in range(stands_per_lot):
            row, col = divmod(i, lot_cols)
            productivity = rng.uniform(0.7, 1.3)
            quality = rng.uniform(0.8, 1.2)
            values = {
                quantity: [
                    float(median * (productivity if quantity in ("npv", "carbon") else quality) * rng.uniform(0.9, 1.1))
                    for median in medians
                ]
                for quantity, medians in _REGIME_MEDIANS.items()
            }
            values["habitat"] = [min(value, 1.0) for value in values["habitat"]]
            stands.append(
                Stand(
                    id=len(stands),
                    lot=lot,
                    row=lot_row * lot_rows + row,
                    col=lot_col * lot_cols + col,
                    area=float(np.clip(rng.lognormal(math.log(1.2), 0.6), 0.3, 4.8)),
                    values=values,
                )
            )

    position = {(stand.row, stand.col): stand.id for stand in stands}
    neighbours = {
        stand.id: [
            position[(stand.row + dr, stand.col + dc)]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if (stand.row + dr, stand.col + dc) in position
        ]
        for stand in stands
    }

    return ForestLandscape(regimes=list(REGIMES), lots=list(LOTS), stands=stands, neighbours=neighbours)
