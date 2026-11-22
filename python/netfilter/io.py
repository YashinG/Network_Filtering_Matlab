"""Data loading and Pajek export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "Data" / "Data.mat"


def load_data(path=DEFAULT_DATA):
    """Load ``stStockRets`` from ``Data.mat`` into a dict of numpy/pandas objects.

    Keys: ``rets`` (n × p), ``dates`` (DatetimeIndex), ``names``,
    ``super_sector_names``, ``industry_num``, ``super_sector_num``,
    ``sector_num``, ``sub_sector_num``, ``all_sector_num`` (4 × p),
    ``pajek_labels``.
    """
    s = loadmat(path, squeeze_me=True, struct_as_record=False)["stStockRets"]
    strs = lambda a: [str(x) for x in np.atleast_1d(a)]
    return {
        "rets": np.asarray(s.rets, dtype=float),
        "dates": pd.to_datetime(strs(s.mlDates), format="%d-%b-%Y"),
        "names": strs(s.aNames),
        "super_sector_names": strs(s.SuperSectorNames),
        "industry_num": np.asarray(s.IndustryNum),
        "super_sector_num": np.asarray(s.SuperSectorNum),
        "sector_num": np.asarray(s.SectorNum),
        "sub_sector_num": np.asarray(s.SubSectorNum),
        "all_sector_num": np.asarray(s.allSectorNum),
        "pajek_labels": strs(s.pajekStrData),
    }


def to_pajek(data, labels, filename, colour_edges=False, edge_colours=None):
    """Write a Pajek ``.net`` file (port of adj2pajek2.m).

    ``data`` is either an (N, N) matrix (written as ``*Matrix``) or an
    (E, 3) edge list ``[i, j, weight]`` with 1-based vertex numbers (written
    as ``*Edges``).
    """
    data = np.asarray(data)
    lines = [f"*Vertices {len(labels)}", *labels]
    if data.ndim == 2 and data.shape[0] == data.shape[1]:
        lines.append("*Matrix")
        lines += [" ".join(str(int(round(v))) for v in row) for row in data]
    else:
        lines.append("*Edges")
        for k, (i, j, wgt) in enumerate(data):
            prefix = f"{int(i)} {int(j)} "
            if colour_edges:
                if edge_colours is not None:
                    lines.append(prefix + ("0.5000 c Grey" if edge_colours[k] > 0 else "2.0000 c Maroon"))
                else:
                    lines.append(prefix + (f"{wgt:.4f} c Black" if wgt > 0 else f"{-wgt:.4f} c Maroon"))
            else:
                lines.append(prefix + f"{wgt:.4f} ")
    Path(f"{filename}.net").write_text("\n".join(lines) + "\n")
