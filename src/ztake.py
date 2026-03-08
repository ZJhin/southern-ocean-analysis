from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import xarray as xr
import requests
import os


@dataclass
class Ztake:
    """
    Select newest datasets per (model, variable) respecting member/grid
    preferences; open with xarray from *file paths*; and compare local
    intake-ESM availability vs ESGF online for the same constraints.
    """
    cmip6_catalog: Any                      # intake-ESM top-level catalog
    constraints: Dict[str, Any]
    chunks: Dict[str, int] = field(default_factory=lambda: {"time": 12})
    prefer_members: Tuple[str, ...] = ("r1i1p1f1", "r1i1p1f2", "r1i1p1f3")
    prefer_grids:   Tuple[str, ...] = ("gn", "gr", "gr1")

    # internals (populated in __post_init__)
    filtered_ds: Any = field(init=False, repr=False)
    _df: pd.DataFrame = field(init=False, repr=False)
    _best_df: pd.DataFrame = field(init=False, repr=False)
    _model_list: List[str] = field(init=False, repr=False)

    # ---------- construction ----------
    def __post_init__(self):
        cons = dict(self.constraints)
        if "variable" in cons and "variable_id" not in cons:
            cons["variable_id"] = cons.pop("variable")

        self.filtered_ds = self.cmip6_catalog.search(**cons)
        self._df = self.filtered_ds.df.copy()

        # ensure required columns
        self._require_cols(self._df, ["source_id", "variable_id", "member_id",
                                      "grid_label", "version"])

        # ensure 'path' column exists (fall back to 'uri' if present)
        if "path" not in self._df.columns:
            fallback = self._df.get("uri")
            if fallback is None:
                # create a dummy string to avoid KeyError during sort/merge
                fallback = pd.Series(np.arange(len(self._df)), index=self._df.index)
            self._df["path"] = fallback.astype(str)

        # parse numeric version, rank preferences
        v = self._df["version"].astype(str).str.lstrip("v")
        self._df["version_num"] = pd.to_numeric(v, errors="coerce").fillna(0).astype(int)
        self._df["member_rank"] = self._rank_by_pref(self._df["member_id"], self.prefer_members)
        self._df["grid_rank"]   = self._rank_by_pref(self._df["grid_label"], self.prefer_grids)

        # pick best row per (model, variable)
        self._best_df = (
            self._df.groupby(["source_id", "variable_id"], group_keys=False)
                    .apply(self._pick_best_one)
                    .reset_index(drop=True)
        )
        self._model_list = sorted(self._best_df["source_id"].unique().tolist())

    # ---------- small utilities ----------
    @staticmethod
    def _require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Required columns missing: {missing}")

    @staticmethod
    def _rank_by_pref(s: pd.Series, prefs: Tuple[str, ...]) -> pd.Series:
        m = {v: i for i, v in enumerate(prefs)}
        return s.map(m).fillna(len(prefs) + 100).astype(int)

    @staticmethod
    def _pick_best_one(g: pd.DataFrame) -> pd.DataFrame:
        # stable mergesort keeps deterministic behavior for ties
        return (g.sort_values(
                    by=["member_rank", "grid_rank", "version_num", "path"],
                    ascending=[True, True, False, True],
                    kind="mergesort")
                .head(1))

    def _paths_for_choice(self, row: pd.Series) -> List[str]:
        """
        Return all file paths for the selected (model, variable, member, grid, version).
        """
        mask = (
            (self._df["source_id"]   == row["source_id"])   &
            (self._df["variable_id"] == row["variable_id"]) &
            (self._df["member_id"]   == row["member_id"])   &
            (self._df["grid_label"]  == row["grid_label"])  &
            (self._df["version"]     == row["version"])
        )
        paths = self._df.loc[mask, "path"].astype(str).tolist()
        paths.sort()
        if not paths:
            raise FileNotFoundError(
                f"No files for {row['source_id']} {row['variable_id']} "
                f"{row['member_id']} {row['grid_label']} v{row['version']}"
            )
        return paths

    # ---------- public: quick views ----------
    @property
    def df(self) -> pd.DataFrame:
        """Full filtered rows (NOT only best choice)."""
        return self._df.copy()

    @property
    def best_per_model_variable(self) -> pd.DataFrame:
        """One best row per (model, variable), with key fields."""
        cols = ["source_id", "variable_id", "member_id",
                "grid_label", "version", "version_num", "path"]
        return self._best_df.loc[:, [c for c in cols if c in self._best_df.columns]].copy()

    def models(self) -> List[str]:
        """Model list after best filtering."""
        return list(self._model_list)

    def models_all(self) -> List[str]:
        """All unique models in the filtered dataframe."""
        return sorted(self._df["source_id"].unique().tolist())

    def models_best(self) -> List[str]:
        """All unique models in the best-choice dataframe."""
        return sorted(self._best_df["source_id"].unique().tolist())

    def variables_for(self, model: str) -> List[str]:
        """Variables available for a given model (best-choice rows)."""
        return sorted(self._best_df.query("source_id == @model")["variable_id"].unique())

    def info(self) -> pd.DataFrame:
        """Per-model summary of chosen rows."""
        df = self._best_df
        return (df.groupby("source_id", as_index=False)
                  .agg(n_vars=("variable_id", "nunique"),
                       members=("member_id",  lambda x: ",".join(sorted(pd.unique(x)))),
                       grids=("grid_label",   lambda x: ",".join(sorted(pd.unique(x)))),
                       newest_version=("version_num", "max"))
                  .sort_values("source_id")
                  .reset_index(drop=True))

    # ---------- open data ----------
    def open(
        self,
        variables: Optional[Iterable[str]] = None,
        time_range: Optional[Tuple[str, str]] = None,
        engine: Optional[str] = None,
        decode_times: bool = True,
        drop_conflicts: bool = True,
    ) -> Dict[str, xr.Dataset]:
        """
        Open per model and merge requested variables. Files are concatenated
        with xr.open_mfdataset(combine="by_coords"); variables merged via xr.merge().
        """
        sel = self._best_df
        if variables is not None:
            variables = tuple(variables)
            sel = sel[sel["variable_id"].isin(variables)]
            if sel.empty:
                raise ValueError(f"No rows match requested variables: {variables}")

        out: Dict[str, xr.Dataset] = {}
        for model, g in sel.groupby("source_id"):
            var_ds: List[xr.Dataset] = []
            for _, row in g.iterrows():
                paths = self._paths_for_choice(row)

                def _pre(ds):
                    keep = [row["variable_id"]]
                    keep += [c for c in ("time", "lat", "latitude", "lon", "longitude", "x", "y")
                             if c in ds.variables]
                    return ds[sorted(set(v for v in keep if v in ds.variables))]

                ds = xr.open_mfdataset(
                    paths,
                    combine="by_coords",
                    parallel=True,
                    chunks=self.chunks,
                    engine=engine,
                    decode_times=decode_times,
                    preprocess=_pre,
                )
                if time_range is not None and "time" in ds.coords:
                    ds = ds.sel(time=slice(*time_range))
                var_ds.append(ds)

            out[model] = var_ds[0] if len(var_ds) == 1 else xr.merge(
                var_ds,
                compat="override" if drop_conflicts else "no_conflicts",
                combine_attrs="drop_conflicts",
            )
        return out

    def open_model(
        self,
        model: str,
        variables: Optional[List[str]] = None,
        time_range: Optional[Tuple[str, str]] = None,
        engine: Optional[str] = None,
        decode_times: bool = True,
        drop_conflicts: bool = True,
    ) -> xr.Dataset:
        """
        Open a single model: newest + preferred (member/grid) for each requested variable.
        """
        if model not in self._best_df["source_id"].unique():
            raise KeyError(f"Model '{model}' not found. Available: {sorted(self._best_df['source_id'].unique())}")

        sel = self._best_df[self._best_df["source_id"] == model]
        if variables is not None:
            missing = set(variables) - set(sel["variable_id"])
            if missing:
                have = sorted(sel["variable_id"].unique())
                raise ValueError(f"{model}: variables not available: {sorted(missing)}. Available: {have}")
            sel = sel[sel["variable_id"].isin(variables)]

        var_ds: List[xr.Dataset] = []
        for _, row in sel.iterrows():
            paths = self._paths_for_choice(row)

            def _pre(ds):
                keep = [row["variable_id"]]
                keep += [c for c in ("time", "lat", "latitude", "lon", "longitude", "x", "y")
                         if c in ds.variables]
                return ds[sorted(set(v for v in keep if v in ds.variables))]

            ds = xr.open_mfdataset(
                paths,
                combine="by_coords",
                parallel=True,
                chunks=self.chunks,
                engine=engine,
                decode_times=decode_times,
                preprocess=_pre,
                use_cftime=True,
            )
            if time_range is not None and "time" in ds.coords:
                ds = ds.sel(time=slice(*time_range))
            var_ds.append(ds)

        ds_model = var_ds[0] if len(var_ds) == 1 else xr.merge(
            var_ds,
            compat="override" if drop_conflicts else "no_conflicts",
            combine_attrs="drop_conflicts",
        )

        rows = sel
        return ds_model.assign_attrs({
            "model": model,
            "selection_variables": ",".join(sorted(rows["variable_id"].unique())),
            "selection_member_ids": ",".join(sorted(rows["member_id"].unique())),
            "selection_grid_labels": ",".join(sorted(rows["grid_label"].unique())),
            "selection_newest_version": int(rows["version_num"].max()),
        })

    # ---------- ESGF compare (embedded) ----------
    @staticmethod
    def _first(x):
        return x[0] if isinstance(x, list) else x

    @staticmethod
    def _vernum(x):
        try:
            return int(str(x).lstrip("v"))
        except Exception:
            return 0

    @staticmethod
    def _ensure_list(v):
        if v is None:
            return None
        return v if isinstance(v, (list, tuple, set)) else [v]

    @staticmethod
    def _local_latest_df(df: pd.DataFrame) -> pd.DataFrame:
        Ztake._require_cols(df, ["source_id", "experiment_id", "member_id",
                                 "table_id", "variable_id", "grid_label", "version"])
        d = df.copy()
        d["version_num"] = d["version"].apply(Ztake._vernum)
        grp = ["source_id", "experiment_id", "member_id", "table_id", "variable_id", "grid_label"]
        return (d.sort_values(grp + ["version_num"])
                 .drop_duplicates(subset=grp, keep="last"))

    @staticmethod
    def _build_keys_from_df(df: pd.DataFrame, include_version: bool) -> set[str]:
        """
        Build unique comparison keys:
          with version:  model.exp.member.table.var.grid.vYYYYMMDD
          without:       model.exp.member.table.var.grid
        """
        Ztake._require_cols(df, ["source_id", "experiment_id", "member_id",
                                 "table_id", "variable_id", "grid_label"])
        if include_version:
            Ztake._require_cols(df, ["version"])
            keys = df.apply(
                lambda r: ".".join([
                    str(r["source_id"]), str(r["experiment_id"]), str(r["member_id"]),
                    str(r["table_id"]), str(r["variable_id"]), str(r["grid_label"]),
                    f"v{str(r['version']).lstrip('v')}"
                ]),
                axis=1
            )
        else:
            keys = df.apply(
                lambda r: ".".join([
                    str(r["source_id"]), str(r["experiment_id"]), str(r["member_id"]),
                    str(r["table_id"]), str(r["variable_id"]), str(r["grid_label"])
                ]),
                axis=1
            )
        return set(keys.astype(str).to_list())

    @staticmethod
    def _base_and_versions_from_df(df: pd.DataFrame):
        """
        Map (model, exp, member, table, var, grid) -> {versions}
        """
        Ztake._require_cols(df, ["source_id", "experiment_id", "member_id",
                                 "table_id", "variable_id", "grid_label", "version"])
        out: Dict[Tuple[str, str, str, str, str, str], set[str]] = {}
        for _, r in df.iterrows():
            base = (r["source_id"], r["experiment_id"], r["member_id"],
                    r["table_id"], r["variable_id"], r["grid_label"])
            ver = f"v{str(r['version']).lstrip('v')}"
            out.setdefault(base, set()).add(ver)
        return out

    @staticmethod
    def _base_and_versions_from_docs(docs: List[dict]):
        """
        Map ESGF docs -> (model, exp, member, table, var, grid) -> {versions}
        (falls back to parsing instance_id if some fields are missing)
        """
        out: Dict[Tuple[str, str, str, str, str, str], set[str]] = {}
        f = Ztake._first
        for d in docs:
            src  = f(d.get("source_id"))
            exp  = f(d.get("experiment_id"))
            mem  = f(d.get("member_id"))
            tab  = f(d.get("table_id"))
            var  = f(d.get("variable_id"))
            grid = f(d.get("grid_label"))
            ver  = f(d.get("version"))
            if not all([src, exp, mem, tab, var, grid]):
                inst = f(d.get("instance_id"))
                if inst:
                    parts = inst.split(".")
                    if len(parts) >= 10:
                        src, exp, mem, tab, var, grid = parts[3], parts[4], parts[5], parts[6], parts[7], parts[8]
                        ver = parts[9]
            if not all([src, exp, mem, tab, var, grid]) or ver is None:
                continue
            ver = f"v{str(ver).lstrip('v')}"
            out.setdefault((src, exp, mem, tab, var, grid), set()).add(ver)
        return out

    @staticmethod
    def _summarize_version_mismatch(local_map, online_map):
        """
        For bases present on both sides, show version-set differences and who is newer.
        """
        out: Dict[str, Dict[str, Any]] = {}
        fnum = Ztake._vernum
        for base in sorted(set(local_map) & set(online_map)):
            lv, ov = local_map[base], online_map[base]
            if lv == ov:
                continue
            lmax = max((fnum(v) for v in lv), default=0)
            omax = max((fnum(v) for v in ov), default=0)
            status = "local older" if lmax < omax else ("local newer" if lmax > omax else "different sets")
            out[".".join(base)] = {
                "local_versions": sorted(lv),
                "online_versions": sorted(ov),
                "local_max": f"v{lmax}" if lv else None,
                "online_max": f"v{omax}" if ov else None,
                "status": status,
            }
        return out

    @staticmethod
    def _ids_map_from_docs(docs: List[dict]) -> Dict[str, set]:
        """
        Map comparison keys -> ESGF instance_id ONLY.
        key format: model.exp.member.table.var.grid.vYYYYMMDD
        """
        key2inst: Dict[str, set] = {}
        f = Ztake._first
        for d in docs:
            src  = f(d.get("source_id"))
            exp  = f(d.get("experiment_id"))
            mem  = f(d.get("member_id"))
            tab  = f(d.get("table_id"))
            var  = f(d.get("variable_id"))
            grid = f(d.get("grid_label"))
            ver  = f(d.get("version"))
            if not all([src, exp, mem, tab, var, grid, ver]):
                continue
            key  = ".".join([src, exp, mem, tab, var, grid, f"v{str(ver).lstrip('v')}"])
            inst = f(d.get("instance_id"))
            if inst:
                key2inst.setdefault(key, set()).add(inst)
        return key2inst


    @staticmethod
    def _esgf_query(constraints: dict, latest: bool, limit: int, nodes: Optional[List[str]]):
        """
        Query ESGF Solr API across a list of nodes; return first node with results.
        """
        ESGF_NODES_DEFAULT = [
            "https://esgf-node.llnl.gov/esg-search/search/",
            "https://esgf-data.dkrz.de/esg-search/search/",
            "https://esgf-node.ipsl.upmc.fr/esg-search/search/",
            "https://esgf.nci.org.au/esg-search/search/",
            "https://esgf-node.ornl.gov/esg-search/search/",
        ]
        nodes = nodes or ESGF_NODES_DEFAULT
        params = {
            "project": constraints.get("project", "CMIP6"),
            "type": "Dataset",
            "latest": str(latest).lower(),
            "format": "application/solr+json",
            "limit": str(limit),
            "offset": "0",
        }
        for k in ["experiment_id", "variable_id", "member_id", "table_id",
                  "source_id", "grid_label", "institution_id", "activity_id"]:
            v = constraints.get(k)
            if v is not None:
                params[k] = Ztake._ensure_list(v)

        headers = {"Accept": "application/solr+json", "User-Agent": "requests-esgf-compare"}
        last_err = None
        for base in nodes:
            try:
                r = requests.get(base, params=params, headers=headers, timeout=30)
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code} from {base}"; continue
                try:
                    data = r.json()
                except Exception as e:
                    last_err = f"JSON parse failed from {base}: {e}"; continue
                docs = data.get("response", {}).get("docs", [])
                if docs:
                    return base, docs
            except requests.RequestException as e:
                last_err = e; continue
        raise RuntimeError(f"No ESGF node returned usable results. Last error: {last_err}")

    def compare_with_esgf(
        self,
        mode: str = "latest",
        limit: int = 10000,
        nodes: Optional[List[str]] = None,
        extra_constraints: Optional[Dict[str, Any]] = None,
        return_ids: bool = True,
        request_ids: bool = False,  # if True, auto-save only_online_instance_ids
    ) -> Dict[str, Any]:
        """
        Compare local intake (this Ztake's constraints) vs ESGF online.
        Prints a brief console summary and returns a dict of results.
        """
        cons = dict(self.constraints)
        if extra_constraints:
            cons.update(extra_constraints)
        if "variable" in cons and "variable_id" not in cons:
            cons["variable_id"] = cons.pop("variable")

        # local rows and keys
        ds_local = self.cmip6_catalog.search(**cons)
        df_local_all = ds_local.df.copy()
        self._require_cols(df_local_all, ["version", "grid_label"])
        local_models = sorted(df_local_all["source_id"].unique())

        if mode == "latest":
            df_local_latest = Ztake._local_latest_df(df_local_all)
            local_keys = Ztake._build_keys_from_df(df_local_latest, include_version=True)
        elif mode == "ignore_version":
            local_keys = Ztake._build_keys_from_df(df_local_all, include_version=False)
        elif mode == "all_versions":
            local_keys = Ztake._build_keys_from_df(df_local_all, include_version=True)
        else:
            raise ValueError("mode must be one of: 'latest', 'ignore_version', 'all_versions'.")

        # online rows and keys
        node_used, docs = Ztake._esgf_query(cons, latest=(mode != "all_versions"),
                                            limit=limit, nodes=nodes)
        online_models = sorted({Ztake._first(d.get("source_id")) for d in docs if d.get("source_id")})
        online_base_versions = Ztake._base_and_versions_from_docs(docs)
        if mode == "ignore_version":
            online_keys = set(".".join(k) for k in online_base_versions.keys())
        else:
            online_keys = set(".".join((*k, v)) for k, vers in online_base_versions.items() for v in vers)

        # compare + version mismatches
        local_base_versions = Ztake._base_and_versions_from_df(df_local_all)
        only_local  = sorted(local_keys - online_keys)
        only_online = sorted(online_keys - local_keys)
        common      = sorted(local_keys & online_keys)
        version_mismatch = Ztake._summarize_version_mismatch(local_base_versions, online_base_versions)

        # return payload
        result: Dict[str, Any] = {
            "node": node_used,
            "mode": mode,
            "common": common,
            "only_local": only_local,
            "only_online": only_online,
            "local_count": len(local_keys),
            "online_count": len(online_keys),
            "version_mismatch": version_mismatch,
            "local_models": local_models,
            "online_models": online_models,
        }

        # console summary (file remains clean)
        print("Node:", node_used)
        print("Only online dataset:", len(only_online))
        if version_mismatch:
            print("Version mismatches:", len(version_mismatch))

        # include / optionally save official instance IDs only
        if return_ids:
            key2inst = Ztake._ids_map_from_docs(docs)
            only_online_instance_ids = sorted({iid for k in only_online for iid in key2inst.get(k, [])})
            result.update({
                "only_online_instance_ids": only_online_instance_ids,
            })
            if request_ids and only_online_instance_ids:
                Ztake.save_ids_to_file(
                    only_online_instance_ids,
                    filename="only_online_instance_ids.txt",
                    prefix="instance_id",
                    add_header=False,
                    add_request_note=False,
                )

        return result

    # ---------- file output ----------
    @staticmethod
    def save_ids_to_file(
        ids: List[str],
        filename: str,
        prefix: str = "instance_id",
        add_header: bool = False,
        add_request_note: bool = False,
    ) -> str:
        """
        Save IDs to file as 'instance_id=<id>' per line. Returns absolute path.
        """
        import os
        if prefix != "instance_id":
            raise ValueError("This project saves instance IDs only. Use prefix='instance_id'.")
        abspath = os.path.abspath(filename)
        with open(abspath, "w") as f:
            for _id in ids:
                f.write(f"{prefix}={_id}\n")
        print(f"✅ Saved {len(ids)} IDs to {abspath}")
        print("Next: open https://help.nci.org.au/ and attach this file to your help request,")
        print("or email help@nci.org.au and attach the file.")
        return abspath
def get_lat_lon_coords(dataset: xr.Dataset) -> tuple:
    """
    Determine the latitude and longitude coordinate names in an xarray dataset.

    Args:
        dataset (xr.Dataset): The dataset to check.

    Returns:
        tuple: A tuple containing the names of the latitude and longitude coordinates (lat_coord, lon_coord).
    """
    lat_coord = (
        'nav_lat' if 'nav_lat' in dataset.coords else
        'latitude' if 'latitude' in dataset.coords else
        'lat' if 'lat' in dataset.coords else None
    )
    lon_coord = (
        'nav_lon' if 'nav_lon' in dataset.coords else
        'longitude' if 'longitude' in dataset.coords else
        'lon' if 'lon' in dataset.coords else None
    )

    if not lat_coord or not lon_coord:
        raise ValueError("Latitude and/or longitude coordinates not found in the dataset.")

    return lat_coord, lon_coord

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath

def plot_south_polar_auto(
    da,
    title=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    label="",
    figsize=(6, 6),
    add_contour=None,
    contour_levels=None,
    contour_color="k",
):
    """
    Universal South Polar plot for ANY DataArray.
    Automatically detects lat/lon names and 1D vs 2D grids.

    Parameters
    ----------
    da : xr.DataArray               2D field (lat, lon or y,x)
    add_contour : xr.DataArray      Optional contour overlay (e.g., SIC)
    """

    # ---------------------------------------------------------
    # Auto-detect lat/lon coordinate names
    # ---------------------------------------------------------

    # common name patterns
    lat_keys = ["lat", "latitude", "nav_lat", "TLAT", "yt_ocean", "Y"]
    lon_keys = ["lon", "longitude", "nav_lon", "TLONG", "xt_ocean", "X"]

    lat_name = None
    lon_name = None

    for key in lat_keys:
        if key in da.coords:
            lat_name = key
            break

    for key in lon_keys:
        if key in da.coords:
            lon_name = key
            break

    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Could not find lat/lon names in DataArray coords {list(da.coords)}"
        )

    # ---------------------------------------------------------
    # Round 2D or 1D coordinate arrays?
    # ---------------------------------------------------------

    lats = da[lat_name]
    lons = da[lon_name]

    # If longitude contains negatives, wrap to [0, 360]
    if lons.min() < 0:
        lons = (lons + 360) % 360

    # ---------------------------------------------------------
    # Create figure and axis
    # ---------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(
        1, 1, 1,
        projection=ccrs.Orthographic(central_longitude=0, central_latitude=-90)
    )

    # ---------------------------------------------------------
    # Main plot
    # ---------------------------------------------------------
    im = da.plot(
        x=lon_name,
        y=lat_name,
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )

    # ---------------------------------------------------------
    # Optional contour overlay (e.g., SIC)
    # ---------------------------------------------------------
    if add_contour is not None:
        ax.contour(
            add_contour[lon_name], add_contour[lat_name], add_contour,
            levels=contour_levels,
            colors=contour_color,
            linewidths=0.8,
            transform=ccrs.PlateCarree(),
        )

    # ---------------------------------------------------------
    # Map polish
    # ---------------------------------------------------------
    ax.coastlines(linewidth=0.7)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.set_extent([-180, 180, -90, -30], crs=ccrs.PlateCarree())

    # Circular boundary (beautiful polar mask)
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = mpath.Path(
        np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5
    )
    ax.set_boundary(circle, transform=ax.transAxes)

    ax.gridlines(linestyle=":", linewidth=0.5, draw_labels=False)

    # ---------------------------------------------------------
    # Colorbar
    # ---------------------------------------------------------
    cbar_ax = fig.add_axes([0.25, 0.08, 0.55, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(label, fontsize=11)

    ax.set_title(title if title else (da.name or ""), fontsize=12)

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.show()
