# Codebase Issue Backlog (Proposed Tasks)

## 1) Typo fix task
**Task:** Correct user-facing typos in `src/wmt.py` docstrings/comments.

**Why:** There are obvious spelling mistakes such as:
- `Gitrs SeaWater` (should be `Gibbs SeaWater`/`GSW`).
- `Anveraged Buoyancy flux` (should be `Averaged Buoyancy flux`).
- `Calaculate` (should be `Calculate`).

**Acceptance criteria:**
- All identified typos are corrected without changing runtime behavior.
- Run a quick lint/format pass to ensure no syntax or style regressions.

---

## 2) Bug fix task
**Task:** Harden `Zlook.smart_open` in `src/zclef_v2.py` to only open NetCDF files and fail clearly when none match.

**Why:** `smart_open()` currently uses `file_in_path()` which returns every directory entry, not only `.nc` files. This can pass non-NetCDF items to `xr.open_mfdataset`, causing runtime failures. It also does not guard against an empty `files_to_open` list.

**Acceptance criteria:**
- Filter inputs to `.nc` files only (or reuse `kw_search`).
- Raise a clear exception when no files fall in range.
- Add regression tests for non-NetCDF entries and empty selections.

---

## 3) Documentation discrepancy task
**Task:** Align `buoyancy_flux` parameter docs with function signature in `src/wmt.py`.

**Why:** The `p` parameter description says "Default value is 0", but the function signature requires `p` and has no default. This is misleading for users.

**Acceptance criteria:**
- Either add a true default (`p=0`) in code **or** remove/correct the default claim in documentation.
- Update any related helper docs/examples for consistency.

---

## 4) Test improvement task
**Task:** Add a first `pytest` suite for parsing and coordinate-detection behavior.

**Why:** The repository currently has no automated tests, and key utility logic is unprotected.

**Suggested coverage:**
- `Zlook.load_data_range()` year-window filtering and malformed filename handling.
- `get_lat_lon_coords()` precedence (`nav_lat/nav_lon` > `latitude/longitude` > `lat/lon`) and failure path.
- `load_data_in_range()` with mocked `xr.open_dataset`/`xr.concat` to validate decade-chunk logic.

**Acceptance criteria:**
- New tests run with `pytest` and pass locally.
- At least one negative-path test per function.
