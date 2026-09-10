## Pipeline Stages

| Step | Script | Purpose | Input | Output |
|------|--------|---------|-------|--------|
| 1 | `01_sky_subtraction.py` | Subtract sky from each OB cube using `kmos_sky_tweak` + `kmos_combine` | Reduced `SINGLE_CUBES_KMOS` cubes | Sky-subtracted cubes (`SKY_TWEAK` → combined) |
| 2 | `02_auto_astrometry.py` | Automatic astrometric correction | | |
| 3 | `03_manual_astrometry.py` | Manual (interactive) astrometric correction | | |
| 4 | `04_extract_spectra.py` | Extract 1D spectra | | |

---

### 1. Sky Subtraction — `01_sky_subtraction.py`

Subtracts the sky from each **OB** (observation block) cube using the ESO
recipe `kmos_sky_tweak`, then combines the resulting tweaked cubes with
`kmos_combine`.

**What the script does**

1. For a given pointing (e.g. `P1`) and a list of concatenation folders
   (e.g. `con1`, `con2`…), it locates the reference **sky** cube inside a
   `Sky/` subfolder.
2. It then discovers every **OB** subfolder inside each concatenation and
   finds the corresponding `*SINGLE_CUBES_KMOS*.fits` science cube.
3. For each OB it:
   - creates a clean output folder `sky_tweak/`,
   - writes a `single_cubes.sof` (object + sky cube),
   - runs `esorex kmos_sky_tweak single_cubes.sof`,
   - writes a `decombine.sof` pointing to the resulting `SKY_TWEAK.fits`,
   - runs `esorex kmos_combine --method=header decombine.sof`.

**Input**

- A directory tree under `base_path`:
  ```
  <base_path>/
    ├── con1/
    │   ├── Sky/
    │   │   └── <date>/KMOS.*_tpl/*SINGLE_CUBES_KMOS*.fits
    │   ├── OB1/
    │   │   └── <date>/KMOS.*_tpl/*SINGLE_CUBES_KMOS*.fits
    │   └── OB2/ ...
    └── con2/ ...
  ```

**Output**

Inside every `OB*/` folder:
```
OB1/
  └── sky_tweak/
      ├── single_cubes.sof
      ├── SKY_TWEAK.fits
      ├── decombine.sof
      └── <combined output>.fits
```

**Configuration (top of the script)**

| Variable | Meaning |
|----------|---------|
| `pointing` | Pointing name, e.g. `"P0"`, `"P1"` |
| `cons` | List of concatenation folders to process, e.g. `["con1", "con2"]` |
| `SKY_FOLDER` | Name of the sky subfolder inside each concatenation (default `"Sky"`) |
| `base_path` | Root path to the reduced data |
| `ESOREX` | Full path to the `esorex` executable |

**Run**

```bash
python pipeline/01_sky_subtraction.py
```

**Requirements**

- `esorex` installed and the KMOS recipes available
  (`kmos_sky_tweak`, `kmos_combine`).
- Python ≥ 3.9 (standard library only — no external dependencies).

**Notes / caveats**

- `os.chdir()` is used inside the script; make sure you don't have other
  code relying on the working directory afterwards.
- Directories named `sky_tweak` and `tmp` are ignored when searching for
  cubes to avoid picking up previous outputs.
- Any OB whose cube cannot be found is skipped with a printed error, so a
  partial run never aborts the whole loop.
