# MBeRCS Water Quality Explorer

A [Quarto](https://quarto.org) website for exploring the discrete water-quality
dataset from the **MBON Bi-Monthly Research Cruises for SFER (MBeRCS)** program
(Florida Keys and west Florida shelf). The pages walk through data coverage, a
sampling map, per-parameter statistics, seasonality, vertical structure and
interannual trends, mirroring the results sections of the project outline
(`../MBeRCS-outline.md`).

## Data

The site reads a single CSV from the repository's `data/` directory:

```
<repo>/data/SFER_data.csv      # this site expects it here, one level up
```

From inside this `quarto-analysis/` folder that file is `../data/SFER_data.csv`.

It is **not** committed here. Download it from the upstream repository and save
it as `../data/SFER_data.csv`:

```bash
mkdir -p ../data
curl -L -o ../data/SFER_data.csv \
  https://github.com/Ecosystem-Assessment-Lab/SFER/blob/main/DATA/SFER_data.csv
```

> The `blob/...` link is a GitHub web page. If `curl` saves an HTML file instead
> of the CSV, use the raw URL
> `https://raw.githubusercontent.com/Ecosystem-Assessment-Lab/SFER/main/DATA/SFER_data.csv`
> or the "Download raw file" button on that page.

The upstream `Ecosystem-Assessment-Lab/SFER` repository is not publicly
readable (it returns 404 to anonymous requests), so obtaining the file may
require GitHub access to that organization. The site itself only needs the CSV
to be present at `../data/SFER_data.csv`; it does not care how it got there.

If the file is missing, every page fails fast with a message pointing back here.

## Rendering

Requires Quarto (tested with 1.8) and a Python environment with `pandas`,
`numpy`, `matplotlib`, `statsmodels` and `scipy` (the `python3` Jupyter
kernel). `statsmodels` and `scipy` are used only by the harmonic / spectral
model at the end of the trends page; the rest of the site needs only the first
three.

```bash
quarto preview     # live-reloading local preview
quarto render      # build the static site into _site/
```

Then open `_site/index.html`.

## Layout

```
quarto-analysis/
├── _quarto.yml            # site configuration (navbar, theme, jupyter engine)
├── index.qmd              # landing page / overview
├── coverage.qmd           # sampling effort and analyte availability
├── map.qmd                # sampling maps (uses assets/florida-land.geojson)
├── parameters.qmd         # per-parameter summary statistics and distributions
├── seasonality.qmd        # monthly / seasonal cycles
├── vertical.qmd           # surface–mid–bottom structure; oxygen example
├── trends.qmd             # interannual trends, seasonal decomposition, harmonic model
├── sfer_common.py         # shared data loading, metadata and plotting helpers
├── styles.css             # light custom styling
└── assets/
    └── florida-land.geojson   # small Natural Earth land extract for the maps
```

`assets/florida-land.geojson` is a cropped, simplified extract of the public
domain [Natural Earth](https://www.naturalearthdata.com/) 1:10m land polygons.
It is vendored so the maps render without a network connection.

## Caveats

- The `*_flag` quality-control columns are shown but **not** used to exclude
  data: their codebook is maintained upstream and is not bundled with the CSV.
- A few physically impossible values exist (e.g. `o2_ctd` in the thousands of
  mg/L); these are filtered out of the relevant plots but kept in the summary
  tables.
- Every page loops over **all 12 analytes**, including the two sparse oxygen
  series. CTD oxygen (`o2_ctd`) only covers 2022–2026 and Winkler oxygen
  (`o2_disc`) only five non-consecutive years (2014–2017, 2025–2026); the pages
  flag where this makes a figure indicative rather than conclusive.
- The `execute: freeze: auto` setting caches executed notebooks in `_freeze/`.
  Because the pages depend on the CSV rather than only on the `.qmd` source,
  re-render with `quarto render` after the data changes (delete `_freeze/` to
  force a full re-execution).
