# segbench

Code of *Measuring Segregation in Heterogeneous Robot Swarms: Formal Properties,
Continuous Scores, and a Benchmark Evaluation* (Cruz, Santos and Macharet).
It implements the metrics, the benchmark with certified ground truth, the
canonical configurations S1-S14, four controller ports, and every result,
table, figure and video of the paper.

## Run

Everything goes through `run.sh`:

```bash
./run.sh build                      # build container/segbench.sif (apptainer --fakeroot), once
./run.sh -c -d --jobs 16            # whole pipeline in the container, detached
./run.sh status                     # processes of the run and last log lines
./run.sh stop                       # stop the run and all its workers
tail -f logs/latest.log             # follow the log
```

| Command | Effect |
|---|---|
| `./run.sh [--quick] [--jobs N]` | whole pipeline (scripts 01 to 11) |
| `./run.sh scripts/03_noise.py --jobs N` | one script |
| `./run.sh test` | tests of the formal properties |
| `-c` (first) | run inside `container/segbench.sif` |
| `-d` (first) | run detached (`setsid nohup`), output in `logs/run_<date>.log` |
| `build`, `status`, `stop` | build the image, show the latest run, stop it |

`--quick` runs a reduced configuration in a few minutes. Without `-c`, install
the dependencies first (`pip install -r requirements.txt`, Python 3.10+); the
PSO controller is then skipped unless `rvo2` is installed. `--jobs N` sets the
number of worker processes, each with one thread.

## Pipeline

| Script | Paper | Output |
|---|---|---|
| `00_extract_canonical.py` | Sect. 6.1 | `data/canonical/S*.csv` (shipped, recovered from the original figures) |
| `01_generate_benchmark.py` | Sects. 6.2-6.3 | `data/benchmark.pkl.gz`, `results/benchmark_index.csv`, `results/infeasible.csv` |
| `02_evaluate.py` | Sects. 7.1-7.2 | `results/metrics_benchmark.csv.gz`, `results/metrics_canonical.csv` |
| `03_noise.py` | Sect. 7.4, Prop. 3 | `results/noise.csv.gz` |
| `04_sensitivity.py` | Sect. 7.5 | `results/sensitivity.csv.gz` |
| `05_runtime.py` | Sect. 7.6 | `results/runtime.csv` |
| `06_agreement.py` | Sect. 7.7 | `results/agreement_*.csv` |
| `07_controllers.py` | Sects. 6.4, 7.8 | `data/controllers/`, `results/controllers.csv.gz` |
| `08_figures.py` | all result figures | `figures/` |
| `09_tables.py` | Tables 5 and 6 | `tables/tab_binary.tex`, `tables/tab_continuous.tex`, `tables/numbers.tex` |
| `10_videos.py` | supplementary material | `videos/` |
| `11_validate.py` | Props. 1-4, Sect. 5.3 | `results/validation.json` (fails if a check fails) |

The two `.tex` tables replace the placeholder tables of the manuscript
directly. Parameters are in `configs/default.yaml` (Table 4) and
`configs/quick.yaml`.

## Notes

* **Ground truth** never uses the minimum spanning trees evaluated by the
  metrics. Positive labels need witness trees (Definition 1), negative labels a
  disconnected proximity graph or at least four bichromatic hull edges
  (Props. 2 and 4); other instances are indeterminate.
* **Infeasible sizes.** Rings and arcs of 10 agents, and ten rings or arcs of
  30, cannot be cohesive at rho = 10 r and are listed in `results/infeasible.csv`.
* **Canonical configurations** are recovered exactly from the vector figures and
  by colour segmentation from the raster ones, then rescaled to a median
  nearest-neighbour distance of 4 r.
* **CV is not rotation invariant**: the Moreira-Santos walk starts at the lowest
  point with a fixed direction.

## Controllers

Python ports of the C++/ROS code of Rezeck et al. (ICRA 2021),
<https://github.com/verlab/2021-icra-grf-swarm>, used with the authorization of
its author. Parameters and initial states follow the reference. Deviations:

| Controller | Deviation |
|---|---|
| Santos et al. (2020) | relative velocity subtracted as in Eq. (2) of the paper (the reference adds it); `santos_velocity_sign: code` restores the reference |
| Mitrano et al. (2019) | sensor angle wrapped to (-pi, pi] |
| Inácio et al. (2019) | ORCA from Python-RVO2; neighbour memories read from the previous step |
| Rezeck et al. (2021) | wall points from the current position (difference below 0.02 m) |

## Licence

MIT (`LICENSE`). Please cite the paper (`CITATION.cff`) and, for the
controllers, the original works.
