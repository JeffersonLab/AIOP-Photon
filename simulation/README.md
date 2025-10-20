# AIOP-Photon Cobrem Simulation

This project contains the local standalone version of the GlueX Coherent
Bremsstrahlung simulation and visualization tools originally used in the
SpotFinder web service. It includes the C++ components
(`CobremsGeneration`, `Map2D`, `Couples`, and `rootvisuals`) and a Python
wrapper (`sim_wrapper.py`) for running simulations directly from the
command line.

---

## INSTALLATION

1. Clone the repository and move into the simulation directory:

    ```bash
    git clone https://github.com/JeffersonLab/AIOP-Photon.git
    cd AIOP-Photon/simulation
    ```

2. Build the C++ libraries using ROOT's ACLiC system:

    ```bash
    root -l -b -q makerootvisuals.C
    ```

    This will produce shared libraries and PCM dictionary files such as:

    Map2D_cc.so
    Couples_C.so
    CobremsGeneration_cc.so
    rootvisuals_C.so
    *_ACLiC_dict_rdict.pcm
    
---

## RUNNING THE SIMULATION

The main Python entry point is `sim_wrapper.py`.

Example usage:

```python
python sim_wrapper.py \
  --nproc 4 \
  --edge 8.5 \
  --config PARA \
  --phi 0/90 \
  --diamond-range -1 1 40
```

### Command-line options

| Option | Description |
|---------|-------------|
| `--nproc` | Number of CPU cores to use |
| `--edge` | Nominal coherent edge energy (GeV) |
| `--config` | Polarization geometry (`"PARA"` or `"PERP"`) |
| `--phi` | Crystal azimuth (`"0/90"` or `"45/135"`) |
| `--diamond-range` | Diamond tilt scan: start end steps (mrad) |

---

### Simulation Output

Output files are saved as:

coherent_peaks_<CONFIG>_<PHI>.csv

---