more el notes - 
 - conda was the only thing that worked for me when running this; if you guys are using a different system then you'll have to rework some of the command line stuff
  - apart from training, step 01 took the longest. unless you really want to try a different set of data **you can just skip to the training step and work only with the neural network** and not the bullshit before it. the parsed data is already stored. i don't think the data/data parsing will help improve the accuracy but its up to you
   - there are two models in this code that we can run: there's the ESM-2 35M, which has 35M parameters, and the ESM-2 650M, which has 650 parameters. that's really the only difference, the 650M has way more layers and is **better at gauging long-term relationships.** presumably the 650M will work better but i'm not about to run all that on my CPU lmao. so ideally we run that one but if it's too big that's fine. for our task it only adds a few percentage points of accuracy (allegedly)
 - if the troubleshooting guide that claude made is not helping lmk i can probably help with the package/dependency stuff





# Setup and Troubleshooting Guide

End-to-end instructions for reproducing this protein secondary structure
prediction pipeline on a Windows machine using Conda. A separate
troubleshooting section at the bottom documents every issue we hit during
development and the fix for each one.

---

## A. Step-by-step replication

### Prerequisites

- **Windows 10/11**
- **Miniconda** or **Anaconda** installed
  (download from https://docs.conda.io/en/latest/miniconda.html)
- **Microsoft Visual C++ Redistributable** installed
  (download from https://aka.ms/vs/17/release/vc_redist.x64.exe)
- ~5 GB disk space for PDB files, embeddings, and model checkpoints

### Step 0: Set up the conda environment

Open the **Anaconda Prompt** (search for it in the Start menu) and run:

```cmd
conda create -n ssproj python=3.11 -y
conda activate ssproj
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fair-esm numpy
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
conda deactivate
conda activate ssproj
```

Note: we deliberately install PyTorch and NumPy via **pip**, not conda. The
conda versions on Windows have DLL conflicts that cause both `shm.dll`
errors and OpenMP runtime conflicts.

### Step 1: Get the DSSP binary

Download `mkdssp-4.4.0.exe` (or latest) from:
https://github.com/PDB-REDO/dssp/releases

Save it to a stable path, e.g. `C:\Users\<you>\tools\mkdssp.exe`. Test that
it runs:

```cmd
C:\Users\<you>\tools\mkdssp.exe --version
```

You should see a version number printed. If it complains about missing
DLLs, install the Visual C++ Redistributable (link above).

You can also add the folder containing `mkdssp.exe` to your PATH so you
don't need the full path every time, but specifying `--dssp` explicitly is
fine.

### Step 2: Get a non-redundant PDB chain list

Download a PISCES culled-PDB list from
https://dunbrack.fccc.edu/lab/pisces_culledpdb. Recommended choice:

```
cullpdb_pc25.0_res0.0-2.5_noBrks_len40-10000_R0.3_Xray_d<DATE>_chains<N>.fasta
```

This filters for non-redundant chains (≤25% sequence identity, ≤2.5 Å
resolution, no chain breaks, X-ray only). Place the file in your project
directory.

### Step 3: Place project files

Create or `cd` to your project directory. It should contain:

```
project/
├── 00_parse_pisces.py
├── 01_prepare_data.py
├── 02_embed_with_esm.py
├── 03_model.py
├── 04_dataset.py
├── 05_train.py
├── 06_predict.py
└── cullpdb_pc25.0_res0.0-2.5_noBrks_len40-10000_R0.3_Xray_d<...>.fasta
```

### Step 4: Parse the PISCES file into a PDB list

```cmd
python 00_parse_pisces.py --pisces_file cullpdb_pc25.0_res0.0-2.5_noBrks_len40-10000_R0.3_Xray_d<DATE>_chains<N>.fasta --out pdb_list.txt --n 150 --shuffle
```

Expected output: `Wrote 150 entries to pdb_list.txt`

### Step 5: Download structures and run DSSP

```cmd
python 01_prepare_data.py --pdb_list pdb_list.txt --dssp "C:\Users\<you>\tools\mkdssp.exe"
```

(Drop `--dssp ...` if mkdssp is on your PATH.)

This takes ~5–10 minutes:
- Downloads `.cif` (mmCIF) files from RCSB into `data\pdbs\`
- Runs DSSP on each one
- Extracts the requested chain
- Maps the 8-state DSSP output to 3-state (H/E/C)
- Writes `data\labels.tsv`

Confirm yield:

```cmd
type data\labels.tsv | find /c /v ""
```

Should show 100+ lines (header + ~120–140 successful proteins). Some
failures are expected because of large/unusual structures.

### Step 6: Cache ESM-2 embeddings

For a faster CPU-only run, use the small 35M model:

```cmd
python 02_embed_with_esm.py --model_name esm2_t12_35M_UR50D
```

For better accuracy on a GPU machine, use the default 650M model:

```cmd
python 02_embed_with_esm.py
```

The script downloads the ESM-2 weights on first run (~150 MB for the small
model, ~2.6 GB for 650M) and caches per-residue embeddings to
`data\embeddings\<pdb>_<chain>.pt`.

### Step 7: Train the model

If you used the **35M** model (480-dim embeddings):

```cmd
python 05_train.py --epochs 30 --esm_dim 480 --num_workers 0
```

If you used the **650M** model (1280-dim, the default):

```cmd
python 05_train.py --epochs 30 --num_workers 0
```

To save the full terminal output to a file as well:

```cmd
python 05_train.py --epochs 30 --esm_dim 480 --num_workers 0 > runs\exp1\train_log.txt 2>&1
```

Training takes ~3–10 minutes. Results saved to:
- `runs\exp1\best.pt` — best checkpoint by validation Q3
- `runs\exp1\metrics.json` — full per-epoch history + final test metrics

### Step 8: Predict on a new sequence

```cmd
python 06_predict.py --checkpoint runs\exp1\best.pt --esm_model esm2_t12_35M_UR50D --sequence MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK
```

Output:

```
Seq: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK
SS:  CCHHHHHHHHHHHHHCCCCCCCHHHHHCHHCHCHHHCHCCCCCCCCCCCCHCC
```

Each character in the SS line is the predicted secondary structure for the
amino acid directly above it: `H` = α-helix, `E` = β-strand, `C` = coil.

---

## B. Troubleshooting

Every issue we ran into during development, in roughly the order it
appeared. Each entry has the symptom, the root cause, and the fix.

### B.1. PowerShell rejects backslash line continuations

**Symptom:**

```
At line:2 char:7
+    --pisces_file cullpdb_pc25.0_res0.0-2.5_noBrks_len40-10000_R0.3_X ...
+      ~
Missing expression after unary operator '--'.
```

**Cause:** The line-continuation character `\` is for bash/Linux. PowerShell
uses backtick `` ` `` instead.

**Fix:** Either put the whole command on one line, or use backtick:

```powershell
python 00_parse_pisces.py `
    --pisces_file cullpdb_pc25.0_res0.0-2.5_noBrks_len40-10000_R0.3_Xray_d<DATE>_chains<N>.fasta `
    --out pdb_list.txt --n 150 --shuffle
```

The backtick must be the **last** character on the line — no trailing
spaces.

### B.2. Step 02 reports "Embedding 0 sequences"

**Symptom:**

```
Loading esm2_t12_35M_UR50D...
Using representations from layer 12.
Embedding 0 sequences...
Done. Embeddings saved to data\embeddings
```

**Cause:** `data\labels.tsv` is empty or contains only the header row.
This means step 01 silently skipped every protein, almost always because
DSSP wasn't installed correctly.

**Fix:** Verify DSSP works before running step 01:

```cmd
mkdssp --version
```

If that errors, install DSSP (see step 1 above). Then re-run step 01.

### B.3. Step 01 fails with "system cannot find the file specified" (DSSP)

**Symptom:**

```
[1/150] 8ajp_A
  [warn] DSSP failed on 8ajp.pdb: [WinError 2] The system cannot find the file specified
```

**Cause:** Python's subprocess can't find the `mkdssp` executable, even
though your shell can.

**Fix:** Pass the full path to `--dssp`:

```cmd
python 01_prepare_data.py --pdb_list pdb_list.txt --dssp "C:\Users\<you>\tools\mkdssp.exe"
```

Or, if using conda's DSSP, find where it's installed:

```cmd
where.exe mkdssp
```

and use that path.

### B.4. `conda install -c bioconda dssp` fails with "PackagesNotFoundError"

**Symptom:**

```
PackagesNotFoundError: The following packages are not available from current channels:
  - dssp
```

**Cause:** The `bioconda` channel doesn't have a Windows build of DSSP.
Most bioconda packages are Linux/macOS-only.

**Fix:** Skip conda for DSSP entirely and use the Windows binary from
PDB-REDO releases (see step 1). This is more reliable on Windows than any
conda channel.

### B.5. mkdssp says "This file does not seem to be an mmCIF file"

**Symptom:**

```
[1/150] 8ajp_A
  [warn] mkdssp returned 1: parse error at line 1: This file does not seem to be an mmCIF file
```

**Cause:** Modern mkdssp builds on Windows only parse mmCIF input
reliably. Auto-detection from file extension doesn't work consistently.
Some builds also lack the `--input-format` flag.

**Fix:** Download `.cif` (mmCIF) files instead of `.pdb` files. Already
fixed in `01_prepare_data.py` — it now prefers `.cif` downloads from RCSB.
Make sure you're using the latest version of the script. If you have old
`.pdb` files cached:

```cmd
del data\pdbs\*.pdb
del data\labels.tsv
python 01_prepare_data.py --pdb_list pdb_list.txt
```

### B.6. mkdssp rejects `--input-format` flag

**Symptom:**

```
[warn] mkdssp returned 1: while parsing command line arguments, option 'input-format': unknown option
```

**Cause:** Different mkdssp builds support different command-line options.
Your build doesn't have `--input-format`.

**Fix:** Use the latest `01_prepare_data.py` — it doesn't pass that flag.
Format selection is handled by downloading mmCIF files (see B.5).

### B.7. PDB list has corrupted tab characters

**Symptom:**

```
[150/150] 4wks  C
  [warn] failed to download 4wks       c: URL can't contain control characters. '/download/4wks\tc.pdb' (found at least '\t')
```

**Cause:** Some pdb_list entries have escaped tab characters or trailing
control characters that the splitter doesn't handle cleanly.

**Fix:** Already handled in the latest `01_prepare_data.py`. The
`read_pdb_list` function strips control characters and tolerates both real
tabs and escaped `\t` literals.

### B.8. OpenMP runtime conflict

**Symptom:**

```
OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
```

**Cause:** Multiple OpenMP runtimes get linked into one Python process when
PyTorch and NumPy/MKL are both present. Common in conda environments on
Windows.

**Fix:** Set `KMP_DUPLICATE_LIB_OK=TRUE` before running. Three ways:

**A) Inline in scripts (already done):** the top of `02_embed_with_esm.py`,
`05_train.py`, and `06_predict.py` all set this env var before importing
torch.

**B) Per-session in cmd:**

```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
```

**C) Permanent for the conda env (recommended):**

```cmd
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
conda deactivate
conda activate ssproj
```

### B.9. PyTorch fails to import with "shm.dll" error

**Symptom:**

```
OSError: [WinError 127] The specified procedure could not be found. Error loading "C:\Users\elena\anaconda3\envs\ssproj\Lib\site-packages\torch\lib\shm.dll" or one of its dependencies.
```

**Cause:** Conda's PyTorch build on Windows has unreliable DLL
dependencies — particularly when conda's NumPy (with MKL) is present
alongside.

**Fix:** Reinstall PyTorch from pip:

```cmd
pip uninstall torch torchvision torchaudio -y
pip cache purge
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If the error persists after PyTorch reinstall, the culprit is conda's
NumPy. Reinstall it from pip too:

```cmd
pip uninstall numpy -y
pip install numpy
```

Verify with:

```cmd
python -c "import torch; from torch.utils.data import DataLoader; print('OK')"
```

If the import still fails, the conda env is too broken. Recreate from
scratch (your `data\` folder is preserved):

```cmd
conda deactivate
conda env remove -n ssproj -y
conda create -n ssproj python=3.11 -y
conda activate ssproj
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fair-esm numpy
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
conda deactivate
conda activate ssproj
```

### B.10. `pip install biopython` fails to build biotraj

**Symptom:**

```
× Failed to build installable wheels for some pyproject.toml based projects
╰─> biotraj
```

**Cause:** Recent Biopython versions depend on `biotraj`, which needs a C
compiler (not available on stock Windows).

**Fix:** We don't actually use Biopython in the final pipeline (the script
calls `mkdssp` directly via subprocess). Don't install Biopython at all.
If something transitively pulls it in, pin to an older version:

```cmd
pip install "biopython<1.85"
```

### B.11. `$env:` syntax errors in cmd.exe

**Symptom:**

```
'$env:KMP_DUPLICATE_LIB_OK' is not recognized as an internal or external command
```

**Cause:** `$env:NAME = "value"` is **PowerShell** syntax. The standard
Anaconda Prompt is **cmd.exe**, which uses different syntax.

**Fix:** In cmd.exe, use `set`:

```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
```

(No spaces around `=`.)

In PowerShell (Anaconda PowerShell Prompt):

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

To check which shell you're in: `echo %COMSPEC%` (returns a path in
cmd.exe) versus `$PSVersionTable` (returns version info in PowerShell).

### B.12. Nested tensor warning during training

**Symptom:**

```
UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future.
```

**Cause:** Cosmetic warning from PyTorch's `nn.TransformerEncoder` about
internal use of an experimental API for variable-length sequences.

**Fix:** No action needed — it's harmless and only prints once per
process. To silence it for cleanliness, add to `05_train.py` after the
imports:

```python
import warnings
warnings.filterwarnings("ignore", message=".*nested tensor.*")
```

### B.13. ESM dimension mismatch at training time

**Symptom:** Training crashes with a shape error referencing 480 vs 1280
or similar.

**Cause:** Embeddings were cached with one ESM model size but `--esm_dim`
default doesn't match.

**Fix:** Pass the correct dimension:

| ESM model | `--esm_dim` |
|-----------|-------------|
| `esm2_t12_35M_UR50D` | `480` |
| `esm2_t30_150M_UR50D` | `640` |
| `esm2_t33_650M_UR50D` | `1280` (default) |
| `esm2_t36_3B_UR50D` | `2560` |

Same for `06_predict.py` — use `--esm_model` matching whatever you used
for training:

```cmd
python 06_predict.py --checkpoint runs\exp1\best.pt --esm_model esm2_t12_35M_UR50D --sequence ...
```

---

## C. Sanity-check sequences for `06_predict.py`

Useful sequences for verifying the trained model produces reasonable
output:

**Myoglobin (1MBN, all α-helical, 153 residues)** — expect mostly long `H`
stretches:

```
MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRVKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGNFGADAQGAMNKALELFRKDIAAKYKELGYQG
```

**Ubiquitin (1UBQ, β-grasp fold, 76 residues)** — expect alternating `E`
and `C` with one `H` region:

```
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
```

If the model produces mostly `C` for both of these, something's wrong with
the training. If it correctly assigns `H` to myoglobin and `E` to
ubiquitin, the pipeline is working.

---

## D. Expected results

With the small ESM-2 35M model and ~120-140 training proteins:

- **Test Q3:** 0.78–0.84
- **H precision/recall:** 0.85+
- **E precision/recall:** 0.70+
- **C precision/recall:** 0.75+

The 650M model gets 1–3 points higher Q3 but needs more memory and a GPU
to run quickly. Below 0.70 Q3 indicates a problem (most likely a
training-set issue).
