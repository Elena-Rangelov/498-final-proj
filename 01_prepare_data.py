"""
Step 1: Download PDB structures and extract per-residue secondary structure
        labels with DSSP, for a *specific chain* per entry.

Input:
    pdb_list.txt — one entry per line, "<pdb_id>\\t<chain>"
                    (produced by 00_parse_pisces.py)

For each (pdb_id, chain):
  - Try to download <pdb_id>.pdb; fall back to <pdb_id>.cif if no PDB exists
    (large/recent structures are mmCIF-only).
  - Run mkdssp directly via subprocess to produce DSSP output (mmCIF format).
  - Parse the mmCIF output to get (residue, chain, 8-state SS) triples.
  - Filter to the requested chain and map 8-state -> 3-state (H, E, C).
  - Write a row to labels.tsv.

We bypass Biopython's DSSP wrapper because it's flaky on Windows with
DSSP 4.x. Calling mkdssp directly and parsing its output is more reliable.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError


DSSP_8_TO_3 = {
    "H": "H", "G": "H", "I": "H",
    "E": "E", "B": "E",
    "T": "C", "S": "C", "-": "C", " ": "C", "P": "C",
}

# Three-letter -> one-letter amino acid codes
AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def download_structure(pdb_id: str, out_dir: Path) -> tuple[Path, str] | None:
    """
    Try to fetch <pdb_id>.pdb, then fall back to <pdb_id>.cif.
    Returns (path, file_format) where file_format is 'pdb' or 'mmcif',
    or None on failure.
    """
    pdb_id = pdb_id.lower()
    pdb_path = out_dir / f"{pdb_id}.pdb"
    cif_path = out_dir / f"{pdb_id}.cif"

    if pdb_path.exists():
        return pdb_path, "pdb"
    if cif_path.exists():
        return cif_path, "mmcif"

    # Try .pdb first
    try:
        urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", pdb_path)
        return pdb_path, "pdb"
    except HTTPError as e:
        if e.code != 404:
            print(f"  [warn] {pdb_id}.pdb HTTP {e.code}")
    except URLError as e:
        print(f"  [warn] {pdb_id}.pdb network error: {e}")

    # Fallback: mmCIF
    try:
        urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.cif", cif_path)
        return cif_path, "mmcif"
    except (HTTPError, URLError) as e:
        print(f"  [warn] {pdb_id}: failed to download both .pdb and .cif ({e})")
        return None


def run_mkdssp(structure_path: Path, dssp_executable: str) -> str | None:
    """
    Call mkdssp on the structure file and return its stdout (mmCIF format).
    Returns None on failure.
    """
    # Use a temporary output path so we don't clutter the directory
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        out_path = tmp.name

    try:
        result = subprocess.run(
            [dssp_executable, "--output-format", "mmcif",
             str(structure_path), out_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  [warn] mkdssp returned {result.returncode}: "
                  f"{result.stderr.strip()[:200]}")
            return None
        with open(out_path) as f:
            return f.read()
    except subprocess.TimeoutExpired:
        print(f"  [warn] mkdssp timed out on {structure_path.name}")
        return None
    except FileNotFoundError:
        print(f"  [warn] mkdssp executable not found at: {dssp_executable}")
        return None
    except Exception as e:
        print(f"  [warn] mkdssp call failed on {structure_path.name}: {e}")
        return None
    finally:
        try:
            Path(out_path).unlink()
        except Exception:
            pass


def parse_dssp_mmcif(mmcif_text: str) -> list[tuple[str, str, str]]:
    """
    Parse mmCIF output from mkdssp 4.x and return a list of
    (chain_id, aa3, ss8) tuples in residue order.

    The relevant loop is `_dssp_struct_summary`, which has these key columns:
        label_asn_id, label_seq_id, label_comp_id, secondary_structure
    But the column ordering varies by DSSP version, so we parse the loop
    header to find the right indices.
    """
    lines = mmcif_text.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()
        # Look for the start of the dssp_struct_summary loop
        if line == "loop_":
            # Collect column headers immediately following
            headers = []
            j = i + 1
            while j < n and lines[j].strip().startswith("_"):
                headers.append(lines[j].strip())
                j += 1
            # Is this the summary loop we want?
            if any("_dssp_struct_summary" in h for h in headers):
                # Find the indices we need
                col_chain = col_comp = col_ss = None
                for k, h in enumerate(headers):
                    if h.endswith(".label_asym_id"):
                        col_chain = k
                    elif h.endswith(".label_comp_id"):
                        col_comp = k
                    elif h.endswith(".secondary_structure"):
                        col_ss = k
                if col_chain is None or col_comp is None or col_ss is None:
                    return []

                # Read data rows until next loop_/category/end
                rows = []
                k = j
                while k < n:
                    s = lines[k].strip()
                    if not s or s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
                        break
                    parts = s.split()
                    if len(parts) > max(col_chain, col_comp, col_ss):
                        chain = parts[col_chain]
                        comp = parts[col_comp]
                        ss = parts[col_ss]
                        # mmCIF uses '.' or '?' for missing/none
                        if ss in (".", "?"):
                            ss = "-"
                        rows.append((chain, comp, ss))
                    k += 1
                return rows
            i = j
        else:
            i += 1
    return []


def extract_chain(dssp_rows, target_chain: str):
    """Pull (seq, ss3) for a specific chain from parsed DSSP rows."""
    rows = []
    for chain_id, comp, ss8 in dssp_rows:
        if chain_id != target_chain:
            continue
        aa1 = AA3_TO_AA1.get(comp.upper())
        if aa1 is None:
            continue  # non-standard residue
        ss3 = DSSP_8_TO_3.get(ss8, "C")
        rows.append((aa1, ss3))
    if len(rows) < 30:
        return None
    return "".join(a for a, _ in rows), "".join(s for _, s in rows)


def extract_longest_chain(dssp_rows):
    """Fallback when no chain ID was specified."""
    by_chain = {}
    for chain_id, comp, ss8 in dssp_rows:
        aa1 = AA3_TO_AA1.get(comp.upper())
        if aa1 is None:
            continue
        by_chain.setdefault(chain_id, []).append(
            (aa1, DSSP_8_TO_3.get(ss8, "C"))
        )
    if not by_chain:
        return None
    best = max(by_chain.values(), key=len)
    if len(best) < 30:
        return None
    return "".join(a for a, _ in best), "".join(s for _, s in best)


def read_pdb_list(path: Path) -> list[tuple[str, str | None]]:
    pairs = []
    with open(path) as f:
        for line in f:
            # Normalize whitespace and possible escaped tabs
            line = line.strip().replace("\\t", "\t")
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 1:
                pairs.append((parts[0].lower(), None))
            else:
                pdb_id = parts[0].lower().strip()
                chain = "".join(c for c in parts[1].strip() if c.isalnum())
                pairs.append((pdb_id, chain if chain else None))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_list", required=True,
                    help="Tab-separated list: <pdb_id>\\t<chain> per line. "
                         "If chain is omitted, the longest chain is used.")
    ap.add_argument("--pdb_dir", default="data/pdbs")
    ap.add_argument("--out_tsv", default="data/labels.tsv")
    ap.add_argument("--dssp", default="mkdssp",
                    help="Name (or full path) of the mkdssp executable.")
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)

    pairs = read_pdb_list(Path(args.pdb_list))
    print(f"Processing {len(pairs)} (pdb, chain) entries...")

    n_ok = 0
    with open(args.out_tsv, "w") as out:
        out.write("pdb_id\tchain\tsequence\tss3\n")
        for i, (pdb_id, chain) in enumerate(pairs, 1):
            label = pdb_id if chain is None else f"{pdb_id}_{chain}"
            print(f"[{i}/{len(pairs)}] {label}")

            dl = download_structure(pdb_id, pdb_dir)
            if dl is None:
                continue
            structure_path, _file_format = dl

            mmcif_text = run_mkdssp(structure_path, args.dssp)
            if mmcif_text is None:
                continue

            dssp_rows = parse_dssp_mmcif(mmcif_text)
            if not dssp_rows:
                print(f"  [warn] could not parse DSSP output for {label}")
                continue

            if chain is None:
                result = extract_longest_chain(dssp_rows)
            else:
                result = extract_chain(dssp_rows, chain)
            if result is None:
                print(f"  [skip] no usable residues for {label}")
                continue

            seq, ss3 = result
            assert len(seq) == len(ss3)
            chain_str = chain if chain is not None else ""
            out.write(f"{pdb_id}\t{chain_str}\t{seq}\t{ss3}\n")
            n_ok += 1

    print(f"\nDone. Wrote {n_ok}/{len(pairs)} entries to {args.out_tsv}")


if __name__ == "__main__":
    main()
