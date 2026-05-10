"""
Step 1: Download PDB structures and extract per-residue secondary structure
        labels using DSSP.

For each PDB:
  - Download the .pdb file
  - Run DSSP to get the 8-state SS assignment per residue
  - Map 8-state -> 3-state (H, E, C)
  - Save (sequence, ss_string) as a tab-separated row

Requires:
  pip install biopython
  conda install -c salilab dssp     (or: apt-get install dssp)

Reference for 8->3 mapping (standard convention):
  H, G, I -> H   (alpha-helix, 3_10-helix, pi-helix)
  E, B    -> E   (beta-strand, beta-bridge)
  T, S, -, ' ', P -> C   (everything else = coil)
"""

import os
import gzip
import argparse
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP


# Standard 8 -> 3 state mapping
DSSP_8_TO_3 = {
    "H": "H", "G": "H", "I": "H",
    "E": "E", "B": "E",
    "T": "C", "S": "C", "-": "C", " ": "C", "P": "C",
}

# Three-letter -> one-letter amino acid code
AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def download_pdb(pdb_id: str, out_dir: Path) -> Path | None:
    """Download a PDB file from RCSB. Returns the local path or None on failure."""
    pdb_id = pdb_id.lower()
    out_path = out_dir / f"{pdb_id}.pdb"
    if out_path.exists():
        return out_path

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        urlretrieve(url, out_path)
        return out_path
    except (URLError, Exception) as e:
        print(f"  [warn] failed to download {pdb_id}: {e}")
        return None


def extract_seq_and_ss(pdb_path: Path, dssp_executable: str = "mkdssp"):
    """
    Run DSSP on a PDB file. Returns (sequence, ss3_string) for the first
    chain that yields a usable result, or None if DSSP fails.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    except Exception as e:
        print(f"  [warn] PDBParser failed on {pdb_path.name}: {e}")
        return None

    model = structure[0]
    try:
        dssp = DSSP(model, str(pdb_path), dssp=dssp_executable)
    except Exception as e:
        print(f"  [warn] DSSP failed on {pdb_path.name}: {e}")
        return None

    # Group DSSP output by chain
    by_chain: dict[str, list[tuple[str, str]]] = {}
    for key in dssp.keys():
        chain_id, res_id = key
        record = dssp[key]
        # record = (dssp_idx, aa1, ss8, ...)
        aa1 = record[1]
        ss8 = record[2]
        if aa1 == "X" or aa1 == "-":
            continue
        ss3 = DSSP_8_TO_3.get(ss8, "C")
        by_chain.setdefault(chain_id, []).append((aa1, ss3))

    if not by_chain:
        return None

    # Pick the longest chain
    best_chain = max(by_chain.values(), key=len)
    if len(best_chain) < 30:
        return None  # too short to be useful

    seq = "".join(aa for aa, _ in best_chain)
    ss3 = "".join(s for _, s in best_chain)
    return seq, ss3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_list", required=True,
                    help="Text file with one PDB ID per line.")
    ap.add_argument("--pdb_dir", default="data/pdbs",
                    help="Where to cache downloaded .pdb files.")
    ap.add_argument("--out_tsv", default="data/labels.tsv",
                    help="Output TSV: pdb_id<TAB>sequence<TAB>ss3.")
    ap.add_argument("--dssp", default="mkdssp",
                    help="Name of the DSSP executable (mkdssp or dssp).")
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)

    with open(args.pdb_list) as f:
        pdb_ids = [line.strip() for line in f if line.strip()]
    print(f"Processing {len(pdb_ids)} PDB IDs...")

    n_ok = 0
    with open(args.out_tsv, "w") as out:
        out.write("pdb_id\tsequence\tss3\n")
        for i, pdb_id in enumerate(pdb_ids, 1):
            print(f"[{i}/{len(pdb_ids)}] {pdb_id}")
            pdb_path = download_pdb(pdb_id, pdb_dir)
            if pdb_path is None:
                continue
            result = extract_seq_and_ss(pdb_path, args.dssp)
            if result is None:
                continue
            seq, ss3 = result
            assert len(seq) == len(ss3)
            out.write(f"{pdb_id}\t{seq}\t{ss3}\n")
            n_ok += 1

    print(f"\nDone. Wrote {n_ok}/{len(pdb_ids)} entries to {args.out_tsv}")


if __name__ == "__main__":
    main()
