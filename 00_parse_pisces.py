"""
Step 0: Parse a PISCES culled-PDB list and produce a tab-separated
        list of (pdb_id, chain) pairs to download.

Usage:
    # 1. Download a list from https://dunbrack.fccc.edu/lab/pisces_culledpdb
    #    e.g.
    #    cullpdb_pc25.0_res0.0-2.5_noBrks_len40-10000_R0.3_Xray_d2026_04_24_chains9944.fasta
    # 2. Run:
    #        python 00_parse_pisces.py --pisces_file <downloaded file> \
    #                                  --out pdb_list.txt --n 150 --shuffle

PISCES distributes lists in two formats; this script auto-detects which one:

  (a) FASTA (modern, .fasta extension):
        >1A01_B 146 1.80 0.169 0.223
        MHLTPEEKSAVTALWGK...
        >1A02_F 85 1.90 ...
        ...

  (b) Tabular (older):
        IDs       length  Exptl.  resolution  R-factor  FreeRvalue
        1A01B     146     XRAY    1.80        0.169     0.223

In the FASTA format the header is "PDBID_CHAIN" (e.g. 1A01_B).
In the tabular format the first column is the PDB ID concatenated with the
chain ID (e.g. 1A01B).
"""

import argparse
import random
import re
from pathlib import Path


# Match a PISCES FASTA header. The chain ID can be a single letter (most
# common) or longer for multi-character chain IDs (rare).
_FASTA_HEADER_RE = re.compile(r"^>(\w{4})[_ ]?(\w+)")


def parse_fasta(pisces_path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    with open(pisces_path) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            m = _FASTA_HEADER_RE.match(line)
            if not m:
                continue
            pdb_id = m.group(1).lower()
            chain = m.group(2)
            if pdb_id in seen:
                continue
            seen.add(pdb_id)
            pairs.append((pdb_id, chain))
    return pairs


def parse_tabular(pisces_path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    with open(pisces_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            token = parts[0]
            # Skip header lines like "IDs ..."
            if len(token) < 5 or not token[0].isdigit():
                continue
            pdb_id = token[:4].lower()
            chain = token[4:]
            if pdb_id in seen:
                continue
            seen.add(pdb_id)
            pairs.append((pdb_id, chain))
    return pairs


def parse_pisces(pisces_path: Path) -> list[tuple[str, str]]:
    """Auto-detect FASTA vs tabular and parse accordingly."""
    with open(pisces_path) as f:
        first_real = ""
        for line in f:
            s = line.strip()
            if s:
                first_real = s
                break
    if first_real.startswith(">"):
        print("Detected FASTA-format PISCES list.")
        return parse_fasta(pisces_path)
    print("Detected tabular-format PISCES list.")
    return parse_tabular(pisces_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pisces_file", required=True,
                    help="Path to a PISCES culledpdb list file (.fasta or tabular).")
    ap.add_argument("--out", default="pdb_list.txt",
                    help="Output path. Format: <pdb_id>\\t<chain> per line.")
    ap.add_argument("--n", type=int, default=150,
                    help="Take this many entries (0 = all).")
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle before taking the top N (recommended, since "
                         "PISCES files are sorted alphabetically by PDB ID and "
                         "the first N might be biased toward older structures).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = parse_pisces(Path(args.pisces_file))
    print(f"Parsed {len(pairs)} unique PDB chains from {Path(args.pisces_file).name}")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(pairs)

    if args.n > 0:
        pairs = pairs[: args.n]

    with open(args.out, "w") as out:
        for pdb_id, chain in pairs:
            out.write(f"{pdb_id}\t{chain}\n")

    print(f"Wrote {len(pairs)} entries to {args.out}")


if __name__ == "__main__":
    main()
