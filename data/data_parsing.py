"""
Parse mmCIF (.cif) files to extract per-chain amino acid sequences and
secondary structure labels (H=helix, E=strand, C=coil).

Secondary structure is assigned from:
  _struct_conf        — helices (all HELX_* conf_type_ids -> H)
  _struct_sheet_range — beta strands                      -> E
  everything else     —                                   -> C

Only residues observed in the structure (modeled coordinates) are included,
matching the behaviour of DSSP-based pipelines.
"""

from collections import defaultdict
from pathlib import Path

AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _tokenize(line: str) -> list[str]:
    """Split a CIF data line into tokens, respecting single/double quotes."""
    tokens: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c.isspace():
            i += 1
        elif c in ("'", '"'):
            j = i + 1
            while j < n and line[j] != c:
                j += 1
            tokens.append(line[i + 1 : j])
            i = j + 1
        else:
            j = i
            while j < n and not line[j].isspace():
                j += 1
            tokens.append(line[i:j])
            i = j
    return tokens


def _parse_loop(lines: list[str], start: int, category: str):
    """
    Parse a loop_ block whose columns belong to *category* (e.g. '_struct_conf').

    Returns (col_names, rows, next_line_index).
    col_names: list of field names after the dot  (e.g. ['conf_type_id', 'id', ...])
    rows:      list of token lists, one per data row
    Returns (None, [], start+1) if the loop doesn't match category.
    """
    j = start + 1
    n = len(lines)
    headers: list[str] = []

    while j < n:
        s = lines[j].strip()
        if s.startswith("_"):
            headers.append(s)
            j += 1
        elif not s or s == "#":
            j += 1
        else:
            break

    prefix = category + "."
    if not headers or not headers[0].startswith(prefix):
        return None, [], j

    n_cols = len(headers)
    col_names = [h[len(prefix):] for h in headers]

    rows: list[list[str]] = []
    pending: list[str] = []
    k = j
    while k < n:
        s = lines[k].strip()
        if not s:
            k += 1
            continue
        if s.startswith("#") or s == "loop_" or (s.startswith("_") and "." in s):
            break
        if s.startswith(";"):
            # Multi-line text value — collect until closing semicolon
            text_parts: list[str] = []
            k += 1
            while k < n and not lines[k].startswith(";"):
                text_parts.append(lines[k].rstrip())
                k += 1
            pending.append("\n".join(text_parts))
            k += 1
        else:
            pending.extend(_tokenize(s))
            k += 1

        while len(pending) >= n_cols:
            rows.append(pending[:n_cols])
            pending = pending[n_cols:]

    return col_names, rows, k


def _extract_loops(cif_path: Path, categories: set[str]) -> dict[str, tuple[list[str], list[list[str]]]]:
    """
    Read *cif_path* and return loop data for the requested categories.
    Result: {category: (col_names, rows)}.
    Missing categories are absent from the dict (not an error — some proteins
    have no helices or no beta strands).
    """
    with open(cif_path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    result: dict[str, tuple[list[str], list[list[str]]]] = {}
    i = 0
    n = len(lines)
    remaining = set(categories)

    while i < n and remaining:
        if lines[i].strip() == "loop_":
            # Peek at first header to identify category
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j < n:
                first = lines[j].strip()
                if first.startswith("_") and "." in first:
                    cat = first[: first.rindex(".")]
                    if cat in remaining:
                        col_names, rows, next_i = _parse_loop(lines, i, cat)
                        if col_names is not None:
                            result[cat] = (col_names, rows)
                            remaining.discard(cat)
                            i = next_i
                            continue
            i += 1
        else:
            i += 1

    return result


def parse_cif(cif_path: str | Path, chain: str | None = None) -> dict[str, tuple[str, str]]:
    """
    Parse an mmCIF file and return secondary structure labels.

    Parameters
    ----------
    cif_path : path to the .cif file
    chain    : if given, return only that chain.  Accepts either the author
               chain ID (pdb_strand_id, e.g. 'U') or the internal label
               asym_id (e.g. 'C').  If None, return all protein chains.

    Returns
    -------
    dict mapping auth_chain_id -> (sequence, ss3)
      sequence : one-letter amino acid string (observed residues only)
      ss3      : same-length string of 'H' / 'E' / 'C' labels
    """
    cif_path = Path(cif_path)
    loops = _extract_loops(
        cif_path,
        {"_pdbx_poly_seq_scheme", "_struct_conf", "_struct_sheet_range"},
    )

    # ------------------------------------------------------------------ #
    # 1.  Build per-chain residue lists from _pdbx_poly_seq_scheme        #
    # ------------------------------------------------------------------ #
    poly_cols, poly_rows = loops.get("_pdbx_poly_seq_scheme", ([], []))
    if not poly_cols:
        return {}

    def col(name: str, row: list[str]) -> str:
        idx = poly_cols.index(name) if name in poly_cols else -1
        return row[idx] if 0 <= idx < len(row) else "?"

    # Build auth<->label chain ID mappings (first occurrence wins).
    # CIF files may use different letters for label_asym_id (e.g. 'C') and
    # pdb_strand_id / auth chain (e.g. 'U').  pdb_list.txt uses auth IDs.
    label_to_auth: dict[str, str] = {}
    auth_to_label: dict[str, str] = {}
    for row in poly_rows:
        lab = col("asym_id", row)
        auth = col("pdb_strand_id", row)
        if auth not in ("?", ".") and lab not in label_to_auth:
            label_to_auth[lab] = auth
            auth_to_label.setdefault(auth, lab)

    # Resolve the requested chain to its internal label asym_id.
    if chain is not None:
        if chain in auth_to_label:
            label_filter: str | None = auth_to_label[chain]
        else:
            label_filter = chain  # assume it was already a label asym_id
    else:
        label_filter = None

    # label asym_id -> list of (seq_id, aa1) for observed standard residues
    chain_residues: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for row in poly_rows:
        asym_id = col("asym_id", row)
        if label_filter is not None and asym_id != label_filter:
            continue
        try:
            seq_id = int(col("seq_id", row))
        except ValueError:
            continue
        mon_id = col("mon_id", row).upper()
        pdb_mon_id = col("pdb_mon_id", row)

        # Skip residues absent from the model (in SEQRES but not observed)
        if pdb_mon_id in ("?", "."):
            continue

        aa1 = AA3_TO_AA1.get(mon_id)
        if aa1 is None:
            continue  # non-standard / nucleotide residue

        chain_residues[asym_id].append((seq_id, aa1))

    # Sort by seq_id and build sequence + a seq_id -> position mapping
    chain_seq: dict[str, str] = {}
    chain_idx: dict[str, dict[int, int]] = {}

    for asym_id, residues in chain_residues.items():
        residues.sort(key=lambda r: r[0])
        chain_seq[asym_id] = "".join(aa for _, aa in residues)
        chain_idx[asym_id] = {seq_id: pos for pos, (seq_id, _) in enumerate(residues)}

    # ------------------------------------------------------------------ #
    # 2.  Collect helix ranges from _struct_conf                          #
    # ------------------------------------------------------------------ #
    conf_cols, conf_rows = loops.get("_struct_conf", ([], []))
    helices: list[tuple[str, int, int]] = []

    if conf_cols:
        def ccol(name: str, row: list[str]) -> str:
            idx = conf_cols.index(name) if name in conf_cols else -1
            return row[idx] if 0 <= idx < len(row) else "?"

        for row in conf_rows:
            if not ccol("conf_type_id", row).startswith("HELX"):
                continue
            asym_id = ccol("beg_label_asym_id", row)
            if label_filter is not None and asym_id != label_filter:
                continue
            try:
                beg = int(ccol("beg_label_seq_id", row))
                end = int(ccol("end_label_seq_id", row))
            except ValueError:
                continue
            helices.append((asym_id, beg, end))

    # ------------------------------------------------------------------ #
    # 3.  Collect strand ranges from _struct_sheet_range                  #
    # ------------------------------------------------------------------ #
    sheet_cols, sheet_rows = loops.get("_struct_sheet_range", ([], []))
    strands: list[tuple[str, int, int]] = []

    if sheet_cols:
        def scol(name: str, row: list[str]) -> str:
            idx = sheet_cols.index(name) if name in sheet_cols else -1
            return row[idx] if 0 <= idx < len(row) else "?"

        for row in sheet_rows:
            asym_id = scol("beg_label_asym_id", row)
            if label_filter is not None and asym_id != label_filter:
                continue
            try:
                beg = int(scol("beg_label_seq_id", row))
                end = int(scol("end_label_seq_id", row))
            except ValueError:
                continue
            strands.append((asym_id, beg, end))

    # ------------------------------------------------------------------ #
    # 4.  Build ss3 strings; key final result by auth chain ID            #
    # ------------------------------------------------------------------ #
    result: dict[str, tuple[str, str]] = {}

    for asym_id, seq in chain_seq.items():
        ss3 = ["C"] * len(seq)
        idx_map = chain_idx[asym_id]

        for h_chain, beg, end in helices:
            if h_chain != asym_id:
                continue
            for seq_id in range(beg, end + 1):
                pos = idx_map.get(seq_id)
                if pos is not None:
                    ss3[pos] = "H"

        for s_chain, beg, end in strands:
            if s_chain != asym_id:
                continue
            for seq_id in range(beg, end + 1):
                pos = idx_map.get(seq_id)
                if pos is not None:
                    ss3[pos] = "E"

        out_key = label_to_auth.get(asym_id, asym_id)
        result[out_key] = (seq, "".join(ss3))

    return result


def parse_pdb_list(path: str | Path) -> list[tuple[str, str]]:
    """
    Read a pdb_list.txt file (tab-separated <pdb_id> <chain> per line).

    Returns a list of (pdb_id, chain) tuples with pdb_id lowercased.
    Lines starting with '#' and blank lines are skipped.
    """
    pairs: list[tuple[str, str]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pdb_id = parts[0].lower()
            chain = parts[1] if len(parts) >= 2 else "A"
            pairs.append((pdb_id, chain))
    return pairs


def load_all_chains(
    pdbs_dir: str | Path,
    min_length: int = 30,
) -> dict[str, tuple[str, str]]:
    """
    Parse every .cif file in *pdbs_dir* and return a flat dict mapping
    "<pdb_id>_<chain>" -> (sequence, ss3).

    Chains shorter than *min_length* residues are skipped.
    """
    pdbs_dir = Path(pdbs_dir)
    all_entries: dict[str, tuple[str, str]] = {}

    for cif_file in sorted(pdbs_dir.glob("*.cif")):
        pdb_id = cif_file.stem.lower()
        try:
            chains = parse_cif(cif_file)
        except Exception as exc:
            print(f"[warn] {cif_file.name}: {exc}")
            continue

        for chain_id, (seq, ss3) in chains.items():
            if len(seq) < min_length:
                continue
            key = f"{pdb_id}_{chain_id}"
            all_entries[key] = (seq, ss3)

    return all_entries
