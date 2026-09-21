#!/usr/bin/env python
from __future__ import annotations
import argparse, ast, importlib.util, json, sys
from pathlib import Path

def loadmod(path: Path):
    spec = importlib.util.spec_from_file_location("cebase", path)
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m

def extract_symbol_source(path: Path, symbol: str, kind: str):
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in ast.walk(tree):
        ok = (
            (kind == "class" and isinstance(node, ast.ClassDef) and node.name == symbol)
            or
            (kind == "function" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol)
        )
        if ok:
            end = getattr(node, "end_lineno", node.lineno)
            return "\n".join(lines[node.lineno - 1:end])
    return f"<{kind} {symbol} not found in {path}>"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    sibling = root.parent / "LegalIR"
    base = root.parent / "run_noncal_trainable_ce_boundary_v3_fixed.py"
    if not base.is_file():
        raise FileNotFoundError(base)

    m = loadmod(base)
    sys.path[:0] = [str(root), str(root / "src"), str(sibling), str(sibling / "src")]

    cal_ids, _ = m.get_cal_ids_label_free(root)
    world = m.load_noncal_world(root, sibling, set(cal_ids))
    r = world["render"]

    print("=" * 100)
    print("RENDER PROVENANCE")
    print("base_script:", base)
    print("render_class:", type(r).__module__, type(r).__name__)
    print("render_fingerprint:", getattr(r, "fingerprint", None))

    # Critical provenance first: no introspection before this block.
    for name in ("_matrix", "_qvec"):
        x = getattr(r, name, None)
        print(f"{name}.type:", type(x).__name__)
        print(f"{name}.shape:", getattr(x, "shape", None))
        print(f"{name}.dtype:", getattr(x, "dtype", None))
        print(f"{name}.filename:", getattr(x, "filename", None))
        print(f"{name}.offset:", getattr(x, "offset", None))
        print(f"{name}.mode:", getattr(x, "mode", None))

    print("qrow_n:", len(getattr(r, "_qrow", {})))
    print("questions_n:", len(getattr(r, "questions", {})))
    print("chunk_ids_n:", len(getattr(r, "chunk_ids", [])))
    print("doc_ids_n:", len(getattr(r, "doc_ids", [])))

    public_path = root / "DSC2026-LegalIR-main/v4_run/public_test_dataset/public-official.json"
    if public_path.is_file():
        pub = json.loads(public_path.read_text(encoding="utf-8"))
        pids = set(map(str, pub))
        qrow = set(map(str, getattr(r, "_qrow", {})))
        qs = set(map(str, getattr(r, "questions", {})))
        print("public_path:", public_path)
        print("public_n:", len(pids))
        print("public_in_qrow:", len(pids & qrow))
        print("public_in_questions:", len(pids & qs))
        print("sample_public_in_qrow:", sorted(pids & qrow)[:20])
        print("sample_public_missing_qrow:", sorted(pids - qrow)[:20])
    else:
        print("public_path_missing:", public_path)

    print("\nRENDERDATA SOURCE (AST FROM BASE SCRIPT)")
    print("-" * 100)
    try:
        print(extract_symbol_source(base, "RenderData", "class"))
    except Exception as e:
        print("SOURCE_ERROR", repr(e))

    print("\nload_noncal_world SOURCE (AST FROM BASE SCRIPT)")
    print("-" * 100)
    try:
        print(extract_symbol_source(base, "load_noncal_world", "function"))
    except Exception as e:
        print("SOURCE_ERROR", repr(e))

    print("=" * 100)

if __name__ == "__main__":
    main()
