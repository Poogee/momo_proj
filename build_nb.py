"""Build a Jupyter notebook (.ipynb) from a percent-format .py source.

Cell markers:
    # %%               -> code cell
    # %% [markdown]     -> markdown cell (lines prefixed with '# ' are stripped)

Usage: python build_nb.py src_py/foo.py notebooks/foo.ipynb
"""
import json
import sys


def build(src_path: str, out_path: str) -> None:
    with open(src_path, "r") as f:
        lines = f.read().splitlines()

    cells = []
    cur_type = "code"
    cur_buf: list[str] = []

    def flush() -> None:
        if not cur_buf and not cells:
            return
        src = "\n".join(cur_buf).strip("\n")
        if cur_type == "markdown":
            stripped = []
            for ln in src.splitlines():
                if ln.startswith("# "):
                    stripped.append(ln[2:])
                elif ln == "#":
                    stripped.append("")
                else:
                    stripped.append(ln)
            src = "\n".join(stripped)
        if src.strip() == "":
            return
        cells.append(
            {
                "cell_type": cur_type,
                "metadata": {},
                "source": src.splitlines(keepends=False),
                **(
                    {"outputs": [], "execution_count": None}
                    if cur_type == "code"
                    else {}
                ),
            }
        )

    for ln in lines:
        if ln.strip().startswith("# %%"):
            flush()
            cur_buf = []
            cur_type = "markdown" if "[markdown]" in ln else "code"
        else:
            cur_buf.append(ln)
    flush()

    # join source lines back with newlines for nbformat (list of str w/ \n)
    for c in cells:
        s = c["source"]
        c["source"] = [x + "\n" for x in s[:-1]] + [s[-1]] if s else []

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {out_path} with {len(cells)} cells")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
