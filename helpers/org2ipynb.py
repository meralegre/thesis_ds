#!/usr/bin/env python3
"""Convert an Org-mode file to a Jupyter Notebook (.ipynb),
correctly mapping #+begin_src blocks to code cells."""

import json
import re
import sys
from pathlib import Path


def parse_org_to_cells(org_text, primary_lang="python"):
    """Parse org content into a list of notebook cells."""
    cells = []
    lines = org_text.splitlines(keepends=True)
    i = 0
    md_buffer = []

    def flush_markdown():
        text = "".join(md_buffer).strip()
        if text:
            cells.append(new_markdown_cell(text))
        md_buffer.clear()

    while i < len(lines):
        line = lines[i]
        if re.match(r"^#\+(TITLE|AUTHOR|DATE|PROPERTY|OPTIONS|STARTUP)\b", line, re.IGNORECASE):
            i += 1
            continue

        # detect #+begin_src ... block
        src_match = re.match(
            r"^#\+begin_src\s+(\S+)(.*)$", line, re.IGNORECASE
        )
        if src_match:
            flush_markdown()
            lang = src_match.group(1).strip()
            code_lines = []
            i += 1
            while i < len(lines) and not re.match(r"^#\+end_src\b", lines[i], re.IGNORECASE):
                code_lines.append(lines[i])
                i += 1
            i += 1

            # skip blank lines, then any #+RESULTS block that follows
            peek = i
            while peek < len(lines) and lines[peek].strip() == "":
                peek += 1
            if peek < len(lines) and re.match(r"^#\+RESULTS:", lines[peek], re.IGNORECASE):
                i = peek + 1
                # skip blank lines between #+RESULTS: and the actual content
                while i < len(lines) and lines[i].strip() == "":
                    i += 1
                # results in a #+begin_example ... #+end_example block
                if i < len(lines) and re.match(r"^#\+begin_example", lines[i], re.IGNORECASE):
                    i += 1
                    while i < len(lines) and not re.match(r"^#\+end_example", lines[i], re.IGNORECASE):
                        i += 1
                    if i < len(lines):
                        # skip #+end_example
                        i += 1
                # results as fixed-width lines (': ' prefix)
                else:
                    while i < len(lines) and (lines[i].startswith(": ") or lines[i].strip() == ""):
                        if lines[i].strip() == "":
                            if i + 1 < len(lines) and lines[i + 1].startswith(": "):
                                i += 1
                                continue
                            else:
                                i += 1
                                break
                        i += 1

            cells.append(new_code_cell("".join(code_lines), lang, primary_lang))
            continue

        md_buffer.append(line)
        i += 1

    flush_markdown()
    return cells


def org_heading_to_md(text: str) -> str:
    """Convert org headings (* heading) to markdown (# heading)."""
    def replace_heading(m):
        stars = m.group(1)
        title = m.group(2)
        return "#" * len(stars) + " " + title

    return re.sub(r"^(\*+)\s+(.+)$", replace_heading, text, flags=re.MULTILINE)


def new_markdown_cell(source: str) -> dict:
    source = org_heading_to_md(source)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


# Languages that can be run with Jupyter cell magic (%%magic) in a Python kernel
MAGIC_MAP = {
    "bash": "%%bash",
    "sh": "%%bash",
    "shell": "%%bash",
    "ruby": "%%ruby",
    "perl": "%%perl",
    "javascript": "%%javascript",
    "js": "%%javascript",
    "html": "%%html",
    "latex": "%%latex",
    "sql": "%%sql",
}


def new_code_cell(source: str, language: str = "python", primary_lang: str = "python") -> dict:
    code = source.rstrip("\n")
    # if the block language differs from the notebook kernel, prepend cell magic
    normalised = language.lower().replace("jupyter-", "").replace("ipython", "python")
    if normalised != primary_lang and normalised in MAGIC_MAP:
        code = MAGIC_MAP[normalised] + "\n" + code

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def detect_kernel(org_text: str) -> str:
    """Try to detect the primary language from src blocks.
    Prefers Python over shell languages since shell can run via %%bash magic."""
    langs = re.findall(r"#\+begin_src\s+(\S+)", org_text, re.IGNORECASE)
    cleaned = [lang.replace("jupyter-", "").replace("ipython", "python") for lang in langs]
    if not cleaned:
        return "python"
    if "python" in cleaned:
        return "python"
    return max(set(cleaned), key=cleaned.count)


def build_notebook(cells, language="python"):
    kernel_map = {
        "python": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
                "file_extension": ".py",
                "mimetype": "text/x-python",
            },
        },
        "r": {
            "kernelspec": {
                "display_name": "R",
                "language": "R",
                "name": "ir",
            },
            "language_info": {
                "name": "R",
                "file_extension": ".r",
                "mimetype": "text/x-r-source",
            },
        },
    }

    metadata = kernel_map.get(language, kernel_map["python"])

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": metadata,
        "cells": cells,
    }


def convert(org_path, ipynb_path=None):
    org_file = Path(org_path)
    if ipynb_path is None:
        ipynb_path = org_file.with_suffix(".ipynb")
    else:
        ipynb_path = Path(ipynb_path)

    org_text = org_file.read_text(encoding="utf-8")
    language = detect_kernel(org_text)
    cells = parse_org_to_cells(org_text, primary_lang=language)
    notebook = build_notebook(cells, language)

    ipynb_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Converted {org_file.name} → {ipynb_path.name}  ({len(cells)} cells)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python org2ipynb.py input.org [output.ipynb]")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)