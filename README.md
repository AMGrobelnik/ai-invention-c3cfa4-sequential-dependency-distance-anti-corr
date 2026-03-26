# AI Invention Research Repository

This repository contains artifacts from an AI-generated research project.

## Research Paper

[![Download PDF](https://img.shields.io/badge/Download-PDF-red)](https://github.com/AMGrobelnik/ai-invention-c3cfa4-sequential-dependency-distance-anti-corr/blob/main/paper.pdf) [![LaTeX Source](https://img.shields.io/badge/LaTeX-Source-orange)](https://github.com/AMGrobelnik/ai-invention-c3cfa4-sequential-dependency-distance-anti-corr/tree/main/paper_latex)

## Quick Start - Interactive Demos

Click the badges below to open notebooks directly in Google Colab:

### Jupyter Notebooks

| Folder | Description | Open in Colab |
|--------|-------------|---------------|
| `dataset_iter1_ud_grammar_prof` | UD Grammar Prof | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-c3cfa4-sequential-dependency-distance-anti-corr/blob/main/dataset_iter1_ud_grammar_prof/demo/data_code_demo.ipynb) |
| `dataset_iter1_ud_dep_extract` | UD Dep Extract | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-c3cfa4-sequential-dependency-distance-anti-corr/blob/main/dataset_iter1_ud_dep_extract/demo/data_code_demo.ipynb) |
| `dataset_iter1_ud_typology_tab` | UD Typology Table | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-c3cfa4-sequential-dependency-distance-anti-corr/blob/main/dataset_iter1_ud_typology_tab/demo/data_code_demo.ipynb) |
| `experiment_iter3_rpl_diagnostics` | RPL Diagnostics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-c3cfa4-sequential-dependency-distance-anti-corr/blob/main/experiment_iter3_rpl_diagnostics/demo/method_code_demo.ipynb) |
| `experiment_iter3_mc_geometry` | MC & Geometry | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-c3cfa4-sequential-dependency-distance-anti-corr/blob/main/experiment_iter3_mc_geometry/demo/method_code_demo.ipynb) |
| `experiment_iter3_robustness_4_va` | Robustness 4-Var | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-c3cfa4-sequential-dependency-distance-anti-corr/blob/main/experiment_iter3_robustness_4_va/demo/method_code_demo.ipynb) |
| `experiment_iter3_dd_autocorr_exp` | DD Autocorr Exp | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMGrobelnik/ai-invention-c3cfa4-sequential-dependency-distance-anti-corr/blob/main/experiment_iter3_dd_autocorr_exp/demo/method_code_demo.ipynb) |

### Research & Documentation

| Folder | Description | View Research |
|--------|-------------|---------------|
| `research_iter1_dd_baselines` | DD Baselines | [![View Research](https://img.shields.io/badge/View-Research-green)](https://github.com/AMGrobelnik/ai-invention-c3cfa4-sequential-dependency-distance-anti-corr/blob/main/research_iter1_dd_baselines/demo/research_demo.md) |
| `research_iter1_autocorr_meta` | Autocorr & Meta | [![View Research](https://img.shields.io/badge/View-Research-green)](https://github.com/AMGrobelnik/ai-invention-c3cfa4-sequential-dependency-distance-anti-corr/blob/main/research_iter1_autocorr_meta/demo/research_demo.md) |
| `research_iter3_uid_vs_dd_novel` | UID vs DD Novelty | [![View Research](https://img.shields.io/badge/View-Research-green)](https://github.com/AMGrobelnik/ai-invention-c3cfa4-sequential-dependency-distance-anti-corr/blob/main/research_iter3_uid_vs_dd_novel/demo/research_demo.md) |

## Repository Structure

Each artifact has its own folder with source code and demos:

```
.
├── <artifact_id>/
│   ├── src/                     # Full workspace from execution
│   │   ├── method.py            # Main implementation
│   │   ├── method_out.json      # Full output data
│   │   ├── mini_method_out.json # Mini version (3 examples)
│   │   └── ...                  # All execution artifacts
│   └── demo/                    # Self-contained demos
│       └── method_code_demo.ipynb # Colab-ready notebook (code + data inlined)
├── <another_artifact>/
│   ├── src/
│   └── demo/
├── paper.pdf                    # Research paper
├── paper_latex/                 # LaTeX source files
└── README.md
```

## Running Notebooks

### Option 1: Google Colab (Recommended)

Click the "Open in Colab" badges above to run notebooks directly in your browser.
No installation required!

### Option 2: Local Jupyter

```bash
# Clone the repo
git clone https://github.com/AMGrobelnik/ai-invention-c3cfa4-sequential-dependency-distance-anti-corr
cd ai-invention-c3cfa4-sequential-dependency-distance-anti-corr

# Install dependencies
pip install jupyter

# Run any artifact's demo notebook
jupyter notebook <artifact_folder>/demo/
```

## Source Code

The original source files are in each artifact's `src/` folder.
These files may have external dependencies - use the demo notebooks for a self-contained experience.

---
*Generated by AI Inventor Pipeline - Automated Research Generation*
