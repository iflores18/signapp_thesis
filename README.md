# signapp_thesis
Isolated Sign Language Recognition (ISLR) for Panamanian Sign Language Recognition (LSP). Comparison accross different spatial adn temporal feature extraction methods and computational cost adressing. 

## Requirements

- Python 3.11+
- GPU recommended for training (CUDA-compatible)

## Setup

### With uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that handles virtual environments and dependency locking automatically.

**1. Install uv:**

Linux / macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. Clone and set up the project:**
```bash
git clone https://github.com/iflores18/signapp_thesis.git
cd signapp_thesis
uv sync
```

This creates a virtual environment and installs all dependencies with the exact versions specified in `uv.lock`.

**3. Run scripts:**
```bash
uv run python main.py
```

### With pip

If you prefer not to install uv:

```bash
git clone https://github.com/iflores18/signapp_thesis.git
cd signapp_thesis
python -m venv .venv
```

Activate the environment:

Linux / macOS:
```bash
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
.venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run scripts:
```bash
python main.py
```

## Project Structure

```
signapp_thesis/
├── pyproject.toml        # Project config and dependencies
├── uv.lock               # Locked dependency versions
├── requirements.txt      # For pip-based setup
├── README.md
└── main.py
```
