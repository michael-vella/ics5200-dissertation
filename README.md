# ICS5200-dissertation

Code repository for the Master of Science in Artificial Intelligence (AI) dissertation.

- Study-unit code: ICS5200.
- Deadline: End of June 2027.

## Environment Replication

Requires that Python is pre-installed on the host machine. Python version: 3.12.3.

(TODO) Later on refactor solution to work with docker.

1. Run `python -m venv .venv` to create the Python virtual environment. `python` here refers to the alias of the Python executable path and depends on the alias used on the host machine (full Python path can also be used). Running this command will create a Python virtual environment depending on the base Python version being used to create the environment.
2. Activate virtual environment by running `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux).
3. Upgrade `pip` (Python's package manager) by running `pip install --upgrade pip`.
4. Run `pip install -r requirements.txt` to download any packages required for this project.

## Datasets

Datasets used in this dissertation.

### Tox21

Dataset (CSV) obtained from [DeepChem AWS bucket](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz). Last accessed: 08 March 2026.

### MUV

Dataset (CSV) obtained from [DeepChem AWS bucket](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz). Last accessed: 08 March 2026.

### DUD-E

TODO.

## LIT-PCBA

TODO.