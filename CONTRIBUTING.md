# Contributing

Thank you for considering a contribution.

## Development setup

```powershell
py -m venv "$env:LOCALAPPDATA\faceauth\venv"
$python = "$env:LOCALAPPDATA\faceauth\venv\Scripts\python.exe"
& $python -m pip install -r requirements.txt
& $python -m pip install -e . --no-deps
& $python -m pytest -q
```

The project is Windows-first because biometric storage uses DPAPI and OpenCV
does not provide native Windows ARM64 wheels.

## Pull requests

- Keep changes focused and include tests for behavior changes.
- Preserve deterministic seeds and fixed evaluation splits.
- Run the full test suite and the narrowest relevant evaluation.
- Report negative results and uncertainty; do not select only favorable runs.
- Update documentation when command behavior, storage format, or security
  boundaries change.

## Privacy

Never commit or attach:

- face images or webcam captures;
- enrolled models or crops;
- background-cohort payloads generated from local data;
- credentials, API tokens, or private paths containing personal data.

Use synthetic fixtures or the documented public benchmark preparation workflow.
