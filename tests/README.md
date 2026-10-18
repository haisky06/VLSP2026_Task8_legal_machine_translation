# CPU contract tests

From the repository root on Linux or Bash:

```bash
PYTHONPATH=vinai_modal python -m unittest discover -s tests -v
```

On PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path .\vinai_modal).Path
python -m unittest discover -s tests -v
```

Tests use synthetic text, not competition datasets or GPU training. Passing them does not validate translation accuracy or end-to-end deployment. Archived tests requiring old report/data paths are not included as runnable tests.
