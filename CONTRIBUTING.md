# Contributing

1. Create a focused branch and describe the problem being solved.
2. Install the development dependencies with `python -m pip install -e '.[dev]'`.
3. Add deterministic tests that do not require network access or private datasets.
4. Run `python -m pytest -q` and update documentation for behavior changes.
5. Open a pull request describing what changed, validation performed, and known limitations.

Do not upload credentials, personal data, patient identifiers, or datasets without permission. Disclose AI assistance when it materially contributes to a change, and review and understand all submitted code.
