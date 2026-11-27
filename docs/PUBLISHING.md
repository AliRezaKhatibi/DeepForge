# Publishing DeepForge to GitHub

These steps preserve the existing Git history. Extract the source ZIP first. The archive contains a single `DeepForge` directory with source files, not a `.git` directory.

From PowerShell, choose a directory outside the extracted archive and clone the existing repository:

```powershell
git clone https://github.com/AliRezaKhatibi/DeepForge.git
cd DeepForge
```

Copy the contents of the extracted `DeepForge` directory into this clone, including hidden entries such as `.github` and `.gitignore`. Do not copy `.venv`, credentials, personal datasets or caches. Keep the clone's `.git` directory intact. Review the existing LICENSE before replacing it; preserve the correct owner attribution if it differs.

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
git status --short
git add --all
git diff --cached --check
git diff --cached --stat
git commit -m "Publish DeepForge Studio v0.3.0 release candidate"
git push origin main
```

Inspect the staged files before committing. If the push fails, check the remote and error rather than force-pushing. Review GitHub Actions and resolve any failures before creating a versioned GitHub Release. Use `v0.3.0` as the tag and mark the release as pre-release until platform validation is complete. Publishing to PyPI is separate and requires a package-name availability check, build verification and publisher credentials.
