# Security policy

DeepForge is an early-stage local research utility. The current release candidate is the only development target. There is no guaranteed support window or response SLA.

Report security vulnerabilities privately using GitHub's private vulnerability reporting feature if it is enabled for the repository. If it is unavailable, contact the repository maintainer without posting exploit details, credentials or private datasets in a public issue.

Do not include patient information, tokens, passwords, raw proprietary datasets or identifying paths in reports. DeepForge does not download datasets, execute arbitrary user code, or load serialized model checkpoints. Use current supported versions of Python, Pillow and PyTorch. Treat externally supplied datasets and exported reports as untrusted inputs.

Security reports should include the affected version, a minimal synthetic reproduction, expected and actual behavior, and potential impact. Do not test against data or systems without permission.
