# Historical regression fixtures

These are immutable test inputs, not supported public entrypoints.
PDE_to_ODE.py and CHEYBYSHEV_approximation.py are checked against hashes.json.
The old viewer verifies pure-cosine archive compatibility. Tests pass temporary
cache/output directories explicitly. No runtime module imports these fixtures;
they are excluded from the installed wheel.
