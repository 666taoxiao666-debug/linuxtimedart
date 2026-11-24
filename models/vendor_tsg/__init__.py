"""Vendored TSG missing-data adaptive regression (Stratigakos & Andrianesis, IEEE TSG), MIT License,
Copyright (c) 2023 akylasstrat.  Source: github.com/akylasstrat/wind-forecast-missing-data at commit
409f20f0d1117758b9f2e9ff48c4373fe97e945c.

- clean_finite_adaptability_functions.py: byte-for-byte copy (git blob 8b62b85b...).
- clean_torch_custom_layers.py: copy with exactly two lines replaced by comments, the unused top-level imports
  `import cvxpy as cp` and `from cvxpylayers.torch import CvxpyLayer` (neither name is used anywhere in the file),
  so no new package is needed.  Upstream blob 84086a27...; the algorithm code is unchanged.

This __init__ is ours and imports nothing.  The author files use the absolute import
`from clean_torch_custom_layers import *`, so scripts/run_wjb.load_tsg puts this folder on sys.path and imports the
two modules by their own names (the repository's models/__init__ is not imported).  See LICENSE in this folder."""
