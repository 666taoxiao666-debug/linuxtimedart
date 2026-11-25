"""Vendored Cabin model (Wu et al., Energy 2025), MIT License, Copyright (c) 2024 Senzhen Wu, Yu Chen, Xinhao He,
Zhijin Wang, Xiufeng Liu, Yonggang Fu.  Source: github.com/xiufengliu/cabin-wind-forecasting at commit
392f103426faf600c5469875b020a4693cba6a7d, files cabin/models/{base,components,cabin}.py copied byte-for-byte
(git blob SHAs recorded in configs/wjb_protocol.json and checked by scripts/run_wjb.vendor_fingerprint).
This __init__ is ours.  run_wjb loads this folder through importlib under the package name 'vendor_cabin', so the
repository's models/__init__ is not imported.  See LICENSE in this folder."""
from .cabin import Cabin  # noqa: F401
