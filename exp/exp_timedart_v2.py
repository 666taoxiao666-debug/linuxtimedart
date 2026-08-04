"""Experiment runner for TimeDART-v2.

TimeDART-v2 now uses the same leakage-safe training/evaluation loop as
TimeDART.  The model difference lives entirely in ``models/TimeDART_v2.py``;
keeping a second copy of the loop previously caused training/validation to pass
the real future target into the network while testing did not.
"""

from exp.exp_timedart import Exp_TimeDART


class Exp_TimeDART_v2(Exp_TimeDART):
    def _build_model(self):
        if self.args.downstream_task != "forecast":
            raise ValueError("TimeDART_v2 currently supports forecasting only")
        return super()._build_model()
