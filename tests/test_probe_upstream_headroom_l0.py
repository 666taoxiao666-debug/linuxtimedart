import unittest
import numpy as np,pandas as pd
from scripts import probe_upstream_headroom_l0 as h

class HeadroomTests(unittest.TestCase):
    def test_future_observations_cannot_change_features(self):
        times=pd.date_range('2022-01-01',periods=48,freq='h');stations=[{'src_id':1,'distance_km':30,'bearing_deg':0},{'src_id':2,'distance_km':60,'bearing_deg':90}]
        obs={s['src_id']:pd.DataFrame({'wind_speed':np.arange(48)+1.,'wind_direction':np.zeros(48),'msl_pressure':np.arange(48)+1000.},index=times) for s in stations}
        issue=times[24:26];before=h.station_features(issue,obs,stations)
        for v in obs.values():v.loc[times[25]:]=99999
        after=h.station_features(issue,obs,stations)
        np.testing.assert_array_equal(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[1].shape[1],48)
    def test_windows_do_not_cross_year(self):
        series,_,_=h.e.p2.synthetic_data();w=h.year_windows(series,2021)
        if len(w):
            self.assertTrue((w.dates[w.window_starts]>=np.datetime64('2021-01-01')).all())
            self.assertTrue((w.target_times()[:,-1]<np.datetime64('2022-01-01')).all())
        # Boundary-specific fake series, independent of synthetic_data calendar.
        from types import SimpleNamespace
        dates=pd.date_range('2021-12-30',periods=100,freq='h').to_numpy()
        fake=SimpleNamespace(starts=np.arange(73),dates=dates,values=np.zeros((100,1)),turbines=np.ones(100),input_len=24,horizon=4,channels=['power'],scaler_mean=np.zeros(1),scaler_scale=np.ones(1))
        a=h.year_windows(fake,2021);b=h.year_windows(fake,2022)
        self.assertLess(a.target_times().max(),b.dates[b.window_starts].min())
    def test_frozen_decision_boundaries(self):
        t={'go_mae':1.5,'go_rmse':1.,'stop_mae':.5}
        ds=[{'farm':f,'H_MAE_pct':1.5,'H_RMSE_pct':1.} for f in ['Kelmarsh','Kelmarsh','Penmanshiel','Penmanshiel']]
        self.assertEqual(h.verdict(ds,t)['decision'],'GO')
        ds[0]['H_MAE_pct']=-.6;self.assertEqual(h.verdict(ds,t)['decision'],'STOP')
        ds[0]['H_MAE_pct']=.5;self.assertEqual(h.verdict(ds,t)['decision'],'MARGINAL')
if __name__=='__main__':unittest.main()
