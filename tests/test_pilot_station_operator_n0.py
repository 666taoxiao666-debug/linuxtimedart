import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts import pilot_station_operator_n0 as n


class N0Tests(unittest.TestCase):
    def setUp(self):
        self.q = dict(src_id=1, lat=52., lon=0., region='Kelmarsh')
        self.stations = [self.q, dict(src_id=2, lat=52.05, lon=0.),
                         dict(src_id=3, lat=52.5, lon=0.), dict(src_id=4, lat=55., lon=0.)]
        self.times = pd.date_range('2019-03-01', periods=12, freq='h')
        self.obs = {s['src_id']: pd.DataFrame({'wind_speed': 4., 'wind_direction': 0.,
                   'msl_pressure': 1000.}, index=self.times) for s in self.stations}

    def test_candidates_exclude_target_near_far_and_holdouts(self):
        self.assertEqual([s['src_id'] for s in n.candidates(self.q,self.stations,{1,2,3,4})],[3])
        self.assertEqual(n.candidates(self.q,self.stations,{1,2,4}),[])

    def test_future_and_target_do_not_affect_features(self):
        selected=n.candidates(self.q,self.stations,{1,2,3,4});t=self.times[5:6]
        z=n.tokens(t,self.obs,selected,4)[0]
        h=n.screen.upstream_features(t,self.obs,selected,1)[0]
        self.obs[1].iloc[:]=9999
        self.obs[3].loc[t[0]:]=9999
        np.testing.assert_equal(z,n.tokens(t,self.obs,selected,4)[0])
        np.testing.assert_equal(h,n.screen.upstream_features(t,self.obs,selected,1)[0])

    def test_empty_and_calm(self):
        z,m,u=n.tokens(self.times[5:6],self.obs,[],4)
        self.assertTrue(u.all());self.assertFalse(m.any())
        model=n.Operator().eval()
        with torch.no_grad():
            y=model(torch.from_numpy(z),torch.from_numpy(m))
        self.assertTrue(torch.isfinite(y).all())
        self.obs[3]['wind_speed']=0.;self.obs[3]['wind_direction']=np.nan
        selected=n.candidates(self.q,self.stations,{3})
        z,m,u=n.tokens(self.times[5:6],self.obs,selected,4)
        self.assertTrue(u.all());self.assertTrue(m[:,0].all());self.assertTrue(np.isfinite(z).all())
        self.assertTrue((z[:,0,20]==1).all())
        self.assertTrue((np.abs(z[:,:,18])<=12).all())

    def test_operator_permutation_and_mask(self):
        torch.manual_seed(2024);model=n.Operator().eval()
        z=torch.randn(3,5,n.TOKEN_DIM);m=torch.tensor([[1,1,0,0,1]]*3,dtype=torch.bool)
        with torch.no_grad():
            y=model(z,m);p=[4,2,0,3,1]
            torch.testing.assert_close(y,model(z[:,p],m[:,p]))
            z[~m]=12345
            torch.testing.assert_close(y,model(z,m))
        self.assertLessEqual(sum(p.numel() for p in model.parameters()),20000)

    def test_split(self):
        stations=[dict(src_id=i,region='Kelmarsh') for i in range(36)]
        stations += [dict(src_id=100+i,region='Penmanshiel') for i in range(11)]
        s=n.split_stations(stations)
        self.assertEqual(s['kel_holdout'],list(range(0,36,5)))
        self.assertEqual(len(s['train']),28)
        self.assertFalse(set(s['train']) & set(s['kel_holdout']+s['pen_holdout']))

    def test_forbidden_years(self):
        for year in (2015,2022,2023,2024):
            with self.assertRaises(ValueError):n.permitted_file(f'missing_qcv-1_{year}.csv',year)
        with self.assertRaises(ValueError):n.permitted_file('x_qcv-1_2022.csv',2021)
        n.permitted_file('x_qcv-1_2021.csv',2021)

    def test_r1_counterexample_and_zero(self):
        self.assertIsNone(n.retention(-.1,-.2));self.assertIsNone(n.retention(0,.2))
        self.assertIsNone(n.retention(.019,.1));self.assertAlmostEqual(n.retention(.1,.08),.8)
        r=n.decision([.1]*8,[.1]*8,[-.2]*8,[0]*8,[0]*8)
        self.assertFalse(r['continue'])
        r=n.decision([0]*8,[0]*8,[0]*8,[0]*8,[0]*8)
        self.assertFalse(r['continue'])

    def test_gate_and_simplification(self):
        r=n.decision([.2]*6+[-.1]*2,[.1]*6+[-.1]*2,[.2]*8,[.02]*8,[.02]*8)
        self.assertTrue(r['continue']);self.assertTrue(r['choose_simple'])
        r=n.decision([.2]*5+[-.1]*3,[.1]*8,[.2]*8,[.02]*8,[.02]*8)
        self.assertFalse(r['continue'])

    def test_scaler_and_ridge_never_fit_test(self):
        rng=np.random.default_rng(12);x=rng.normal(size=(160,3));y=np.repeat((x[:,0]*2+3)[:,None],4,axis=1)
        years=np.array([2019]*80+[2020]*40+[2021]*40)
        a=n.ridge_fit(x,y,years);xx=x.copy();yy=y.copy();xx[-40:]=999;yy[-40:]=-999
        b=n.ridge_fit(xx,yy,years)
        np.testing.assert_equal(a['coef'],b['coef']);np.testing.assert_equal(a['lambda'],b['lambda'])
        np.testing.assert_equal(a['stats'],b['stats'])
        stats=n.scaler(np.array([[np.nan,2],[np.nan,2]]))
        self.assertTrue(np.isfinite(n.transform(np.array([[np.nan,2]]),stats)).all())

    def test_transfer_uses_source_intercept_target_stats(self):
        model={'coef':np.array([[3.]*4,[2.]*4])}
        pred=n.ridge_predict(model,np.array([[14.]]),(np.array([10.]),np.array([2.])))
        np.testing.assert_equal(pred,[[7.]*4])

    def test_synthetic_end_to_end(self):
        torch.set_num_threads(2)
        rng=np.random.default_rng(2024);arrays={}
        for sid in (1,2,3):
            arrays[sid]={'L':rng.normal(size=(48,24)).astype('float32'),
                         'H':rng.normal(size=(48,11)).astype('float32'),
                         'z':rng.normal(size=(48,2,n.TOKEN_DIM)).astype('float32'),
                         'mask':np.ones((48,2),bool),'y':rng.uniform(0,8,(48,4)).astype('float32'),
                         'year':np.repeat([2019,2020,2021],16),
                         'time':pd.date_range('2019-01-01',periods=48,freq='h').to_numpy()}
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            op=n.train_model(arrays,[1],'O','cpu',20,folder/'O')
            ph=n.train_model(arrays,[1],'PH','cpu',20,folder/'PH')
            result=n.evaluate(arrays,[2,3],[2,3],op,ph,'cpu',folder/'arrays')
            n.save(folder/'result.json',result)
            self.assertEqual(len(result['pairs']),2)
            self.assertEqual(len(list((folder/'arrays').glob('*.npz'))),8)
            for pair in result['pairs']:
                for arm in ('H','PH','O'):
                    self.assertAlmostEqual(pair['gain'][arm],pair['mae']['L']-pair['mae'][arm])


if __name__=='__main__':unittest.main()
