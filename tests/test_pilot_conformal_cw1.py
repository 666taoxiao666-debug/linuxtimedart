import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from scripts import pilot_conformal_cw1 as c


def pack(n,date):
    issue=np.datetime64(date,'ns')+np.arange(n)*np.timedelta64(1,'h')
    return {'prediction':np.zeros((n,4)),'truth':np.tile(np.arange(1,n+1)[:,None],(1,4)).astype(float),
            'issue':issue,'times':issue[:,None]+np.arange(1,5)*np.timedelta64(1,'h'),
            'turbines':np.ones(n,int),'features':np.zeros((n,20))}


def smoke(out):
    rng=np.random.default_rng(2024)
    s,t,e=pack(600,'2021-01-01'),pack(100,'2021-12-01'),pack(20,'2022-01-01')
    for p in (s,t,e): p['features'][:,:11]=rng.normal(size=(len(p['issue']),11))
    weather=c.weather_fit(s['features'][:,:11],t['features'][:,:11],s['issue'],t['issue'],2022)
    strong=c.hgb_weights(s,t,e,2022)
    radius,caps=c.intervals(s,t,e,(np.ones(600),np.ones(20)),2022,c.BASE+('M-A','W1','W2','W4','B3-G'),
                            weather.predict(e['features'][:,:11]),np.tile([1,2,3],(4,1)),strong)
    report=c.reports(radius,caps,np.abs(e['truth']-e['prediction']))
    ci=c.paired_ci(radius['W1']-radius['B3-A'],e['issue'],out/'bootstrap.npz',100)
    result={'status':'PASSED','smoke':True,'methods':list(report),'ci':ci}
    c.save(out/'result.json',result); return result


class CW1Tests(unittest.TestCase):
    def test_equal_weights_W1_equals_B3(self):
        s,t,e=pack(600,'2021-01-01'),pack(100,'2021-12-01'),pack(50,'2022-01-01')
        e['truth'][:]=1e6
        r,cap=c.intervals(s,t,e,(np.ones(600),np.ones(50)),2022,('W1','B3-A'))
        np.testing.assert_array_equal(r['W1'],r['B3-A']); np.testing.assert_array_equal(cap['W1'],cap['B3-A'])
        self.assertTrue(np.isfinite(r['W1']).all())

    def test_future_suffix_cannot_change_past_group_or_radius(self):
        rng=np.random.default_rng(4); x=rng.normal(size=(100,11)); times=pack(100,'2021-01-01')['issue']
        fitted=c.weather_fit(x,x*2,times,times,2022)
        future=rng.normal(size=(20,11)); changed=future.copy(); changed[5:]+=10000
        g1,g2=fitted.predict(future),fitted.predict(changed)
        np.testing.assert_array_equal(g1[:5],g2[:5])
        s,t,e=pack(100,'2021-01-01'),pack(100,'2021-12-01'),pack(20,'2022-01-01')
        r1,_=c.intervals(s,t,e,(np.ones(100),np.ones(20)),2022,('W4',),g1)
        e['truth'][5:]=1e6
        r2,_=c.intervals(s,t,e,(np.ones(100),np.ones(20)),2022,('W4',),g2)
        np.testing.assert_array_equal(r1['W4'][:5],r2['W4'][:5])
        with self.assertRaises(ValueError): c.weather_fit(x,x,times,times+np.timedelta64(365,'D'),2022)

    def test_feedback_returns_to_original_group(self):
        s,t,e=pack(500,'2021-01-01'),pack(100,'2021-12-01'),pack(5,'2022-01-01')
        g=np.array([0,1,1,1,0]); e['truth'][:]=0
        r0,_=c.intervals(s,t,e,(np.ones(500),np.ones(5)),2022,('W4',),g)
        e['truth'][0,0]=10000
        r1,_=c.intervals(s,t,e,(np.ones(500),np.ones(5)),2022,('W4',),g)
        np.testing.assert_array_equal(r0['W4'][1:4],r1['W4'][1:4])
        self.assertGreater(r1['W4'][4,0],r0['W4'][4,0])
        np.testing.assert_array_equal(r0['W4'][:,1:],r1['W4'][:,1:])

    def test_selection_ties_and_no_eligible(self):
        def row():
            return {'mean_90':{m:{'coverage':.9,'winkler_kw':100.} for m in c.VARIANTS+('B3-A',)},
                    'horizons':{m:[{'winkler_kw':100.}]*4 for m in ('M-A','B3-A')}}
        a,b=row(),row()
        self.assertEqual(c.choose([a,b]),['B3-A']*4)
        self.assertEqual(c.select_variant([a,b])[0],'W1')
        for m in c.VARIANTS: b['mean_90'][m]['coverage']=.95
        self.assertIsNone(c.select_variant([a,b])[0])

    def test_combination_horizons(self):
        r={'M-A':np.ones((5,4)),'B3-A':np.full((5,4),2.)}; caps={m:r[m]>1 for m in r}
        z,k=c.combination(r,caps,['B3-A','M-A','M-A','B3-A'])
        np.testing.assert_array_equal(z[0],[2,1,1,2]); np.testing.assert_array_equal(k[0],[True,False,False,True])

    def test_cluster_bootstrap_preserves_counts_and_indices(self):
        issue=np.array([0,0,1,2,2,2,3,4,5,6,7,8],dtype='datetime64[h]')
        diff=np.tile(np.arange(12)[:,None],(1,4)).astype(float)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'b.npz'; report=c.paired_ci(diff,issue,path,100)
            self.assertEqual(report['mean_gain_kw'],diff.mean())
            with np.load(path) as z:
                idx=(z['block_starts'][0,:,None]+np.arange(int(z['block_length']))).ravel()[:int(z['truncate_to'])]
                self.assertEqual(z['estimates'][0],z['cluster_sums'][idx].sum()/z['cluster_counts'][idx].sum())

    def test_calibration_and_nonfinite(self):
        for year in (2022,2023,2024):
            s,t,e=pack(20,f'{year-1}-01-01'),pack(20,f'{year-1}-12-01'),pack(10,f'{year}-01-01')
            c.intervals(s,t,e,(np.ones(20),np.ones(10)),year,c.BASE)
            s['times'][0,0]=np.datetime64(f'{year}-01-01')
            with self.assertRaises(ValueError): c.intervals(s,t,e,(np.ones(20),np.ones(10)),year,c.BASE)
        with self.assertRaises(ValueError): c.reports({'W1':np.full((2,4),np.inf)},{'W1':np.zeros((2,4),bool)},np.ones((2,4)))

    def test_W4_best_comparator_and_CI(self):
        means={m:{'coverage':.9,'winkler_kw':100.} for m in c.BASE+('B3-G','W4')}
        means['W4']['winkler_kw']=90.
        ci={m:{'ci95_low_kw':1.} for m in ('B3-A','B3-G')}
        self.assertTrue(c.verdict(means,'W4',ci)['pass_all'])
        means['B3-G']['winkler_kw']=91.
        self.assertFalse(c.verdict(means,'W4',ci)['C3'])


if __name__=='__main__': unittest.main()
