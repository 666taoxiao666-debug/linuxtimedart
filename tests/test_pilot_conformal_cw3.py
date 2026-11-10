import unittest
from pathlib import Path
import numpy as np
from scripts import pilot_conformal_cw3 as c
from tests.test_pilot_conformal_cw2 import pack


def unit_bundle(source, test):
    e = np.abs(source['truth'] - source['prediction'])
    return {a: {'scores': [e[:, h]/2 for h in range(4)],
                'sigma_source': [np.full(len(e), 2.) for _ in range(4)],
                'sigma_test': [np.full(len(test['issue']), 2.) for _ in range(4)],
                'floors': [1.]*4} for a in ('P', 'R')}


def smoke(out):
    s, t, e = pack(550, '2021-01-01', 1), pack(80, '2021-12-01', 2), pack(24, '2022-01-01', 3)
    r, caps, bundle = c.intervals(s, t, e, 2022, c.BASE+c.SCALE_ARMS)
    result = {'status': 'PASSED', 'smoke': True,
              'methods': c.cw1.reports(r, caps, np.abs(e['truth']-e['prediction']))}
    c.save(Path(out)/'result.json', result)
    return result


class CW3Tests(unittest.TestCase):
    def test_constant_sigma_reduces_to_power_neighbors(self):
        s, t, e = pack(650, '2021-01-01', 1), pack(70, '2021-12-01', 2), pack(40, '2022-01-01', 3)
        r, caps, _ = c.intervals(s,t,e,2022,c.BASE+c.SCALE_ARMS,unit_bundle(s,e))
        for arm in c.SCALE_ARMS:
            np.testing.assert_allclose(r[arm], r['B3-A'], rtol=1e-12)
            np.testing.assert_array_equal(caps[arm],caps['B3-A'])

    def test_joint_neighbors_match_bruteforce_and_ties(self):
        rng = np.random.default_rng(4)
        p, sig = rng.normal(size=700), np.exp(rng.normal(size=700))
        ps, ss = np.array([.5,-.2]), np.array([2.,3.])
        x=np.c_[p,np.log(sig)]; scale=x.std(axis=0)
        for got, point in zip(c.joint_neighbors(p,sig,ps,ss),np.c_[ps,np.log(ss)]):
            d=((x-point)/scale)**2
            expected=np.lexsort((np.arange(len(p)),d.sum(1)))[:500]
            np.testing.assert_array_equal(got,expected)
        got=next(c.joint_neighbors(np.zeros(600),np.ones(600),[0],[1]))
        np.testing.assert_array_equal(got,np.arange(500))

    def test_delayed_feedback_and_future_truth_invariance(self):
        s,t,e=pack(60,'2021-01-01',1),pack(60,'2021-12-01',2),pack(20,'2022-01-01',3)
        e2={k:v.copy() for k,v in e.items()}; e2['truth'][0]=1e8
        r,_,_=c.intervals(s,t,e,2022,c.SCALE_ARMS,unit_bundle(s,e))
        r2,_,_=c.intervals(s,t,e2,2022,c.SCALE_ARMS,unit_bundle(s,e2))
        for arm in c.SCALE_ARMS:
            for h in range(4):
                np.testing.assert_array_equal(r[arm][:h+1,h],r2[arm][:h+1,h])
        e2['times'][0,0]=np.datetime64('2022-01-01')
        with self.assertRaises(ValueError):
            c.intervals(s,t,e2,2022,c.SCALE_ARMS,unit_bundle(s,e2))

    def test_rule_f_caps_local_group(self):
        s,t,e=pack(30,'2021-01-01'),pack(30,'2021-12-01'),pack(80,'2022-01-01')
        e['truth'][:]=1e8
        r,caps,_=c.intervals(s,t,e,2022,c.SCALE_ARMS,unit_bundle(s,e))
        for arm in c.SCALE_ARMS:
            self.assertTrue(caps[arm].any())
            self.assertTrue(np.isfinite(r[arm]).all())
            np.testing.assert_allclose(r[arm][-1],np.abs(s['truth']-s['prediction']).max(0))

    def test_selection_all_cohorts_and_matched_controls(self):
        def cohort():
            return {'mean_90':{m:{'coverage':.90,'winkler_kw':95. if m in c.CANDIDATES else 100.}
                              for m in c.BASE+c.SCALE_ARMS}}
        rows=[cohort() for _ in range(4)]
        self.assertEqual(c.select_candidate(rows)[0],'LN-R')
        rows[0]['mean_90']['LN-P']['winkler_kw']=94.
        self.assertEqual(c.select_candidate(rows)[0],'LJ-R')
        rows[1]['mean_90']['LJ-R']['coverage']=.87
        self.assertIsNone(c.select_candidate(rows)[0])

    def test_judgment_truth_table(self):
        def judge(wink,ci=1):
            means={m:{'coverage':.90,'winkler_kw':110.} for m in c.BASE+c.SCALE_ARMS}
            means['LN-P']['winkler_kw']=100.
            means['LN-R']['winkler_kw']=wink
            return c.verdict(means,'LN-R',{'LN-P':{'ci95_low_kw':ci},'B3-A':{'ci95_low_kw':ci}})
        self.assertEqual(c.combined_verdict([judge(96),judge(97)]),'EXPLORATORY_SUPPORT_PENDING_CONFIRMATION')
        self.assertEqual(c.combined_verdict([judge(96),judge(98)]),'SMALL_SIGNIFICANT_EXPLORATORY')
        self.assertEqual(c.combined_verdict([judge(96),judge(98,-1)]),'STOP_CW3')

    def test_source_only_scale_fit(self):
        s=pack(500,'2021-01-01'); e=pack(10,'2022-01-01')
        one=c.fit_scale_models(s,e,2022)
        e['truth'][:]=1e8
        two=c.fit_scale_models(s,e,2022)
        for arm in ('P','R'):
            np.testing.assert_array_equal(one[arm]['sigma_source'],two[arm]['sigma_source'])
            np.testing.assert_array_equal(one[arm]['sigma_test'],two[arm]['sigma_test'])


if __name__=='__main__':
    unittest.main()
