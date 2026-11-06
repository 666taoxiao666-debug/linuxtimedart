import unittest
import numpy as np
from scripts.pilot_conformal_c0 import causal_check, weighted_quantile, direction_features, intervals, verdict

class C0Tests(unittest.TestCase):
    def test_future_calibration_rejected(self):
        with self.assertRaises(ValueError):
            causal_check(np.array(['2022-01-01'],dtype='datetime64[ns]'),np.array(['2022-02-01'],dtype='datetime64[ns]'))
        with self.assertRaises(ValueError):
            causal_check(np.array(['2021-12-31T23:00'],dtype='datetime64[ns]'),np.array(['2021-12-31T22:00'],dtype='datetime64[ns]'))

    def test_test_mass_and_equal_weight_correction(self):
        self.assertEqual(weighted_quantile(np.arange(1,10),np.ones(9),1,.1),9)
        self.assertTrue(np.isinf(weighted_quantile(np.arange(1,10),np.ones(9),10,.1)))
        self.assertEqual(weighted_quantile([1,5],[9,1],1,.5),1)

    def test_direction_schema_no_labels(self):
        h=np.zeros((3,11)); x=direction_features(h,[0,359,np.nan])
        self.assertEqual(x.shape,(3,20)); np.testing.assert_array_equal(x[:,11:].argmax(1),[0,7,8])

    def test_delayed_feedback(self):
        def pack(n,date):
            issue=np.datetime64(date,'ns')+np.arange(n)*np.timedelta64(1,'h')
            return dict(prediction=np.zeros((n,4)),truth=np.ones((n,4)),issue=issue,
                        times=issue[:,None]+np.arange(1,5)*np.timedelta64(1,'h'),
                        turbines=np.ones(n,int),missing=np.zeros(n,bool))
        s,t,e=pack(100,'2021-01-01'),pack(100,'2021-12-01'),pack(8,'2022-01-01')
        _,a=intervals(s,t,e,(np.ones(100),np.ones(8)),.1)
        e['truth'][0,3]=1e6
        _,b=intervals(s,t,e,(np.ones(100),np.ones(8)),.1)
        np.testing.assert_array_equal(a['M+'][:4],b['M+'][:4])
        e['times'][0,3]+=np.timedelta64(1,'h')
        with self.assertRaises(ValueError): intervals(s,t,e,(np.ones(100),np.ones(8)),.1)

    def test_decision_fallback(self):
        means={m:dict(coverage=.8,winkler_kw=100.) for m in ('B0','B1','B3')}
        means['M']=dict(coverage=.9,winkler_kw=90.)
        self.assertEqual(verdict(means)['decision'],'continue')
        means['M']['coverage']=.93
        self.assertEqual(verdict(means)['decision'],'stop')

if __name__=='__main__': unittest.main()
