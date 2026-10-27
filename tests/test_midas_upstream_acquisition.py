"""Provider identity, credential privacy, byte resume and sealed isolation; no live HTTP."""
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

from scripts import acquire_midas_upstream_observations as a


class Response:
    def __init__(self, status=200, body=b'', headers=None, obj=None, interrupt=False):
        self.status_code, self.body = status, body
        self.headers, self.obj, self.interrupt = headers or {}, obj, interrupt
        self.closed = False
    def close(self): self.closed = True
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def json(self): return self.obj
    def iter_content(self, size):
        yield self.body
        if self.interrupt: raise requests.exceptions.ChunkedEncodingError('interrupted body')


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.token = self.root/'credential'
        self.token.write_text('private.test.credential\n')
        self.token.chmod(0o600)
        self.archive = a.Archive(self.token)
        self.item = {'path': a.BASE+'/station/example_2016.csv', 'name': 'example_2016.csv',
                     'type':'file', 'size':6, 'md5':hashlib.md5(b'abcdef').hexdigest()}
        self.target = self.root/'development'/'example_2016.csv'

    def test_external_and_traversal_paths_rejected(self):
        for path in ('https://evil.invalid', a.BASE+'/../secret', a.BASE+'/file?key=value', a.BASE+'-wrong/file'):
            with self.subTest(path=path), self.assertRaises(ValueError): a.safe_path(path)

    def test_overlapping_or_nonsealed_roots_rejected(self):
        for dev, sealed in ((self.root,self.root/'sealed'),(self.root/'d',self.root/'other'),
                            (self.root/'sealed/d',self.root/'sealed')):
            with self.assertRaises(ValueError): a.checked_roots(dev,sealed)

    def test_authentication_is_header_only_without_redirect(self):
        with mock.patch.object(a.requests, 'get', return_value=Response()) as get:
            self.archive.get(self.item['path'])
        args, kwargs = get.call_args
        self.assertNotIn('credential', args[0])
        self.assertEqual(kwargs['headers'], {'Authorization':'Bearer private.test.credential'})
        self.assertFalse(kwargs['allow_redirects'])

    @unittest.skipIf(os.name=='nt','POSIX credential mode check is enforced on server')
    def test_world_readable_token_refused(self):
        self.token.chmod(0o644)
        with self.assertRaises(a.AuthorizationError): self.archive.headers()

    def test_public_listing_does_not_send_credential_or_keep_preview(self):
        item={**self.item,'content':'FUTURE OBSERVATION PREVIEW MUST NOT BE USED'}
        with mock.patch.object(a.requests,'get',return_value=Response(obj={'items':[item]})) as get:
            found=self.archive.listing(a.BASE+'/station')
        self.assertEqual(get.call_args.kwargs['headers'],{})
        self.assertNotIn('content',found[0])

    def test_listing_child_escape_refused(self):
        with mock.patch.object(self.archive,'get',return_value=Response(obj={'items':[self.item]})):
            with self.assertRaisesRegex(ValueError,'escaped'): self.archive.listing(a.BASE+'/other')

    def test_expired_auth_or_redirect_not_retried_or_dumped(self):
        for status in (302,401,403):
            with mock.patch.object(a.requests,'get',return_value=Response(status=status)) as get:
                with self.assertRaises(a.AuthorizationError): self.archive.get(self.item['path'])
                self.assertEqual(get.call_count,1)

    def test_network_failure_does_not_echo_exception_secret(self):
        with mock.patch.object(a.requests,'get',side_effect=requests.ConnectionError('private.test.credential')), mock.patch.object(a.time,'sleep'):
            with self.assertRaises(RuntimeError) as caught: self.archive.get(self.item['path'])
        self.assertNotIn('private.test.credential',str(caught.exception))

    def test_retry_after_is_respected(self):
        with mock.patch.object(a.requests,'get',side_effect=[Response(429,headers={'Retry-After':'7'}),Response()]), mock.patch.object(a.time,'sleep') as sleep:
            self.archive.get(self.item['path'])
        sleep.assert_called_once_with(7)

    def test_full_bytes_md5_verified_and_published(self):
        with mock.patch.object(self.archive,'get',return_value=Response(body=b'abcdef')):
            self.archive.fetch(self.item,self.target)
        self.assertEqual(self.target.read_bytes(),b'abcdef')
        self.assertFalse(self.target.with_name(self.target.name+'.part').exists())

    def test_interrupted_body_reuses_range_bytes(self):
        responses=[Response(body=b'abc',interrupt=True),Response(206,body=b'def',headers={'Content-Range':'bytes 3-5/6'})]
        with mock.patch.object(self.archive,'get',side_effect=responses) as get, mock.patch.object(a.time,'sleep'):
            self.archive.fetch(self.item,self.target)
        self.assertEqual(get.call_args_list[1].kwargs['offset'],3)
        self.assertEqual(self.target.read_bytes(),b'abcdef')

    def test_existing_verified_file_no_network(self):
        self.target.parent.mkdir()
        self.target.write_bytes(b'abcdef')
        with mock.patch.object(self.archive,'get') as get: self.archive.fetch(self.item,self.target)
        get.assert_not_called()

    def test_bad_existing_bytes_preserved(self):
        self.target.parent.mkdir()
        self.target.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'Existing file'): self.archive.fetch(self.item,self.target)
        self.assertEqual(self.target.read_bytes(),b'bad')

    def test_bad_checksum_keeps_fragment_not_published(self):
        with mock.patch.object(self.archive,'get',return_value=Response(body=b'xxxxxx')):
            with self.assertRaisesRegex(ValueError,'MD5'): self.archive.fetch(self.item,self.target)
        self.assertFalse(self.target.exists())
        self.assertTrue(self.target.with_name(self.target.name+'.part').exists())

    def test_login_html_never_becomes_csv(self):
        with mock.patch.object(self.archive,'get',return_value=Response(body=b'<html>',headers={'Content-Type':'text/html'})):
            with self.assertRaisesRegex(ValueError,'dataset file'): self.archive.fetch(self.item,self.target)
        self.assertFalse(self.target.exists())

    def test_ignored_range_preserved_not_appended(self):
        self.target.parent.mkdir()
        part=self.target.with_name(self.target.name+'.part'); part.write_bytes(b'abc')
        with mock.patch.object(self.archive,'get',return_value=Response(body=b'abcdef')):
            with self.assertRaisesRegex(ValueError,'ignored resume'): self.archive.fetch(self.item,self.target)
        self.assertEqual(part.read_bytes(),b'abc')

    def metadata(self):
        p=self.root/'metadata.csv'
        p.write_text('src_id,station_name,station_file_name,historic_county,station_latitude,station_longitude,first_year,last_year\n'
                     '1,Nearby,nearby,county,52.4,-.94,2010,2025\n'
                     '2,Closed,closed,county,52.4,-.94,1900,2010\n'
                     '3,Far,far,county,0,0,2010,2025\n')
        return p

    def test_plan_uses_geography_dates_and_requested_years_only(self):
        meta=self.metadata()
        entries=[{**self.item,'path':a.BASE+f'/county/00001_nearby/qc-version-1/x_{y}.csv','name':f'x_{y}.csv'} for y in (2016,2022,2023,2024,2025)]
        with mock.patch.object(self.archive,'listing',return_value=entries) as listing:
            plan=a.make_plan(self.archive,meta,{'md5':'a'*32})
        self.assertEqual(plan['candidate_stations'],2)
        self.assertEqual(plan['stations_overlapping_requested_years'],1)
        self.assertEqual([x['year'] for x in plan['files']],[2016,2022,2023,2024])
        self.assertEqual([x['sealed'] for x in plan['files']],[False,False,True,True])
        self.assertIn('qc-version-1',listing.call_args.args[0])

    def test_screen_completion_reused_and_tamper_refused(self):
        dev=self.root/'dev'; dev.mkdir()
        metadata=dev/'metadata.csv'; metadata.write_text('metadata')
        ev=self.root/'run'/'evaluation'; ev.mkdir(parents=True)
        (ev.parent/'completed.json').write_text('{}')
        audit=self.root/'audit'; audit.mkdir()
        with mock.patch.object(a,'load_year'), mock.patch.object(a,'screen',return_value={'status':'SCREENED','passed':False}) as fitter:
            a.run_screen(dev,metadata,{2021:ev},audit)
            a.run_screen(dev,metadata,{2021:ev},audit)
            self.assertEqual(fitter.call_count,1)
            (audit/'screen.json').write_text('{"passed":true}')
            with self.assertRaisesRegex(ValueError,'identity differs'): a.run_screen(dev,metadata,{2021:ev},audit)
            self.assertEqual(fitter.call_count,1)

    def fake_archive(self):
        metadata=self.metadata().read_bytes()
        meta={'name':a.META,'path':a.BASE+'/'+a.META,'type':'file','size':len(metadata),'md5':hashlib.md5(metadata).hexdigest()}
        rows=[]; payloads={meta['path']:metadata}
        for year in (2016,2022,2023,2024):
            name=f'midas-open_uk-hourly-weather-obs_dv-202607_county_00001_nearby_qcv-1_{year}.csv'
            body=b'RAW SEALED BYTES NOT A CSV' if year>=2023 else f'development raw bytes {year}'.encode()
            item={'name':name,'path':a.BASE+'/county/00001_nearby/qc-version-1/'+name,'type':'file','size':len(body),'md5':hashlib.md5(body).hexdigest()}
            rows.append(item); payloads[item['path']]=body
        class Fake:
            def listing(self,path): return [meta] if path==a.BASE else rows
            def fetch(self,item,path):
                path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():
                    assert path.read_bytes()==payloads[item['path']]
                else: path.write_bytes(payloads[item['path']])
        return Fake()

    def test_acquisition_separates_raw_sealed_and_screens_after_development(self):
        dev, sealed, audit = self.root/'dev',self.root/'sealed'/'midas',self.root/'audit'
        def screening(*args):
            self.assertEqual(len(list(dev.rglob('*_2016.csv'))),1)
            self.assertFalse(sealed.exists())
        with mock.patch.object(a,'run_screen',side_effect=screening) as screen:
            result=a.acquire(self.fake_archive(),dev,sealed,audit,{2021:self.root/'eval'})
        screen.assert_called_once()
        self.assertEqual(result['status'],'DOWNLOADED_VERIFIED')
        self.assertEqual(len(result['files']),4)
        self.assertEqual(len(list(dev.rglob('*_2023.csv'))),0)
        self.assertEqual(len(list(sealed.rglob('*_2023.csv'))),1)

    def test_acquisition_resume_reuses_pinned_plan_and_raw_bytes(self):
        dev, sealed, audit = self.root/'dev',self.root/'sealed'/'midas',self.root/'audit'
        archive=self.fake_archive()
        first=a.acquire(archive,dev,sealed,audit)
        with mock.patch.object(a,'make_plan',side_effect=AssertionError('plan rebuilt')):
            second=a.acquire(archive,dev,sealed,audit)
        self.assertEqual(first['plan_sha256'],second['plan_sha256'])
        self.assertEqual(first['files'],second['files'])

    def test_cached_plan_cannot_move_sealed_year_into_development(self):
        dev, sealed, audit = self.root/'dev',self.root/'sealed'/'midas',self.root/'audit'
        archive=self.fake_archive(); a.acquire(archive,dev,sealed,audit)
        p=audit/'download_plan.json'; plan=json.loads(p.read_text())
        next(x for x in plan['files'] if x['year']==2023)['sealed']=False
        p.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'sealed assignment'): a.acquire(archive,dev,sealed,audit)

    def test_interruption_records_state_without_deleting_published_bytes(self):
        dev, sealed, audit = self.root/'dev',self.root/'sealed'/'midas',self.root/'audit'
        archive=self.fake_archive(); original=archive.fetch
        def fail(item,path):
            if item['name'].endswith('_2023.csv'): raise RuntimeError('network interrupted')
            original(item,path)
        archive.fetch=fail
        with self.assertRaisesRegex(RuntimeError,'network interrupted'): a.acquire(archive,dev,sealed,audit)
        state=json.loads((audit/'download_manifest.json').read_text())
        self.assertEqual(state['status'],'INTERRUPTED')
        self.assertEqual(len(list(dev.rglob('*_2016.csv'))),1)
        self.assertTrue(all(not x['sealed'] for x in state['files']))

    def test_screen_identity_changes_refuse_repeated_fit(self):
        dev=self.root/'dev'; dev.mkdir()
        metadata=dev/'metadata.csv'; metadata.write_text('metadata')
        ev=self.root/'run'/'evaluation'; ev.mkdir(parents=True)
        (ev.parent/'completed.json').write_text('{}')
        audit=self.root/'audit'; audit.mkdir()
        with mock.patch.object(a,'load_year'),mock.patch.object(a,'screen',return_value={'status':'SCREENED','passed':False}) as fitter:
            a.run_screen(dev,metadata,{2021:ev},audit)
            metadata.write_text('different metadata')
            with self.assertRaisesRegex(ValueError,'identity differs'): a.run_screen(dev,metadata,{2021:ev},audit)
        self.assertEqual(fitter.call_count,1)

    def test_invalid_screen_not_negative_gain_or_repeated_fit(self):
        dev=self.root/'dev'; dev.mkdir()
        metadata=dev/'metadata.csv'; metadata.write_text('metadata')
        ev=self.root/'run'/'evaluation'; ev.mkdir(parents=True)
        (ev.parent/'completed.json').write_text('{}')
        audit=self.root/'audit'; audit.mkdir()
        with mock.patch.object(a,'load_year'), mock.patch.object(a,'screen',side_effect=ValueError('insufficient stations')) as fitter:
            a.run_screen(dev,metadata,{2021:ev},audit)
            a.run_screen(dev,metadata,{2021:ev},audit)
        self.assertEqual(fitter.call_count,1)
        self.assertIsNone(json.loads((audit/'screen.json').read_text())['passed'])


if __name__=='__main__': unittest.main()
