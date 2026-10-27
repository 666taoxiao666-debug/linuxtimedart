"""Test download scope, predecessor barriers and byte-only external-data handling."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import requests

from scripts import acquire_sealed_external_validation as s
from scripts.acquire_midas_upstream_observations import BASE, DATASET, RADIUS_KM


def md5(data):
    return hashlib.md5(data).hexdigest()


class ExternalAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.body = b'not a CSV or valid archive; raw bytes only'
        self.fig = {'id': 24798654, 'version': 2, 'doi': '10.6084/m9.figshare.24798654.v2',
                    'license': {'url': 'https://creativecommons.org/licenses/by/4.0/'}, 'files': []}
        for file_id in [11, 12]:
            self.fig['files'].append({'id': file_id, 'name': '.DS_Store', 'size': len(self.body),
                                      'computed_md5': md5(self.body), 'supplied_md5': md5(self.body),
                                      'download_url': f'https://ndownloader.figshare.com/files/{file_id}'})
        self.pen = {'id': 16807304, 'conceptrecid': '5946807', 'doi': '10.5281/zenodo.16807304',
                    'metadata': {'license': {'id': 'cc-by-4.0'}},
                    'files': [{'id': 'abcd-1234', 'key': 'file with spaces.zip', 'size': len(self.body),
                               'checksum': 'md5:'+md5(self.body), 'links': {
                                   'self': 'https://zenodo.org/api/records/16807304/files/file%20with%20spaces.zip/content'}}]}
        self.figpath, self.penpath = self.root/'fig.json', self.root/'pen.json'
        self.audit = self.root/'audit'
        self.sealed = self.root/'sealed'/'external'
        self.refresh()

    def refresh(self):
        s.save_json(self.figpath, self.fig)
        s.save_json(self.penpath, self.pen)
        self.plan = s.build_inventory(self.figpath, self.penpath)
        s.save_json(self.audit/'inventory.json', self.plan)

    def download(self, url, target, size, checksum, workers):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.body)

    def prerequisites(self):
        self.ma = self.root/'midas_audit'
        self.dev = self.root/'development'
        self.ms = self.root/'sealed'/'midas'
        self.kel = self.root/'sealed'/'kelmarsh'
        self.dev.mkdir(exist_ok=True)
        (self.dev/s.META).write_text('station metadata only')
        name = 'midas-open_dv-202607_qcv-1_2016.csv'
        data = self.dev/'381'/name
        data.parent.mkdir(exist_ok=True)
        data.write_bytes(self.body)
        provider = BASE+'/county/00381_example/qc-version-1/'+name
        entry = {'name': name, 'path': provider, 'year': 2016, 'src_id': 381, 'sealed': False,
                 'md5': md5(self.body), 'size': len(self.body)}
        p = {'dataset': DATASET, 'radius_km': RADIUS_KM, 'metadata_sha256': s.digest(self.dev/s.META), 'files': [entry]}
        s.save_json(self.ma/'download_plan.json', p)
        manifest = {'status': 'DOWNLOADED_VERIFIED', 'plan_sha256': s.digest(self.ma/'download_plan.json'),
                    'planned_files': 1, 'planned_bytes': len(self.body),
                    'files': [{'provider_path': provider, 'src_id': 381, 'year': 2016, 'sealed': False,
                               'path': str(data.resolve()), 'bytes': len(self.body), 'md5': md5(self.body),
                               'sha256': s.digest(data)}]}
        s.save_json(self.ma/'download_manifest.json', manifest)
        k = {'status': 'SEALED_VERIFIED', 'record_id': 16807551, 'files': []}
        self.kel.mkdir(parents=True)
        for year, name in s.FILES.items():
            path = self.kel/name
            path.write_bytes(self.body)
            k['files'].append({'year': year, 'file': name, 'bytes': len(self.body),
                               'md5': md5(self.body), 'sha256': s.digest(path)})
        s.save_json(self.kel/'download_manifest.json', k)
        patch = mock.patch.object(s, 'OFFICIAL', {y: (len(self.body), md5(self.body)) for y in (2023, 2024)})
        patch.start()
        self.addCleanup(patch.stop)

    def verify(self):
        return s.verify_prerequisites(self.ma, self.dev, self.ms, self.kel)

    def test_all_files_include_duplicate_names_without_collision(self):
        entries = [x for d in self.plan['datasets'] for x in d['files']]
        self.assertEqual(len(entries), 3)
        self.assertEqual(len({x['relative_path'] for x in entries}), 3)
        self.assertEqual([x['name'] for x in entries[:2]], ['.DS_Store', '.DS_Store'])

    def test_wrong_record_version_parent_or_license_refused(self):
        for target, key, value in [(self.fig, 'version', 1), (self.pen, 'conceptrecid', 'wrong'),
                                   (self.fig, 'id', 99)]:
            old = target[key]
            target[key] = value
            with self.assertRaises((ValueError, TypeError)): self.refresh()
            target[key] = old
        self.pen['metadata']['license']['id'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'license'): self.refresh()

    def test_provider_checksums_disagreement_refused(self):
        self.fig['files'][0]['supplied_md5'] = '0'*32
        with self.assertRaisesRegex(ValueError, 'disagree'): self.refresh()

    def test_link_only_file_refused_not_silently_skipped(self):
        self.fig['files'][0]['is_link_only'] = True
        with self.assertRaisesRegex(ValueError, 'Link-only'): self.refresh()

    def test_unsafe_names_and_paths_refused(self):
        for name in ['../escape', 'a/b', 'a\\b', '..', 'C:evil', 'a\n']:
            with self.assertRaises(ValueError): s.filename(name)
        with self.assertRaises(ValueError): s.contained(self.root, '../outside')
        with self.assertRaises(ValueError): s.sealed_root(self.root/'ordinary')

    def test_endpoint_and_file_id_tampering_refused(self):
        for field, value in [('url', 'https://evil.invalid/secret'), ('relative_path', '../escape'),
                              ('md5', ''), ('bytes', -1)]:
            changed = copy.deepcopy(self.plan)
            changed['datasets'][0]['files'][0][field] = value
            with self.assertRaises(ValueError): s.validate_inventory(changed)
        changed = copy.deepcopy(self.plan)
        changed['datasets'][0]['files'].append(changed['datasets'][0]['files'][0])
        with self.assertRaises(ValueError): s.validate_inventory(changed)

    def test_missing_or_interrupted_predecessor_never_ready(self):
        self.prerequisites()
        path = self.ma/'download_manifest.json'
        for status in ['DOWNLOADING', 'INTERRUPTED', 'SCREENED']:
            m = json.loads(path.read_text()); m['status'] = status; s.save_json(path, m)
            self.assertFalse(s.statuses(self.ma, self.kel)[0])
            with self.assertRaisesRegex(ValueError, 'not both completed'): self.verify()

    def test_complete_predecessors_verified_without_parsing_values(self):
        self.prerequisites()
        with mock.patch('pandas.read_csv', side_effect=AssertionError('data parsed')), \
             mock.patch('zipfile.ZipFile', side_effect=AssertionError('archive opened')):
            evidence = self.verify()
        self.assertEqual(set(evidence), {'midas_manifest_sha256', 'midas_plan_sha256', 'kelmarsh_manifest_sha256'})

    def test_predecessor_same_size_corruption_refused(self):
        self.prerequisites()
        (self.kel/s.FILES[2023]).write_bytes(b'x'*len(self.body))
        with self.assertRaisesRegex(ValueError, 'raw bytes'): self.verify()

    def test_missing_duplicate_or_repointed_predecessor_refused(self):
        self.prerequisites()
        path = self.ma/'download_manifest.json'
        original = json.loads(path.read_text())
        for mode in ['missing', 'duplicate', 'repoint']:
            m = copy.deepcopy(original)
            if mode == 'missing': m['files'] = []
            elif mode == 'duplicate': m['files'] *= 2
            else: m['files'][0]['path'] = str(self.root/'other')
            s.save_json(path, m)
            with self.assertRaises(ValueError): self.verify()

    def test_full_acquisition_and_completed_reuse_without_http_or_parsing(self):
        with mock.patch('pandas.read_csv', side_effect=AssertionError('data parsed')), \
             mock.patch('zipfile.ZipFile', side_effect=AssertionError('archive opened')):
            result = s.acquire(self.plan, self.sealed, self.audit, {'verified': True}, self.download)
            self.assertEqual(result['status'], 'SEALED_VERIFIED')
            self.assertEqual(len(result['files']), 3)
            with mock.patch.object(s, 'checked_download', side_effect=AssertionError('network')):
                reused = s.acquire(self.plan, self.sealed, self.audit, {'verified': True},
                                   downloader=mock.Mock(side_effect=AssertionError('network')))
            self.assertEqual(result, reused)

    def test_completed_byte_corruption_refused_not_redownloaded(self):
        result = s.acquire(self.plan, self.sealed, self.audit, {}, self.download)
        path = s.contained(self.sealed, result['files'][0]['relative_path'])
        path.write_bytes(b'x'*len(self.body))
        with self.assertRaisesRegex(ValueError, 'raw bytes'):
            s.acquire(self.plan, self.sealed, self.audit, {}, mock.Mock(side_effect=AssertionError('network')))

    def test_changed_inventory_cannot_reinterpret_existing_files(self):
        changed = copy.deepcopy(self.plan)
        changed['datasets'][0]['files'][0]['md5'] = '0'*32
        with self.assertRaisesRegex(ValueError, 'registered inventory'):
            s.acquire(changed, self.sealed, self.audit, {}, self.download)

    def test_network_error_retried_but_identity_error_not_retried(self):
        calls = []
        def flaky(*args, **kwargs):
            calls.append(args[0])
            if len(calls) == 1: raise requests.ConnectionError('transient')
            return self.download(*args, **kwargs)
        sleep = mock.Mock()
        result = s.acquire(self.plan, self.sealed, self.audit, {}, flaky, sleep=sleep)
        self.assertEqual(result['status'], 'SEALED_VERIFIED')
        sleep.assert_called_once_with(60)
        other = self.root/'other_audit'; s.save_json(other/'inventory.json', self.plan)
        bad = mock.Mock(side_effect=ValueError('MD5 mismatch'))
        with self.assertRaisesRegex(ValueError, 'MD5 mismatch'):
            s.acquire(self.plan, self.sealed, other, {}, bad, sleep=sleep)
        bad.assert_called_once()
        self.assertEqual(json.loads((other/'download_manifest.json').read_text())['status'], 'INTERRUPTED')

    def test_provider_denial_not_retried_by_outer_transport_loop(self):
        deny = mock.Mock(side_effect=requests.HTTPError('403 Forbidden'))
        sleep = mock.Mock()
        with self.assertRaises(requests.HTTPError):
            s.acquire(self.plan, self.sealed, self.audit, {}, deny, sleep=sleep)
        deny.assert_called_once()
        sleep.assert_not_called()

    @unittest.skipIf(os.name == 'nt', 'Linux fcntl queue lock enforced on server')
    def test_queue_waits_for_both_predecessors_then_transfers(self):
        self.prerequisites()
        args = SimpleNamespace(sealed_root=self.sealed, audit_root=self.audit, midas_audit=self.ma,
                               midas_development=self.dev, midas_sealed=self.ms, kelmarsh_sealed=self.kel,
                               poll_seconds=30)
        path = self.ma/'download_manifest.json'
        complete = json.loads(path.read_text())
        pending = {**complete, 'status': 'DOWNLOADING'}; s.save_json(path, pending)
        transfer = mock.Mock(side_effect=self.download)
        def advance(seconds):
            transfer.assert_not_called()
            self.assertFalse(self.sealed.exists())
            s.save_json(path, complete)
        result = s.run_queue(args, sleep=advance, downloader=transfer)
        self.assertEqual(result['status'], 'SEALED_VERIFIED')
        self.assertEqual(transfer.call_count, 3)


if __name__ == '__main__':
    unittest.main()
