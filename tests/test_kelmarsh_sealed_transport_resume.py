import unittest
from unittest import mock

import requests

from scripts.resume_kelmarsh_sealed_transport import resume_transport


class ResumeTests(unittest.TestCase):
    def test_transport_retry_preserves_same_operation_inputs(self):
        record, root = {'id': 16807551}, 'sealed/raw'
        operation = mock.Mock(side_effect=[requests.exceptions.ChunkedEncodingError('cut'),
                                           {'status': 'SEALED_VERIFIED'}])
        sleep = mock.Mock()
        result = resume_transport(record, root, operation=operation, sleep=sleep)
        self.assertEqual(result['status'], 'SEALED_VERIFIED')
        self.assertEqual(operation.call_args_list, [mock.call(record, root)]*2)
        sleep.assert_called_once_with(300)

    def test_identity_and_provider_errors_are_not_retried(self):
        for error in [ValueError('MD5'), requests.HTTPError('403'), RuntimeError('Retry-After too long')]:
            operation, sleep = mock.Mock(side_effect=error), mock.Mock()
            with self.assertRaises(type(error)):
                resume_transport({}, 'sealed/raw', operation=operation, sleep=sleep)
            operation.assert_called_once()
            sleep.assert_not_called()

    def test_bounded_attempts_propagate_final_failure(self):
        operation, sleep = mock.Mock(side_effect=requests.Timeout('timeout')), mock.Mock()
        with self.assertRaises(requests.Timeout):
            resume_transport({}, 'sealed/raw', operation=operation, sleep=sleep, attempts=3)
        self.assertEqual(operation.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_completed_operation_returns_without_retry(self):
        operation, sleep = mock.Mock(return_value={'status': 'SEALED_VERIFIED'}), mock.Mock()
        resume_transport({}, 'sealed/raw', operation=operation, sleep=sleep)
        operation.assert_called_once()
        sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
