"""Meaningful spending-control checks. All HTTP responses are simulated."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
import httpx
import budget_guard as guard


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ledger = Path(self.temp.name) / 'budget.json'

    def tearDown(self):
        self.temp.cleanup()

    def request(self, **overrides):
        body = {'model': 'gpt-4o-mini', 'messages': [{'role': 'user', 'content': 'Hello'}]}
        body.update(overrides)
        return httpx.Request('POST', 'https://api.openai.com/v1/chat/completions', json=body)

    def test_output_cap_and_persistence(self):
        request = guard.guard_request(self.request(max_tokens=10000), self.ledger)
        self.assertEqual(json.loads(request.content)['max_tokens'], 512)
        first = json.loads(self.ledger.read_text())['reserved_nanodollars']
        guard.guard_request(self.request(), self.ledger)
        state = json.loads(self.ledger.read_text())
        self.assertGreater(state['reserved_nanodollars'], first)
        self.assertEqual(state['requests'], 2)

    def test_reject_unbounded_requests(self):
        for options in ({'model': 'gpt-4o'}, {'stream': True}, {'n': 2},
                        {'tools': []}, {'max_tokens': -1},
                        {'messages': [{'role': 'user', 'content': 'x' * 40000}]}):
            with self.subTest(options=list(options)):
                with self.assertRaises(guard.BudgetExceeded):
                    guard.guard_request(self.request(**options), self.ledger)
        self.assertFalse(self.ledger.exists())

    def test_fail_closed_on_limit_or_corrupt_ledger(self):
        self.ledger.write_text(json.dumps({'reserved_nanodollars': guard.LIMIT, 'requests': 1}))
        with self.assertRaises(guard.BudgetExceeded):
            guard.guard_request(self.request(), self.ledger)
        self.ledger.write_text('corrupt')
        with self.assertRaises(ValueError):
            guard.guard_request(self.request(), self.ledger)

    def test_sync_async_and_failed_requests_are_reserved(self):
        original, original_async = httpx.Client.send, httpx.AsyncClient.send
        calls = []

        def response(request):
            calls.append(request)
            return httpx.Response(500, json={'error': 'simulated'})

        try:
            guard.install(self.ledger)
            with httpx.Client(transport=httpx.MockTransport(response)) as client:
                self.assertEqual(client.send(self.request()).status_code, 500)

            async def run():
                async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
                    await client.send(self.request())
            asyncio.run(run())
            self.assertEqual(json.loads(self.ledger.read_text())['requests'], 2)
            self.assertEqual(len(calls), 2)
            self.ledger.write_text(json.dumps({'reserved_nanodollars': 0, 'requests': guard.MAX_REQUESTS}))
            with httpx.Client(transport=httpx.MockTransport(response)) as client:
                with self.assertRaises(guard.BudgetExceeded):
                    client.send(self.request())
            self.assertEqual(len(calls), 2)
        finally:
            httpx.Client.send, httpx.AsyncClient.send = original, original_async


if __name__ == '__main__':
    unittest.main()
