import asyncio
import json
from pathlib import Path
import tempfile
import unittest
import httpx
import budget_guard as guard
import metered_guard


class MeteredTests(unittest.TestCase):
    def test_malformed_success_keeps_reservation_sync_and_async(self):
        original, original_async = httpx.Client.send, httpx.AsyncClient.send
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / 'budget.json'
            def request():
                return httpx.Request('POST', 'https://api.openai.com/v1/chat/completions',
                    json={'model': 'gpt-4o-mini', 'messages': [{'role': 'user', 'content': 'hi'}]})
            try:
                metered_guard.install(ledger)
                transport = httpx.MockTransport(lambda req: httpx.Response(200, text=''))
                with httpx.Client(transport=transport) as client:
                    with self.assertRaises(httpx.RemoteProtocolError):
                        client.send(request())
                first = json.loads(ledger.read_text())['reserved_nanodollars']
                async def run():
                    async with httpx.AsyncClient(transport=transport) as client:
                        with self.assertRaises(httpx.RemoteProtocolError):
                            await client.send(request())
                asyncio.run(run())
                state = json.loads(ledger.read_text())
                self.assertEqual(state['reserved_nanodollars'], 2 * first)
                self.assertEqual(state['requests'], 2)
                self.assertEqual(state.get('settled_requests', 0), 0)
            finally:
                httpx.Client.send, httpx.AsyncClient.send = original, original_async

    def test_multiple_completions_and_changed_allowance(self):
        previous = guard.MAX_COMPLETIONS
        try:
            guard.MAX_COMPLETIONS = 3
            with tempfile.TemporaryDirectory() as directory:
                ledger = Path(directory) / 'budget.json'
                request = httpx.Request('POST', 'https://api.openai.com/v1/chat/completions',
                                        json={'model': 'gpt-4o-mini', 'n': 3,
                                              'messages': [{'role': 'user', 'content': 'hello'}]})
                guarded = guard.guard_request(request, ledger)
                reserved = guarded.extensions['d2r_reservation']
                self.assertGreater(reserved, 3 * guard.MAX_OUTPUT * guard.OUTPUT_RATE)
                state = json.loads(ledger.read_text())
                state['limit_nanodollars'] = guard.LIMIT + 1
                ledger.write_text(json.dumps(state))
                with self.assertRaises(guard.BudgetExceeded):
                    guard.guard_request(request, ledger)
        finally:
            guard.MAX_COMPLETIONS = previous

    def test_success_settlement_failures_and_async(self):
        original, original_async = httpx.Client.send, httpx.AsyncClient.send
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / 'budget.json'
            def request():
                return httpx.Request('POST', 'https://api.openai.com/v1/chat/completions',
                                     json={'model': 'gpt-4o-mini', 'messages': [{'role': 'user', 'content': 'hello'}]})
            def success(req):
                return httpx.Response(200, json={'model': 'gpt-4o-mini-2024-07-18',
                                                 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}})
            try:
                metered_guard.install(ledger)
                with httpx.Client(transport=httpx.MockTransport(success)) as client:
                    client.send(request())
                one_cost = 10 * guard.INPUT_RATE + 5 * guard.OUTPUT_RATE
                self.assertEqual(json.loads(ledger.read_text())['reserved_nanodollars'], one_cost)
                async def run():
                    async with httpx.AsyncClient(transport=httpx.MockTransport(success)) as client:
                        await client.send(request())
                asyncio.run(run())
                self.assertEqual(json.loads(ledger.read_text())['reserved_nanodollars'], 2 * one_cost)
                with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(500))) as client:
                    client.send(request())
                state = json.loads(ledger.read_text())
                self.assertGreater(state['reserved_nanodollars'], 2 * one_cost)
                self.assertEqual(state['requests'], 3)
                self.assertEqual(state['settled_requests'], 2)
                state['reserved_nanodollars'] = guard.LIMIT
                ledger.write_text(json.dumps(state))
                with httpx.Client(transport=httpx.MockTransport(success)) as client:
                    with self.assertRaises(guard.BudgetExceeded):
                        client.send(request())
            finally:
                httpx.Client.send, httpx.AsyncClient.send = original, original_async


if __name__ == '__main__':
    unittest.main()
