"""Full-run guard: reserve maximum cost; settle successful token usage only.

Failed requests or missing usage keep the entire maximum reservation. An
interrupted process cannot lose the reservation saved before sending.
"""
import fcntl
import json
import os
from pathlib import Path
import time
import budget_guard as guard


def settle(ledger, reserved, payload):
    usage = payload.get('usage') or {}
    prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
    if (not str(payload.get('model', '')).startswith('gpt-4o-mini')
            or type(prompt) is not int or type(completion) is not int
            or prompt < 0 or completion < 0):
        return  # Keep maximum reservation when usage cannot be verified.
    amount = prompt * guard.INPUT_RATE + completion * guard.OUTPUT_RATE
    if amount > reserved:
        raise guard.BudgetExceeded('Usage exceeded conservative reservation; stop and investigate')
    ledger = Path(ledger)
    with open(str(ledger) + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(ledger.read_text())
        if state['reserved_nanodollars'] < reserved:
            raise guard.BudgetExceeded('Ledger inconsistency; refusing to settle')
        state['reserved_nanodollars'] -= reserved - amount
        state['successful_usage_nanodollars'] = state.get('successful_usage_nanodollars', 0) + amount
        state['settled_requests'] = state.get('settled_requests', 0) + 1
        state['updated_unix'] = time.time()
        temporary = ledger.with_name(ledger.name + '.tmp')
        with temporary.open('w') as stream:
            json.dump(state, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, ledger)


def install(ledger):
    import httpx
    if getattr(httpx.Client.send, '_d2r_guard', False):
        raise RuntimeError('A spending guard is already installed')
    original, original_async = httpx.Client.send, httpx.AsyncClient.send

    def finish(request, response):
        reserved = request.extensions.get('d2r_reservation')
        if reserved is not None and response.is_success:
            try:
                payload = response.json()
            except (ValueError, UnicodeError):
                # Let the SDK's bounded connection-error retries handle an
                # empty/malformed successful response. Keep its reservation;
                # the next attempt must reserve separately before sending.
                raise httpx.RemoteProtocolError(
                    'OpenAI returned a successful response with invalid JSON',
                    request=request,
                )
            if isinstance(payload, dict):
                settle(ledger, reserved, payload)

    def send(client, request, *args, **kwargs):
        request = guard.guard_request(request, ledger) or request
        kwargs['follow_redirects'] = False
        response = original(client, request, *args, **kwargs)
        if 'd2r_reservation' in request.extensions:
            response.read()
            finish(request, response)
        return response

    async def asend(client, request, *args, **kwargs):
        request = guard.guard_request(request, ledger) or request
        kwargs['follow_redirects'] = False
        response = await original_async(client, request, *args, **kwargs)
        if 'd2r_reservation' in request.extensions:
            await response.aread()
            finish(request, response)
        return response

    send._d2r_guard = True
    httpx.Client.send, httpx.AsyncClient.send = send, asend
