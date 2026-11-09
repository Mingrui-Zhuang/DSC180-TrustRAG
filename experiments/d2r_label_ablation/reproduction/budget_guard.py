"""Conservative, persistent spending guard for the smoke runner's HTTPX calls.

Reservations are never refunded, including on failure. This overestimates spend
and charges retries to the same allowance. Not an account-wide billing limit.
"""
from contextvars import ContextVar

import fcntl
import json
import os
from pathlib import Path
import time

# USD nanodollars per token, verified against official GPT-4o-mini pricing.
INPUT_RATE = 150
OUTPUT_RATE = 600
LIMIT = 4_000_000_000  # $4 cumulative reservation allowance
MAX_REQUESTS = 40
MAX_OUTPUT = 512
MAX_INPUT_BYTES = 32_768
MAX_COMPLETIONS = 1


class BudgetExceeded(RuntimeError):
    pass


def reserve(path, amount):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock on a separate inode so atomic ledger replacement is safe.
    with open(str(path) + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            state = json.loads(path.read_text())
        else:
            state = {'reserved_nanodollars': 0, 'requests': 0}
        if state.get('limit_nanodollars', LIMIT) != LIMIT:
            raise BudgetExceeded('Stored allowance differs; refusing to change it')
        used, count = state['reserved_nanodollars'], state['requests']
        if type(used) is not int or type(count) is not int or used < 0 or count < 0:
            raise BudgetExceeded('Invalid budget ledger; refusing requests')
        if count >= MAX_REQUESTS or used + amount > LIMIT:
            raise BudgetExceeded('Smoke budget or request limit reached; refusing request')
        state.update(reserved_nanodollars=used + amount, requests=count + 1,
                     updated_unix=time.time(), limit_nanodollars=LIMIT)
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('w') as output:
            json.dump(state, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)


def guard_request(request, ledger):
    host = request.url.host
    key = os.environ.get('OPENAI_API_KEY', '')
    authorization = request.headers.get('authorization', '')
    if host != 'api.openai.com':
        if key and authorization == 'Bearer ' + key:
            raise BudgetExceeded('API-key destination is not permitted')
        return  # Model/data downloads are not OpenAI inference calls.
    if (request.url.scheme != 'https' or request.method != 'POST'
            or request.url.path != '/v1/chat/completions'):
        raise BudgetExceeded('Only HTTPS OpenAI chat completions are permitted')
    body = json.loads(request.content)
    permitted = {'model', 'messages', 'temperature', 'top_p', 'max_tokens',
                 'max_completion_tokens', 'stream', 'n', 'stop', 'seed',
                 'presence_penalty', 'frequency_penalty', 'user', 'response_format'}
    if set(body) - permitted:
        raise BudgetExceeded('Unsupported request options; cannot bound cost')
    if body.get('model') != 'gpt-4o-mini':
        raise BudgetExceeded('Only gpt-4o-mini is permitted')
    completions = body.get('n', 1)
    if (body.get('stream', False) is not False or type(completions) is not int
            or not 1 <= completions <= MAX_COMPLETIONS):
        raise BudgetExceeded('Streaming or completion count is not permitted')
    messages = body.get('messages')
    if not isinstance(messages, list) or not 1 <= len(messages) <= 32:
        raise BudgetExceeded('Expected 1 to 32 text messages')
    for message in messages:
        if (not isinstance(message, dict)
                or set(message) - {'role', 'content', 'name'}
                or message.get('role') not in {'system', 'user', 'assistant'}
                or not isinstance(message.get('content'), str)):
            raise BudgetExceeded('Only plain text chat messages are supported')
    for field in ('max_tokens', 'max_completion_tokens'):
        if field in body and (type(body[field]) is not int or body[field] <= 0):
            raise BudgetExceeded('Invalid output token limit')
    output_limit = min(body.get('max_tokens', MAX_OUTPUT),
                       body.get('max_completion_tokens', MAX_OUTPUT), MAX_OUTPUT)
    body.pop('max_completion_tokens', None)
    body['max_tokens'] = output_limit
    encoded = json.dumps(body, ensure_ascii=False).encode('utf-8')
    if len(encoded) > MAX_INPUT_BYTES:
        raise BudgetExceeded('Input exceeds smoke request size limit')
    # UTF-8 bytes upper-bound byte-BPE text tokens; add 1024 for message framing
    # (at most 32 messages). Include the entire body to stay conservative.
    input_bound = len(encoded) + 1024
    reservation = input_bound * INPUT_RATE + output_limit * OUTPUT_RATE * completions
    reserve(ledger, reservation)
    # Rebuild the request because both content and Content-Length must change.
    import httpx
    headers = dict(request.headers)
    headers.pop('content-length', None)
    extensions = dict(request.extensions, d2r_reservation=reservation)
    return httpx.Request(request.method, request.url, headers=headers,
                         content=encoded, extensions=extensions)


chat_case = ContextVar("d2r_chat_case", default=None)


def install(ledger, *, chat_log=None, case_phase="draft_case"):
    import httpx
    if getattr(httpx.Client.send, '_d2r_guard', False):
        raise RuntimeError('Budget guard is already installed')
    sync_send, async_send = httpx.Client.send, httpx.AsyncClient.send

    def should_log(request):
        return chat_log is not None and "d2r_reservation" in request.extensions

    def record(request, response):
        case_id = chat_case.get()
        row = {
            "case_id": case_id,
            "phase": "setup" if case_id is None else case_phase,
            "request": json.loads(request.content),
            "status_code": response.status_code,
        }
        if response.is_success:
            try:
                row["response"] = response.json()
            except ValueError:
                row["response"] = response.text
        with chat_log.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def send(client, request, *args, **kwargs):
        request = guard_request(request, ledger) or request
        kwargs['follow_redirects'] = False
        response = sync_send(client, request, *args, **kwargs)
        if should_log(request):
            if response.is_success:
                response.read()
            record(request, response)
        return response

    async def asend(client, request, *args, **kwargs):
        request = guard_request(request, ledger) or request
        kwargs['follow_redirects'] = False
        response = await async_send(client, request, *args, **kwargs)
        if should_log(request):
            if response.is_success:
                await response.aread()
            record(request, response)
        return response

    send._d2r_guard = True
    httpx.Client.send = send
    httpx.AsyncClient.send = asend
