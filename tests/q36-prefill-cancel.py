"""Parent-run protocol regression; launches an engine only when explicitly invoked.

python tests/q36-prefill-cancel.py --command-json qwen-command.json --chunk 512 --output qwen-cancel-result.json
The JSON contains the complete argv (exe, --model, --serve, --kv-type, --mtp, etc.).
Optional: --image-pad ID --image-width 2048; the engine must already have --vision on.
Synthetic image embeddings test cache boundaries, not image understanding/quality.
"""
from __future__ import annotations
import argparse, json, os, queue, struct, subprocess, threading, time
from pathlib import Path


def require(value, message):
    if not value:
        raise AssertionError(message)


class Engine:
    def __init__(self, command, output, timeout):
        self.timeout = timeout
        self.stderr = output.with_suffix('.engine.log').open('w', encoding='utf-8')
        options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                                     text=True, encoding='utf-8', errors='replace', bufsize=1, **options)
        self.lines = queue.Queue()
        def reader():
            for line in self.proc.stdout:
                self.lines.put(line.rstrip('\r\n'))
            self.lines.put(None)
        threading.Thread(target=reader, daemon=True).start()
        while True:
            line = self.line(timeout)
            if line.startswith('READY '):
                break
        self.pid = self.proc.pid

    def line(self, timeout):
        line = self.lines.get(timeout=max(0.01, timeout))
        require(line is not None, f'engine exited: {self.proc.poll()}')
        require(not line.startswith('ERR'), line)
        return line

    def send(self, text):
        require(self.proc.poll() is None and self.proc.pid == self.pid, 'engine process changed or exited')
        self.proc.stdin.write(text + '\n')
        self.proc.stdin.flush()

    def request(self, ids, *, turn=0, cancel_after=None, immediate=False, image=None, max_new=1):
        keys = 'temperature=0 penalty_repeat=1 penalty_freq=0 penalty_present=0'
        if turn:
            keys += f' turn_at={turn}'
        head = 'GENI' if image else 'GEN'
        file = str(image) + ' ' if image else ''
        command = f'{head} {max_new} {keys} {file}' + ','.join(map(str, ids))
        self.send(command + ('\nSTOP' if immediate else ''))
        begin = time.monotonic(); stopped_at = begin if immediate else None
        lines, tokens, resume = [], [], 0
        while True:
            now = time.monotonic()
            deadline = (stopped_at + 60) if stopped_at is not None else (begin + self.timeout)
            line = self.line(deadline - now)
            lines.append(line)
            if line.startswith('RESUME '):
                resume = int(line.split()[1])
            if line.startswith('PP ') and cancel_after is not None and stopped_at is None:
                if int(line.split()[1]) >= cancel_after:
                    self.send('STOP'); stopped_at = time.monotonic()
            if line.startswith('T '):
                tokens.append(int(line.split()[1]))
            if line.startswith('DONE '):
                fields = line.split()
                result = {'generated': int(fields[1]), 'total': int(fields[2]), 'finish': fields[5],
                          'drafts_accepted': int(fields[6]), 'drafts_offered': int(fields[7]),
                          'reused': int(fields[8]), 'read': int(fields[14]), 'resume': resume,
                          'tokens': tokens, 'lines': lines, 'pid': self.proc.pid,
                          'stop_to_done_s': time.monotonic() - stopped_at if stopped_at is not None else None}
                require(self.proc.poll() is None and self.proc.pid == self.pid, 'engine restarted/exited')
                return result

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.terminate(); self.proc.wait(timeout=30)
        self.stderr.close()


def run(args):
    command = json.loads(args.command_json.read_text(encoding='utf-8-sig'))
    require(isinstance(command, list) and all(isinstance(v, str) for v in command), 'argv JSON must be an array of strings')
    args.output.parent.mkdir(exist_ok=True, parents=True)
    engine = Engine(command, args.output, args.timeout)
    results = {'command': command, 'pid': engine.pid, 'cases': {}}
    try:
        total = max(args.prompt_tokens, args.chunk * 10)
        turn = args.chunk * 3
        def prompt(marker, length=total):
            return [marker] + [args.filler] * (length - 1)
        def cancelled(name, result, positive=True):
            require(result['finish'] == 'cancel' and result['generated'] == 0 and not result['tokens'], name + ': cancellation emitted output')
            require(0 <= result['read'] < result['total'], name + ': prompt was not stopped early')
            if positive:
                require(result['read'] > 0, name + ': no partial prefix was exercised')
            results['cases'][name] = result
        first = prompt(args.filler + 1)
        stopped = engine.request(first, turn=turn, cancel_after=args.chunk)
        cancelled('before_turn_checkpoint', stopped)
        retry = engine.request(first, turn=turn, max_new=16)
        require(retry['finish'] != 'cancel' and 1 <= retry['generated'] <= 16, 'retry failed')
        require(retry['resume'] == stopped['read'] == retry['reused'], 'partial prefix was not reused exactly')
        results['cases']['matching_retry'] = retry
        reset = engine.request(prompt(args.filler + 100, 16))
        require(reset['reused'] == 0, 'reset prompt unexpectedly reused prior state')
        cold = engine.request(first, turn=turn, max_new=16)
        require(cold['reused'] == cold['resume'] == 0, 'cold comparison reused old state')
        require(retry['tokens'] == cold['tokens'], 'matching retry token IDs differ from the cold request')
        results['cases']['matching_retry_cold'] = cold

        second = prompt(args.filler + 2)
        stopped = engine.request(second, turn=turn, cancel_after=turn + args.chunk)
        cancelled('after_turn_checkpoint', stopped)
        require(stopped['read'] > turn, 'saved turn was not exercised')
        edited = second[:]; edited[turn] = args.filler + 3
        restored = engine.request(edited, turn=turn, max_new=16)
        require(restored['resume'] == restored['reused'] == turn, 'existing turn checkpoint was not restored')
        require(restored['finish'] != 'cancel' and 1 <= restored['generated'] <= 16, 'turn restoration failed')
        results['cases']['edited_suffix_restores_turn'] = restored
        reset = engine.request(prompt(args.filler + 101, 16))
        require(reset['reused'] == 0, 'reset prompt unexpectedly reused prior state')
        cold = engine.request(edited, turn=turn, max_new=16)
        require(cold['reused'] == cold['resume'] == 0, 'cold comparison reused old state')
        require(restored['tokens'] == cold['tokens'], 'restored turn token IDs differ from the cold request')
        results['cases']['edited_suffix_cold'] = cold

        unrelated = engine.request(prompt(args.filler + 4, args.chunk * 2))
        require(unrelated['resume'] == unrelated['reused'] == 0, 'unrelated prefix reused stale state')
        require(unrelated['generated'] == 1, 'unrelated request failed')
        results['cases']['unrelated_prompt'] = unrelated

        queued = prompt(args.filler + 5)
        quick = engine.request(queued, immediate=True)
        cancelled('queued_stop', quick, positive=False)
        next_request = engine.request(queued)
        require(next_request['finish'] != 'cancel' and next_request['generated'] == 1, 'STOP leaked into next request')
        results['cases']['next_request_after_queued_stop'] = next_request

        if args.image_pad is not None:
            image_rows = args.chunk * 3
            image = args.output.with_suffix('.sve1')
            # A partial first chunk necessarily falls within the image span.
            image.write_bytes(struct.pack('<5i', 0x31455653, image_rows, image_rows, 1, args.image_width)
                              + bytes(image_rows * args.image_width * 4))
            ids = [args.filler + 6] + [args.image_pad] * image_rows + [args.filler] * (args.chunk * 4)
            stopped = engine.request(ids, cancel_after=args.chunk, image=image)
            cancelled('inside_image', stopped)
            require(1 < stopped['read'] < image_rows + 1, 'image split boundary was not exercised')
            retry = engine.request(ids, image=image)
            require(retry['resume'] == retry['reused'] == 0, 'partial image was reused')
            require(retry['generated'] == 1 and retry['finish'] != 'cancel', 'whole-image retry failed')
            results['cases']['image_retry'] = retry
        results['passed'] = True
        args.output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'passed': True, 'cases': len(results['cases']), 'pid': engine.pid, 'output': str(args.output)}))
    finally:
        engine.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--command-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk', type=int, default=512)
    parser.add_argument('--prompt-tokens', type=int, default=8192)
    parser.add_argument('--filler', type=int, default=198)
    parser.add_argument('--timeout', type=float, default=1200)
    parser.add_argument('--image-pad', type=int)
    parser.add_argument('--image-width', type=int, default=2048)
    run(parser.parse_args())
