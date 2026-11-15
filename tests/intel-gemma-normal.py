"""Explicit real-model Gemma target-only regression; never part of CPU CTest.

Use the unchanged pinned Google QAT Q4_0 target and an app-local Intel runtime:
  python tests/intel-gemma-normal.py --model MODEL.gguf --exe NATIVE.exe \
      --runtime-directory RUNTIME --output NEW_OUTPUT_DIRECTORY
The three normal-thinking screens are separate from canonical benchmarks.
Full SHA verification precedes loading and can warm the Windows file cache.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
import pathlib
import queue
import re
import subprocess
import sys
import threading
import time

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
C = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C))
from serve.gguf_reader import GGUFFile
from serve.tokenizer import Tokenizer
from serve.frontend import ChatTemplate, OutputParser, gemma_model_turn
from serve.tinyconfig import GEMMA4_SAMPLING
import psutil


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=pathlib.Path, required=True)
    ap.add_argument('--model', type=pathlib.Path, required=True)
    ap.add_argument('--exe', type=pathlib.Path, required=True)
    ap.add_argument('--runtime-directory', type=pathlib.Path, required=True)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--cases', default='short,4k,16k')
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    model = args.model.resolve()
    exe = args.exe.resolve()
    stage = args.runtime_directory.resolve()
    if os.name != 'nt':
        raise RuntimeError('This explicit native backend regression requires Windows')
    if not model.is_file() or not exe.is_file() or not (stage/'sycl9.dll').is_file():
        raise RuntimeError('Model, native executable or staged Intel runtime is missing')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    pin = kernel.CreateFileW(str(model),0x80000000,1,None,3,0x80,None)
    if pin == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        stat_before = [model.stat().st_size, model.stat().st_mtime_ns, model.stat().st_ino]
        if stat_before[0] != 14439363584:
            raise RuntimeError('Pinned model size differs')
        digest = hashlib.sha256()
        print('Verifying pinned Gemma target SHA-256 before loading', flush=True)
        with model.open('rb') as source:
            for block in iter(lambda: source.read(4*1024*1024), b''):
                digest.update(block)
        actual_model_sha256 = digest.hexdigest()
        if actual_model_sha256 != '3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d':
            raise RuntimeError('Pinned Gemma target SHA-256 differs')
        md = GGUFFile(model).metadata
        template_path = out/'chat_template.jinja'
        template_path.write_text(md['tokenizer.chat_template'], encoding='utf-8')
        tokenizer = Tokenizer.from_gguf(model)
        template = ChatTemplate(template_path)
        assert template.call_format == 'gemma'
        special_markers={m:tokenizer.encode(m,parse_special=True)==[tokenizer.ids[m]] for m in ('<bos>','<|turn>','<turn|>','<|channel>','<channel|>')}
        assert all(special_markers.values()),'Gemma control tokens must retain the native API tokenizer IDs'
        terminal_ids={tokenizer.ids['<turn|>']}
        terminal_ids.update(value for key,value in tokenizer.special_ids.items() if key.endswith(('eos_token_id','eot_token_id','eom_token_id')))
        expected = {'id':'R17','participants':12,'fee':800,'venue':3200,'start':'14:30','remaining':6400}
        short = ('Reservation R17 has12 participants. Each pays800 yen. The venue costs3200 yen. '
                 'Start time is14:30. Calculate money remaining after paying the venue. '
                 'Return one JSON object with exactly id, participants, fee, venue, start, remaining. '
                 'Use the stated facts and give a final answer.')
        prefix = ('Read the archive and use the FINAL CORRECTION at the end. Earlier entries are superseded.\n'
                  'DRAFT: Reservation R17 had10 participants, fee700 yen, venue2500 yen, start15:00.\n')
        suffix = ('\nFINAL CORRECTION: Reservation R17 now has12 participants, fee800 yen each, '
                  'venue3200 yen, start14:30. The earlier draft is cancelled. '
                  'Calculate remaining money after paying the venue. Return one JSON object with exactly '
                  'id, participants, fee, venue, start, remaining, using only the final correction.')

        def render(user):
            text = template.render([{'role':'user','content':user}], **tokenizer.template_tokens, enable_thinking=True)
            text = gemma_model_turn(text, True)
            return text, tokenizer.encode(text,parse_special=True)

        inputs = []
        for case in args.cases.split(','):
            if case == 'short':
                user = short
            else:
                target = {'4k':4096,'16k':16384}[case]
                lo, hi = 0, target
                while lo < hi:
                    mid = (lo+hi+1)//2
                    _, ids = render(prefix+' reference'*mid+suffix)
                    if len(ids) <= target:
                        lo = mid
                    else:
                        hi = mid-1
                user = prefix+' reference'*lo+suffix
            prompt, ids = render(user)
            inputs.append({'case':case,'prompt':prompt,'ids':ids,'tokens':len(ids),
                           'ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                           'thought_opened':prompt.endswith('<|channel>thought\n')})
            save(out/(case+'-request.json'), {'messages':[{'role':'user','content':user}], 'prompt_tokens':len(ids), 'thinking':True,
                'normal_sampling':GEMMA4_SAMPLING,'seed':2026,'max_new_tokens':4096,'expected_final_json':expected,'ids':ids})
        result = {'passed':False,'preparation_only':args.prepare_only,'scope':'Private normal-thinking target-only Gemma screen, separate from canonical fixed-protocol benchmarks.',
                  'native_executable_sha256':sha(exe),'model':{'file':model.name,'bytes':stat_before[0],
                      'sha256':actual_model_sha256,
                      'sha256_recomputed_this_run':True,'read_only_model_handle_held':True},
                  'template_sha256':sha(template_path),'special_markers_single_token_verified':special_markers,'parse_special':True,'normal_sampling':GEMMA4_SAMPLING,'seed':2026,
                  'context':32768,'chunk':512,'actual_kv':'int8','mtp':0,'all_selected_experts':8,'max_new_tokens':4096,
                  'records':[]}
        save(out/'result.json',result)
        print('Prepared Gemma inputs', [(x['case'],x['tokens'],x['thought_opened']) for x in inputs], flush=True)
        if args.prepare_only:
            return
        env = dict(os.environ)
        for key in list(env):
            if key.startswith(('STRATA_','TINYSTRATA_')):
                env.pop(key)
        env['PATH']=str(stage)+';'+os.environ['SystemRoot']+'\\System32;'+os.environ['SystemRoot']
        env['ONEAPI_DEVICE_SELECTOR']='level_zero:gpu'
        env['UR_LOADER_PRELOAD_FILTER']='1'
        command=[str(exe),'--model',str(model),'--serve','--max-context','32768','--chunk','512','--kv-type','int8','--mtp','0','--prefill-mkl']
        result['command']=command
        stdout_log = (out/'native.stdout.log').open('w',encoding='utf-8')
        stderr_log = (out/'native.stderr.log').open('wb')
        started = time.perf_counter()
        native = subprocess.Popen(command,env=env,cwd=C,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr_log,text=True,encoding='utf-8',bufsize=1)
        result['native_process']={'pid':native.pid,'created':psutil.Process(native.pid).create_time(),'executable':str(exe)}
        save(out/'result.json',result)
        lines = queue.Queue()

        def reader():
            for line in native.stdout:
                stdout_log.write(line);stdout_log.flush();lines.put((time.perf_counter(),line.rstrip()))
            lines.put((time.perf_counter(),None))

        worker = threading.Thread(target=reader,daemon=True)
        worker.start()
        def receive():
            while True:
                control=out/'control.txt'
                if control.exists():
                    raw_control=control.read_bytes()
                    action=raw_control.decode('utf-16' if raw_control.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig').strip()
                    if action not in ('STOP','STOP_QUIT'):raise RuntimeError('Unknown external native control')
                    native.stdin.write('STOP\n'+('QUIT\n' if action=='STOP_QUIT' else ''));native.stdin.flush()
                    control.rename(out/'control.handled.txt')
                    save(out/'external-control-receipt.json',{'action':action,'time_unix':time.time(),'native_pid':native.pid,'through_owned_stdin':True,'forced':False})
                try:
                    timestamp,line=lines.get(timeout=1)
                except queue.Empty:
                    if native.poll() is not None:
                        raise RuntimeError('Native exited without protocol completion')
                    continue
                if line is None:
                    raise RuntimeError('Native stdout closed before expected protocol completion')
                if line.startswith('ERROR '):
                    raise RuntimeError(line)
                return timestamp,line
        try:
            while True:
                _,line=receive()
                if line.startswith('INFO '):
                    result['native_info']=line
                if line.startswith('weights '):
                    print(line,flush=True)
                if line.startswith('READY '):
                    break
            result['load_seconds']=time.perf_counter()-started
            paths=[m.path for m in psutil.Process(native.pid).memory_maps()]
            sdk=[p for p in paths if '\\intel\\oneapi\\' in p.lower()]
            result['runtime_module_isolation']={'compiler_sdk_dll_count':len(sdk),'loaded_paths':paths}
            if sdk:
                raise RuntimeError('Unexpected compiler-SDK runtime DLL')
            save(out/'result.json',result)
            print('Gemma model loaded seconds',round(result['load_seconds'],3),'SDK DLLs',len(sdk),flush=True)
            sampling_names={'repetition_penalty':'penalty_repeat','frequency_penalty':'penalty_freq','presence_penalty':'penalty_present'}
            fields=' '.join(f'{sampling_names.get(k,k)}={v}' for k,v in GEMMA4_SAMPLING.items())+' seed=2026 ckpt=0'
            for request in inputs:
                sent=time.perf_counter()
                native.stdin.write('GEN 4096 '+fields+' '+','.join(map(str,request['ids']))+'\n')
                native.stdin.flush()
                output_ids=[];first=None;last_feedback=sent;first_terminal=None;overrun_stop=False
                while True:
                    timestamp,line=receive()
                    if line.startswith('PP '):
                        _,read,total,ms=line.split();pp=1000*int(read)/float(ms)
                        print(request['case'],f'PP{read}/{total} {pp:.2f}tok/s',flush=True)
                    if line.startswith('T '):
                        output_ids.append(int(line.split()[1]))
                        if first_terminal is not None and not overrun_stop:
                            native.stdin.write('STOP\nQUIT\n');native.stdin.flush();overrun_stop=True
                            save(out/'terminal-overrun-control.json',{'case':request['case'],'first_terminal_position':first_terminal,'observed_output':len(output_ids),'action':'STOP then QUIT through owned stdin','forced':False})
                        if first_terminal is None and output_ids[-1] in terminal_ids:first_terminal=len(output_ids)
                        if first is None:first=timestamp
                        if timestamp-last_feedback>=10:
                            rate=(len(output_ids)-1)/max(timestamp-first,1e-9)
                            print(request['case'],'generated',len(output_ids),f'{rate:.2f}tok/s',flush=True)
                            last_feedback=timestamp
                    if line.startswith('DONE '):
                        done=line.split();break
                raw=tokenizer.decode(output_ids)
                parser=OutputParser(thinking=True,call_format=template.call_format,opened=request['thought_opened'])
                events=parser.feed(raw)
                state_before_finish=parser.state
                events+=parser.finish('stop' if done[5]=='stop' else 'length')
                content=''.join(e.text for e in events if e.kind=='content')
                reasoning=''.join(e.text for e in events if e.kind=='reasoning')
                facts=False;final_json=None
                for match in re.finditer(r'\{',content):
                    try:
                        obj,_=json.JSONDecoder().raw_decode(content[match.start():])
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj,dict) and all(obj.get(k)==v for k,v in expected.items()):
                        facts=True;final_json=obj;break
                n=int(done[1]);pp_ms=float(done[3]);tg_ms=float(done[4])
                record={'case':request['case'],'prompt_tokens':request['tokens'],'output_tokens':n,'output_ids':output_ids,
                    'finish':done[5],'native_done':line,'prefill_ms':pp_ms,'decode_ms':tg_ms,
                    'prefill_tok_s':request['tokens']*1000/pp_ms,'decode_steps_tok_s':(n-1)*1000/tg_ms if n>1 else 0,
                    'first_token_seconds':first-sent if first else None,'wall_seconds':timestamp-sent,
                    'reused_tokens':int(done[8]),'raw':raw,'content':content,'reasoning':reasoning,
                    'parser_state_before_finish':state_before_finish,'thinking_closed':state_before_finish!='reasoning',
                    'six_facts_correct':facts,'final_json':final_json,
                    'first_terminal_output_position':first_terminal,'terminal_ids':sorted(terminal_ids),
                    'terminal_is_last_output':first_terminal==n,'terminal_overrun_stop_sent':overrun_stop,
                    'passed':done[5]=='stop' and facts and state_before_finish!='reasoning' and first_terminal==n and not overrun_stop}
                result['records'].append(record);save(out/'result.json',result)
                print(request['case'],'PASS' if record['passed'] else 'FAIL',f'PP{record["prefill_tok_s"]:.2f}/TG{record["decode_steps_tok_s"]:.2f}tok/s','output',n,'finish',done[5],flush=True)
                current_paths=[m.path for m in psutil.Process(native.pid).memory_maps()]
                sdk=[p for p in current_paths if '\\intel\\oneapi\\' in p.lower()]
                record['compiler_sdk_dll_count_after_answer']=len(sdk)
                if sdk:raise RuntimeError('Compiler-SDK DLL appeared during generation')
                if not record['passed']:
                    native.stdin.write('QUIT\n');native.stdin.flush()
                    result['native_exit_code']=native.wait()
                    save(out/'result.json',result)
                    raise RuntimeError('Completed Gemma answer failed; longer cases await diagnosis')
            native.stdin.write('QUIT\n');native.stdin.flush();native.stdin.close()
            code=native.wait(timeout=60);worker.join(timeout=5)
            result['native_exit_code']=code
            result['model_identity_unchanged']=stat_before==[model.stat().st_size,model.stat().st_mtime_ns,model.stat().st_ino]
            result['exe_unchanged']=sha(exe)==result['native_executable_sha256']
            result['passed']=code==0 and result['model_identity_unchanged'] and result['exe_unchanged'] and all(r['passed'] for r in result['records'])
            save(out/'result.json',result)
            if not result['passed']:
                raise RuntimeError('Normal Gemma screen did not pass')
        except Exception as error:
            result['error']=repr(error)
            save(out/'result.json',result)
            raise
        finally:
            if native.poll() is None:
                # On a protocol failure request ordinary shutdown. A normally
                # generating request is never stopped for a wall-time budget.
                try:native.stdin.write('QUIT\n');native.stdin.flush()
                except (OSError,ValueError):pass
            native.wait()
            result['native_exit_code']=native.poll()
            result['native_process_terminated']=native.poll() is not None
            result['model_identity_unchanged']=stat_before==[model.stat().st_size,model.stat().st_mtime_ns,model.stat().st_ino]
            save(out/'result.json',result)
            stdout_log.close();stderr_log.close()
    finally:
        kernel.CloseHandle(pin)


if __name__=='__main__':
    main()
