"""Real Windows job ownership: a running MCP .cmd wrapper must not lose its early child.

The test host owns a private job; only its own children are ended. No installed app, GPU, or user process is touched.
Adapted from Strata 5d87c4d8eb9d536c663698fdc38a9fc90537090a (MIT). See LICENSE and NOTICE.
"""
import ctypes
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import winjob

ECHO = "import os,sys; print(sys.stdin.readline().upper().strip()); print(os.getcwd()); print(os.environ['TINY_JOB_TEST']); print('err', file=sys.stderr)"
WAIT = "import time; time.sleep(120)"
REAL_POPEN = subprocess.Popen
DIRECT_PYTHON = str(Path(sys.base_prefix) / "python.exe") if os.name == "nt" else sys.executable
FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def line_with_deadline(pipe, seconds=15):
    answer = queue.Queue()
    def read():
        try:
            answer.put(pipe.readline())
        except Exception as error:
            answer.put(error)
    threading.Thread(target=read, daemon=True).start()
    result = answer.get(timeout=seconds)
    if isinstance(result, Exception):
        raise result
    return result


@unittest.skipUnless(os.name == "nt", "Windows job objects")
class ContainedStartup(unittest.TestCase):
    def test_flags_pipes_unicode_cwd_and_env(self):
        seen = {}
        def record(args, **kwargs):
            seen.update(kwargs)
            return REAL_POPEN(args, **kwargs)
        with tempfile.TemporaryDirectory(prefix="Tiny-資料-") as tmp:
            env = {**os.environ, "TINY_JOB_TEST": "日本語"}
            with mock.patch.object(subprocess, "Popen", side_effect=record):
                proc = winjob.start_contained([sys.executable, "-X", "utf8", "-c", ECHO], cwd=tmp, env=env,
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                             text=True, encoding="utf-8", creationflags=FLAGS)
            out, err = proc.communicate("hello\n", timeout=15)
            self.assertEqual(proc.returncode, 0, err)
            self.assertEqual(out.splitlines(), ["HELLO", tmp, "日本語"])
            self.assertEqual(err.strip(), "err")
            self.assertEqual(seen["creationflags"], FLAGS | winjob.CREATE_SUSPENDED)
            self.assertIs(seen["env"], env)
            self.assertEqual([seen[k] for k in ("stdin", "stdout", "stderr")], [subprocess.PIPE] * 3)

    def test_refused_job_still_runs(self):
        with mock.patch.object(winjob, "contain", return_value=False):
            proc = winjob.start_contained([sys.executable, "-c", "print('resumed')"], stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, creationflags=FLAGS)
        out, err = proc.communicate(timeout=15)
        self.assertEqual((proc.returncode, out.strip()), (0, "resumed"), err)

    def failure_case(self, stage, failure):
        created = []
        def record(args, **kwargs):
            proc = REAL_POPEN(args, **kwargs)
            created.append(proc)
            return proc
        with mock.patch.object(subprocess, "Popen", side_effect=record), \
                mock.patch.object(winjob, stage, **failure):
            with self.assertRaises((OSError, RuntimeError)):
                winjob.start_contained([sys.executable, "-c", WAIT], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=FLAGS)
        self.assertEqual(len(created), 1)
        self.assertIsNotNone(created[0].poll(), "an unpublished suspended child survived")
        self.assertTrue(all(p.closed for p in (created[0].stdin, created[0].stdout, created[0].stderr)))

    def test_refused_resume_cleans_child_and_pipes(self):
        self.failure_case("_resume", {"return_value": False})

    def test_resume_exception_cleans_child_and_pipes(self):
        self.failure_case("_resume", {"side_effect": OSError("snapshot refused")})

    def test_assignment_exception_cleans_child_and_pipes(self):
        self.failure_case("contain", {"side_effect": RuntimeError("assignment failed")})

    def test_spawn_failure_is_reported(self):
        with self.assertRaises(OSError):
            winjob.start_contained([str(ROOT / "no-such-worker-0c3e64.exe")], creationflags=FLAGS)

    def test_nested_job_child_runs(self):
        script = ("import sys; sys.path.insert(0, " + repr(str(ROOT)) + "); from serve import winjob; "
                  "import subprocess; p=winjob.start_contained([sys.executable,'-c',\"print('nested')\"], "
                  "stdout=subprocess.PIPE,text=True,creationflags=" + str(FLAGS) + "); "
                  "out=p.communicate(timeout=15)[0]; print(out.strip(), p.returncode)")
        proc = winjob.start_contained([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, creationflags=FLAGS)
        out, err = proc.communicate(timeout=20)
        self.assertEqual((proc.returncode, out.strip()), (0, "nested 0"), err)

    def test_mcp_cmd_rpc_and_early_grandchild_end_when_host_dies(self):
        # A real .cmd shim starts a Node MCP server, which starts another child before reading any request.
        # Delaying contain makes the historical Popen-then-contain race deterministic.
        node = shutil.which('node')
        if not node:
            self.skipTest('Node required for the npx-style wrapper fixture')
        worker = '''const {spawn} = require('child_process');
const g = spawn(process.execPath, ['-e', 'setInterval(()=>{},1000)'], {stdio:'ignore',windowsHide:true});
require('fs').writeFileSync(process.argv[2], String(g.pid));
require('readline').createInterface({input:process.stdin}).on('line', line => {
 const msg=JSON.parse(line); if (!('id' in msg)) return;
 const result=msg.method==='initialize' ? {protocolVersion:'2025-06-18',serverInfo:{name:'private-test'}} :
  msg.method==='tools/list' ? {tools:[{name:'echo',inputSchema:{type:'object',properties:{text:{type:'string'}}}}]} :
  {content:[{type:'text',text:msg.params.arguments.text}]};
 console.log(JSON.stringify({jsonrpc:'2.0',id:msg.id,result}));
});
'''
        with tempfile.TemporaryDirectory(prefix="tiny-job-mcp-") as tmp:
            temp = Path(tmp)
            (temp / "worker.js").write_text(worker, encoding="utf-8")
            (temp / "shim.cmd").write_text('@echo off\r\n"%s" "%s" "%s"\r\n' %
                                           (node, temp / "worker.js", temp / "grandchild.pid"), encoding="ascii")
            host = '''import json,os,sys,time
sys.path.insert(0, ROOT)
from serve import winjob
from serve.mcp import StdioTransport
real_contain = winjob.contain
assigned = []
def delayed(proc):
    time.sleep(0.5)
    result = real_contain(proc)
    assigned.append(result)
    return result
winjob.contain = delayed
t = StdioTransport('private', {'command':SHIM, 'env':{'TINY_JOB_TEST':'rpc'}})
t.start()
a = t.request('initialize', {}, 10)
b = t.request('tools/list', {}, 10)
c = t.request('tools/call', {'name':'echo','arguments':{'text':'42台 / 12月22日'}}, 10)
print(json.dumps({'owner':os.getpid(),'child':t.proc.pid,'grandchild':int(open(PIDFILE).read()),'init':a,'tools':b,'result':c,'assigned':assigned,'job':winjob._job}), flush=True)
time.sleep(120)
'''
            host = "ROOT=%r\nSHIM=%r\nPIDFILE=%r\n" % (str(ROOT), str(temp / "shim.cmd"), str(temp / "grandchild.pid")) + host
            # A Windows venv redirector may outlive or be killed independently of the actual Python host.
            # Launch the interpreter directly so killing this handle really closes the host's job handle.
            proc = REAL_POPEN([DIRECT_PYTHON, "-X", "utf8", "-c", host], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, encoding="utf-8", creationflags=FLAGS)
            handles = []
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenProcess.restype = ctypes.c_void_p
            k32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
            k32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
            k32.TerminateProcess.argtypes = (ctypes.c_void_p, ctypes.c_uint)
            k32.CloseHandle.argtypes = (ctypes.c_void_p,)
            try:
                line = line_with_deadline(proc.stdout)
                if not line:
                    self.fail(proc.stderr.read())
                result = json.loads(line)
                self.assertEqual(result['assigned'], [True], result)
                self.assertEqual(result['owner'], proc.pid, result)
                self.assertEqual(result['tools']['tools'][0]['name'], 'echo')
                self.assertEqual(result['result']['content'][0]['text'], '42台 / 12月22日')
                for key in ('child', 'grandchild'):
                    handle = k32.OpenProcess(0x00100000 | 0x0001, False, result[key])
                    self.assertTrue(handle, key)
                    handles.append(handle)
                    self.assertEqual(k32.WaitForSingleObject(handle, 0), 258, key + ' already ended')
                proc.kill()  # hard host termination: no transport.close(), finally or atexit is run
                proc.wait(10)
                for index, handle in enumerate(handles):
                    self.assertEqual(k32.WaitForSingleObject(handle, 10000), 0, f'early descendant {index} escaped the job: {result}')
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(10)
                for handle in handles:
                    if k32.WaitForSingleObject(handle, 0) != 0:
                        k32.TerminateProcess(handle, 1)
                    k32.CloseHandle(handle)
                for pipe in (proc.stdout, proc.stderr):
                    pipe.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
