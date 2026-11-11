"""Exercise the shipping native argv adapter through real Unicode child starts.

Set NATIVE_UTF8_ARGS_TEST_EXE to the CMake native_utf8_args_test executable.
This test starts no model, GPU program or application service.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


EXE = Path(os.environ.get('NATIVE_UTF8_ARGS_TEST_EXE', ''))


@unittest.skipUnless(EXE.is_file(), 'Build native_utf8_args_test and set NATIVE_UTF8_ARGS_TEST_EXE')
class NativeUtf8Arguments(unittest.TestCase):
    def echo(self, values, executable=EXE, cwd=None):
        result = subprocess.run([str(executable.resolve()), *values], cwd=cwd,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8', 'replace'))
        self.assertEqual(result.stderr, b'')
        # splitlines() retains the empty argument's blank line.
        return [bytes.fromhex(line).decode('utf-8') for line in result.stdout.decode('ascii').splitlines()]

    def test_japanese_spaces_and_non_ansi_characters(self):
        values = ['--model', 'C:\\資料 と モデル\\日本語.gguf', '𠮷野家_模型_😀.gguf', 'é_العربية_한글']
        self.assertEqual(self.echo(values), values)

    def test_argument_boundaries_and_windows_quoting(self):
        values = ['', 'simple', 'folder with spaces\\', '二重"引用符', '日本語\\\\', '\\"', '&%$;()']
        self.assertEqual(self.echo(values), values)

    def test_long_argument_not_truncated_or_ansi_encoded(self):
        values = ['資料/長い パス/' + '日本語😀' * 900 + '.gguf']
        self.assertEqual(self.echo(values), values)

    def test_unicode_executable_and_working_directory(self):
        with tempfile.TemporaryDirectory(prefix='tinystrata-argv-') as directory:
            folder = Path(directory) / '日本語 資料_𠮷😀'
            folder.mkdir()
            executable = folder / ('検証 プログラム' + EXE.suffix)
            shutil.copy2(EXE, executable)
            values = ['相対 日本語.gguf', '😀', '']
            self.assertEqual(self.echo(values, executable, folder), values)


if __name__ == '__main__':
    unittest.main()
