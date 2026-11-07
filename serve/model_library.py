"""Bounded deletion of fixed managed model files; never recursively delete a directory.

Configuration, credentials, history, native executables and arbitrary local
models are outside this manifest. Catalog reads only metadata, never GGUF data.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import uuid

from serve.model_download import FIXED_GEMMA4_ASSISTANT, manifest_for_arch, projector_for_arch
from serve.model_setup import TOKENIZER_FILES


class UnsafeModelPath(ValueError):
    pass


def _absolute(path):
    return Path(os.path.abspath(Path(path).expanduser()))


def _key(path):
    return os.path.normcase(str(_absolute(path)))


def _configured(cfg, value):
    if not value or not isinstance(value, str):
        return None
    path = Path(value).expanduser()
    return _absolute(path if path.is_absolute() else Path(cfg.get('cwd') or '.') / path)


def _argument(cfg, flag):
    args = cfg.get('args') or []
    return args[args.index(flag) + 1] if flag in args[:-1] else None


def _references(profile):
    cfg = profile.config
    vision = cfg.get('vision') if isinstance(cfg.get('vision'), dict) else {}
    values = (_argument(cfg, '--model'), _argument(cfg, '--draft-model'), cfg.get('draft_model'),
              cfg.get('vision_mmproj'), vision.get('mmproj'), vision.get('model'), cfg.get('tokenizer'))
    return {_key(path) for value in values if (path := _configured(cfg, value)) is not None}


def artifacts(profile):
    result = [manifest_for_arch(profile.arch, profile.id)]
    projector = projector_for_arch(profile.arch, profile.id)
    if projector is not None and not (profile.arch == 'clover_moe' and profile.config.get('vision_mode') == 'no'):
        result.append(projector)
    if profile.arch == 'gemma4':
        result.append(FIXED_GEMMA4_ASSISTANT)
    return tuple(result)


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode)


def _no_link(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
        raise UnsafeModelPath('A managed model path is a symbolic link or junction; it was left unchanged')
    return info


def _checked(path, root):
    path, root = _absolute(path), _absolute(root)
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise UnsafeModelPath('A model file is outside its managed directory') from None
    if not relative.parts:
        raise UnsafeModelPath('A model file must not name its managed directory')
    current = root
    for part in (None, *relative.parts):
        if part is not None:
            current = current / part
        try:
            info = _no_link(current)
        except FileNotFoundError:
            return None
        if current != path and not stat.S_ISDIR(info.st_mode):
            raise UnsafeModelPath('A managed model parent is not a directory')
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise UnsafeModelPath('A managed model entry is not a private regular file')
    return info


@dataclass(frozen=True)
class ManagedFile:
    path: Path
    root: Path
    signature: tuple
    size: int
    shared: bool = False


@dataclass
class Inventory:
    management: dict
    files: tuple[ManagedFile, ...]


def inventory(profile, peers):
    directory = _absolute(profile.path.parent / 'model-files')
    manifests = artifacts(profile)
    primary = directory / manifests[0].filename
    selected = _configured(profile.config, _argument(profile.config, '--model'))
    managed = profile.config.get('model_source') != 'local' and (selected is None or _key(selected) == _key(primary))
    metadata = dict(source='managed' if managed else 'local', can_delete=False, delete_reason=None,
                    disk_bytes=0, reclaimable_bytes=0, shared_bytes=0, download_bytes=sum(a.size for a in manifests),
                    complete=False, can_download=False)
    # Readiness is independent of permission to remove a file. In particular,
    # benchmark hardlinks and custom paths can still be fully usable models.
    availability = []
    for index, artifact in enumerate(manifests):
        vision = profile.config.get('vision') if isinstance(profile.config.get('vision'), dict) else {}
        configured = selected if index == 0 else _configured(profile.config,
            _argument(profile.config, '--draft-model') or profile.config.get('draft_model')) if artifact == FIXED_GEMMA4_ASSISTANT else \
            _configured(profile.config, profile.config.get('vision_mmproj') or vision.get('mmproj'))
        path = configured or directory / artifact.filename
        try:
            present = path.is_file()
            availability.append(present)
            if present:
                metadata['disk_bytes'] += path.stat().st_size
        except OSError:
            availability.append(False)
    metadata['complete'] = all(availability)
    metadata['can_download'] = not metadata['complete'] and managed
    if not managed:
        # No fixed-bundle readiness assertion for arbitrary user model setups;
        # existing primary availability still decides picker eligibility.
        metadata['complete'] = None
        metadata['delete_reason'] = 'custom'
        return Inventory(metadata, ())
    references = set().union(*(_references(peer) for peer in peers if peer.id != profile.id))
    files = {}
    def add(path, root, shared=False):
        info = _checked(path, root)
        if info is not None:
            files[_key(path)] = ManagedFile(path, root, _signature(info), info.st_size, shared)
        return info
    try:
        primary_shared = False
        for artifact in manifests:
            path = directory / artifact.filename
            shared = _key(path) in references
            primary_shared |= path == primary and shared
            add(path, directory, shared)
            # The downloader owns only this exact resumable suffix.
            add(path.with_name(path.name + '.part'), directory, shared)
        # Tokenizer ownership is the same path/size/mtime identity used by
        # prepare_installed_model(). Unknown or stale cache folders stay put.
        primary_info = _checked(primary, directory)
        if primary_info is not None:
            identity = f'{primary.resolve()}\n{primary_info.st_size}\n{primary_info.st_mtime_ns}'
            cache_root = _absolute(profile.path.parent / 'models')
            cache = cache_root / hashlib.sha256(identity.encode()).hexdigest()[:16] / 'tokenizer'
            configured = _configured(profile.config, profile.config.get('tokenizer'))
            if configured is not None and _key(configured) == _key(cache):
                shared = _key(cache) in references
                for filename in (*TOKENIZER_FILES, 'merges.json'):
                    add(cache / filename, cache_root, shared)
        metadata['disk_bytes'] = sum(file.size for file in files.values())
        metadata['shared_bytes'] = sum(file.size for file in files.values() if file.shared)
        metadata['reclaimable_bytes'] = sum(file.size for file in files.values() if not file.shared)
        metadata['delete_reason'] = 'shared' if primary_shared else None if any(not file.shared for file in files.values()) else 'not_downloaded'
        metadata['can_delete'] = metadata['delete_reason'] is None
        metadata['can_download'] = not metadata['complete']
    except (OSError, UnsafeModelPath):
        metadata['delete_reason'] = 'unsafe_path'
        metadata['can_delete'] = metadata['can_download'] = False
        return Inventory(metadata, ())
    return Inventory(metadata, tuple(files.values()))


@contextmanager
def _pin_directories(paths):
    """On Windows deny directory rename/junction replacement during staging."""
    if os.name != 'nt':
        yield
        return
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    handles = []
    directories = {parent for path in paths for parent in _absolute(path).parents}
    try:
        for directory in sorted(directories, key=lambda item: len(item.parts)):
            if not directory.exists():
                continue
            _no_link(directory)
            # FILE_READ_ATTRIBUTES; FILE_SHARE_READ|WRITE, deliberately no DELETE.
            handle = create(str(directory), 0x80, 3, None, 3, 0x02000000 | 0x00200000, None)
            if handle == ctypes.c_void_p(-1).value:
                raise OSError(ctypes.get_last_error(), 'Cannot safely lock a managed model directory')
            handles.append(handle)
            _no_link(directory)
        yield
    finally:
        for handle in reversed(handles):
            close(handle)


def delete_files(plan):
    """Stage exact regular files before unlinking; failed staging restores them."""
    if not plan.management['can_delete']:
        raise ValueError('This model has no safely removable managed files')
    files = [file for file in plan.files if not file.shared]
    token = uuid.uuid4().hex
    staged = []
    def restore_pending():
        failures = 0
        for file, temporary in reversed(staged):
            if os.path.lexists(temporary):
                if os.path.lexists(file.path):
                    failures += 1  # Never overwrite a newer file to restore an old one.
                    continue
                try:
                    temporary.rename(file.path)
                except OSError:
                    failures += 1
        return failures
    with _pin_directories(file.path for file in files):
        try:
            for index, file in enumerate(files):
                info = _checked(file.path, file.root)
                if info is None or _signature(info) != file.signature:
                    raise UnsafeModelPath('Model files changed while removing them; retry from the model list')
                if getattr(info, 'st_file_attributes', 0) & 1:
                    raise PermissionError('A managed model file is read-only; it was left unchanged')
                staged_path = file.path.with_name(f'.tinystrata-delete-{token}-{index}')
                file.path.rename(staged_path)
                staged.append((file, staged_path))
                moved = _checked(staged_path, file.root)
                # Rename can update ctime, so compare stable identity/data fields.
                if moved is None or _signature(moved)[:4] != file.signature[:4]:
                    raise UnsafeModelPath('A model file changed during removal; it was left unchanged')
        except BaseException:
            if restore_pending():
                raise OSError('Removal staging failed; some undeleted files could not be restored and remain in their private staging names') from None
            raise
        removed = 0
        try:
            for file, temporary in staged:
                current = _checked(temporary, file.root)
                if current is None or _signature(current)[:4] != file.signature[:4]:
                    raise UnsafeModelPath('A staged model file changed during removal')
                temporary.unlink()
                removed += file.size
        except BaseException as error:
            failures = restore_pending()
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            note = f'{failures} undeleted files remain in private staging names' if failures else 'all remaining undeleted files were restored'
            raise OSError(f'Removal stopped after deleting {removed} bytes; {note}: {error}') from None
    # Keep empty cache directories: deleting only explicit manifest files also
    # avoids a junction race during a separate directory-cleanup pass.
    return removed
