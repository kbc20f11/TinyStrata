"""Local profiles for TinyStrata's fixed model families.

Model names select these records, never an arbitrary path or download URL.
This module prepares files/configuration only; it never starts a native process.
The model controller owns admission, download jobs and runtime switching.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from serve import config, model_setup, tinyconfig
from serve.model_download import manifest_for_arch


_MODELS = {
    "qwen3.6-35b-a3b": "qwen35moe",
    "lfm2.5-24b-a2b": "lfm2moe",
    "gemma-4-26b-a4b-it": "gemma4",
}
# Supported model/engine ceilings, not automatically allocated KV capacities.
_CONTEXT_LIMITS = {"qwen35moe": 262144, "lfm2moe": 131072, "gemma4": 262144}
_ALIASES = {
    "qwen36": "qwen3.6-35b-a3b", "qwen3.6": "qwen3.6-35b-a3b",
    "qwen35moe": "qwen3.6-35b-a3b",
    "lfm2": "lfm2.5-24b-a2b", "lfm2.5": "lfm2.5-24b-a2b",
    "lfm2moe": "lfm2.5-24b-a2b",
    "gemma4": "gemma-4-26b-a4b-it", "gemma-4": "gemma-4-26b-a4b-it",
    "gemma4-26b-a4b": "gemma-4-26b-a4b-it", "gemma4-26b-a4b-it": "gemma-4-26b-a4b-it",
}


@dataclass(frozen=True)
class ModelProfile:
    id: str
    arch: str
    path: Path
    config: dict

    @property
    def shared_path(self) -> Path:
        return Path(str(self.path.with_suffix("")) + ".shared-settings.json")


class ModelProfiles:
    """Keep model settings separate while sharing the selected GPU/runtime."""

    def __init__(self, config_path: str | Path, cfg: dict):
        current_path = Path(config_path).expanduser().resolve()
        current_arch = cfg.get("arch", "qwen35moe")
        if current_arch not in _MODELS.values():
            raise ValueError(f"Unsupported model profile architecture: {current_arch}")
        self.current_id = next(name for name, arch in _MODELS.items() if arch == current_arch)
        self._hardware = {key: deepcopy(cfg[key]) for key in ("gpu", "hip_ordinal", "backend", "lib_dirs") if key in cfg}
        self._engine = model_setup.resolve_executable(cfg)
        self._profiles = {self.current_id: ModelProfile(self.current_id, current_arch, current_path, deepcopy(cfg))}
        for name, arch in _MODELS.items():
            if name == self.current_id:
                continue
            path = self._profile_path(current_path.parent, arch, current_path)
            loaded = config.load(path) if path.is_file() else self._defaults(name, arch)
            self._check_arch(path, loaded, arch)
            self._profiles[name] = ModelProfile(name, arch, path, loaded)
        # Both configured download directories and model folders can contain
        # the other pinned artifact. Look only for its exact fixed filename.
        directories = []
        for profile in self._profiles.values():
            existing = model_setup.model_path(profile.config)
            if existing is not None:
                directories.append(existing.parent)
            if profile.config.get("download_dir"):
                directory = Path(profile.config["download_dir"]).expanduser()
                if not directory.is_absolute():
                    directory = Path(profile.config.get("cwd") or model_setup.ROOT) / directory
                directories.append(directory)
            directories.append(profile.path.parent / "model-files")
        self._directories = tuple(dict.fromkeys(path.resolve() for path in directories))
        for name, profile in tuple(self._profiles.items()):
            prepared = self._settings(profile.config, profile)
            self._profiles[name] = ModelProfile(name, profile.arch, profile.path, prepared)

    @staticmethod
    def resolve(name: str) -> str:
        if not isinstance(name, str):
            raise ValueError("model must be one of TinyStrata's supported model names")
        key = name.strip().lower()
        if key in _MODELS:
            return key
        if key in _ALIASES:
            return _ALIASES[key]
        raise ValueError(f"Unknown TinyStrata model: {name!r}")

    @staticmethod
    def _check_arch(path: Path, cfg: dict, arch: str) -> None:
        if cfg.get("arch", "qwen35moe") != arch:
            raise ValueError(f"Model profile {path.name} belongs to another model family")

    @staticmethod
    def _profile_path(parent: Path, arch: str, current: Path) -> Path:
        names = {"qwen35moe": ("config.json", "config-qwen36.json"),
                 "lfm2moe": ("config-lfm2.json", "config-lfm2-profile.json"),
                 "gemma4": ("config-gemma4.json", "config-gemma4-profile.json")}[arch]
        for filename in names:
            path = parent / filename
            if path == current:
                continue
            if not path.exists():
                return path
            if path.is_file() and config.load(path).get("arch", "qwen35moe") == arch:
                return path
        raise ValueError(f"No separate configuration path is available for {arch}")

    def _defaults(self, name: str, arch: str) -> dict:
        args = ["--max-context", "auto", "--kv-type", "auto", "--chunk", "auto"]
        if arch in ("qwen35moe", "gemma4"):
            args += ["--mtp", "3"]
        result = {"arch": arch, "model_name": name, "cwd": str(model_setup.ROOT),
                  "args": args, "memory_policy_version": tinyconfig.MEMORY_POLICY_VERSION,
                  **deepcopy(self._hardware)}
        if self._engine is not None:
            filename = {"lfm2moe": "tinystrata-lfm2", "gemma4": "tinystrata-gemma4"}.get(arch, "tinystrata")
            sibling = self._engine.with_name(filename + self._engine.suffix)
            if sibling.is_file():
                result["exe"] = str(sibling)
        return result

    def _settings(self, cfg: dict, profile: ModelProfile) -> dict:
        from serve.backend import configure_backend
        self._check_arch(profile.path, cfg, profile.arch)
        source = deepcopy(cfg)
        # All profiles in one API process share the same device/runtime. A saved
        # CUDA profile must not switch an AMD installation back to CUDA.
        backend_changed = (self._hardware.get("backend") is not None and
                           source.get("backend", "cuda") != self._hardware["backend"])
        # Normalize KV while the profile still names its original backend.
        # Overwriting it first would mistake a CUDA FP8 setting for a deliberate
        # HIP choice and bypass the migration's engine auto policy.
        source = configure_backend(source, self._hardware.get("backend"))
        if self._hardware.get("backend") == "hip" or backend_changed:
            for key in ("gpu", "hip_ordinal", "backend", "lib_dirs"):
                source.pop(key, None)
            source.update(deepcopy(self._hardware))
        if backend_changed:
            source.pop("exe", None)
            source.pop("vision_exe", None)
            source.pop("vision", None)
        if self._engine is not None and (self._hardware.get("backend") == "hip" or backend_changed):
            filename = {"lfm2moe": "tinystrata-lfm2", "gemma4": "tinystrata-gemma4"}.get(profile.arch, "tinystrata")
            sibling = self._engine.with_name(filename + self._engine.suffix)
            if sibling.is_file():
                source["exe"] = str(sibling)
                source["vision_exe"] = str(sibling.with_name("tinystrata-vision" + sibling.suffix))
        source.setdefault("arch", profile.arch)
        source.setdefault("model_name", profile.id)
        result, _ = tinyconfig.migrate_settings(source)
        # Retain an explicitly configured download directory as a lookup source
        # before configure_download normalizes the managed destination.
        candidates = [model_setup.model_path(result)]
        if result.get("download_dir"):
            directory = Path(result["download_dir"]).expanduser()
            if not directory.is_absolute():
                directory = Path(result.get("cwd") or model_setup.ROOT) / directory
            candidates.append(directory / manifest_for_arch(profile.arch).filename)
        result = model_setup.shared_library_dirs(model_setup.configure_download(result, profile.path))
        candidates += [model_setup.model_path(result)]
        candidates += [directory / manifest_for_arch(profile.arch).filename for directory in self._directories]
        existing = next((path for path in candidates if path is not None and path.is_file()), None)
        if existing is not None:
            index = result["args"].index("--model") + 1
            result["args"][index] = str(existing.resolve())
        else:
            result["model_bootstrap"] = True
        if profile.arch == "gemma4":
            from serve.model_download import FIXED_GEMMA4_ASSISTANT, FIXED_GEMMA4_MMPROJ
            for field, artifact in (("draft_model", FIXED_GEMMA4_ASSISTANT), ("vision_mmproj", FIXED_GEMMA4_MMPROJ)):
                if not result.get(field):
                    folders = ((existing.parent,) if existing is not None else ()) + self._directories
                    found = next((folder / artifact.filename for folder in folders if (folder / artifact.filename).is_file()), None)
                    if found is not None:
                        result[field] = str(found)
        return result

    def get(self, name: str) -> ModelProfile:
        return self._profiles[self.resolve(name)]

    def context_info(self, name: str) -> dict:
        """Configured context and model ceiling; allocation requires native READY."""
        profile = self.get(name)
        limit = _CONTEXT_LIMITS[profile.arch]
        requested = config.value_of(profile.config, "max_context")
        if requested in (None, "auto", 0):
            requested = "auto"
        else:
            requested = config.check("max_context", requested, profile.config)
            if requested > limit:
                raise ValueError(f"{profile.id} supports at most {limit} context tokens")
        return {"allocated_tokens": None, "model_limit": limit, "requested": requested}

    def list(self) -> list[dict]:
        """Cheap availability metadata; no checksums, GGUF reads or processes."""
        entries = []
        for profile in self._profiles.values():
            model = model_setup.model_path(profile.config)
            exe = model_setup.resolve_executable(profile.config)
            installed = model is not None and model.is_file()
            entries.append({"id": profile.id, "arch": profile.arch, "installed": installed,
                            "path": str(model) if model is not None else None,
                            "executable": str(exe) if exe is not None else None,
                            "config_path": str(profile.path), "shared_settings_path": str(profile.shared_path),
                            "context": self.context_info(profile.id),
                            "vision": profile.arch in ("qwen35moe", "gemma4") and profile.config.get("vision_mode", "gpu") != "no"})
        return entries

    def update(self, name: str, cfg: dict, *, save: bool = False) -> ModelProfile:
        """Update the controller's profile cache; optionally persist that profile."""
        current = self.get(name)
        self._check_arch(current.path, cfg, current.arch)
        profile = ModelProfile(current.id, current.arch, current.path, deepcopy(cfg))
        if save:
            model_setup.save_settings(profile.path, profile.config)
        self._profiles[profile.id] = profile
        return profile

    def prepare(self, name: str, model_path: str | Path | None = None, *,
                projector: str | Path | None = None, projector_verified: bool = False,
                assistant: str | Path | None = None, assistant_verified: bool = False) -> dict:
        """Prepare and save only the chosen profile; no native process is started.

        Missing executables remain a downloadable/prepared bootstrap profile.
        The controller must verify native/Vision readiness before unloading the
        old model. A supplied model_path is used by a verified download callback.
        """
        profile = self.get(name)
        saved = config.load(profile.path) if profile.path.is_file() else profile.config
        cfg = self._settings(saved, profile)
        model = Path(model_path).expanduser().resolve() if model_path is not None else model_setup.model_path(cfg)
        if model is None or not model.is_file():
            raise ValueError(f"{profile.id} is not installed; download it before switching")
        if model_path is not None:
            cfg["args"][cfg["args"].index("--model") + 1] = str(model)
        tinyconfig.validate(cfg, allow_uninstalled=True)
        extra = {"assistant": Path(assistant) if assistant is not None else None,
                 "assistant_verified": assistant_verified} if profile.arch == "gemma4" else {}
        prepared = model_setup.prepare_installed_model(cfg, profile.path, model,
                    projector=Path(projector) if projector is not None else None,
                    projector_verified=projector_verified, **extra)
        tinyconfig.validate(prepared, allow_uninstalled=True)
        tinyconfig.check_files(prepared, allow_uninstalled=True)
        self.update(profile.id, prepared)
        return deepcopy(prepared)
