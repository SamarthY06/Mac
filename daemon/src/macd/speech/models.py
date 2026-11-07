"""Model locations and a resumable, mirrored, safe downloader.

Each model has an ordered list of sources (GitHub release first, HuggingFace mirrors
next). A source is tried until one works: connect/read timeouts and HTTP errors fall
through to the next. Downloads go to ``<file>.part`` with HTTP Range resume and an
atomic rename; archives are extracted with path checks.

``python -m macd.speech.models`` downloads whatever is missing into config.MODELS_DIR.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from macd import config

log = logging.getLogger(__name__)

KWS_NAME = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
PARAKEET_NAME = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
POCKET_NAME = "kyutai-pocket-tts-int8"
CONNECT_TIMEOUT_S = 10.0
_GH = "https://github.com"
_HF = "https://huggingface.co"


@dataclass(frozen=True)
class Source:
    """Where to get a model. ``files`` empty = a single file (or a .tar.bz2 archive for
    directory models) at ``url``; otherwise ``url`` is a base and each file is fetched from
    ``url/<file>``. Every source is saved under the spec's canonical ``target`` name."""

    url: str
    files: tuple[str, ...] = ()

    @property
    def archive(self) -> bool:
        return not self.files and self.url.endswith((".tar.bz2", ".tar.gz", ".tgz"))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    target: str  # file or directory name under MODELS_DIR
    sources: tuple[Source, ...]
    required: tuple[str, ...] = ()  # glob patterns that must exist inside a directory target
    optional: bool = False  # not needed by default (only when a feature using it is enabled)

    @property
    def is_dir(self) -> bool:
        return bool(self.required)


_KWS_FILES = (
    "tokens.txt",
    "bpe.model",
    "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    "decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
    "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
)
_PARAKEET_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
_POCKET_FILES = (
    "lm_main.int8.onnx", "lm_flow.int8.onnx", "encoder.onnx", "decoder.int8.onnx",
    "text_conditioner.onnx", "vocab.json", "token_scores.json", "LICENSE",
)

SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "kws",
        KWS_NAME,
        (
            Source(f"{_GH}/k2-fsa/sherpa-onnx/releases/download/kws-models/{KWS_NAME}.tar.bz2"),
            Source(f"{_HF}/mobilebytesensei/sherpa-onnx-kws-zipformer-gigaspeech-3.3M/resolve/main", _KWS_FILES),
        ),
        required=("tokens.txt", "bpe.model", "encoder*.onnx", "decoder*.onnx", "joiner*.onnx"),
    ),
    ModelSpec(
        "vad",
        "silero_vad.onnx",
        (
            Source(f"{_GH}/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"),
            Source(f"{_HF}/lquint/silero-vad-v4-onnx/resolve/main/silero_vad.onnx"),
            Source(f"{_HF}/deepghs/silero-vad-onnx/resolve/main/silero_vad.onnx"),
        ),
    ),
    ModelSpec(
        "kokoro",
        "kokoro-v1.0.int8.onnx",
        (
            Source(f"{_GH}/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx"),
            # No mirror of the exact int8 file exists; this is an 8-bit quantisation of the same v1.0 export.
            Source(f"{_HF}/Arxell/kokoro-v1.0.onnx/resolve/main/model_quantized.onnx"),
        ),
    ),
    ModelSpec(
        "voices",
        "voices-v1.0.bin",
        (
            Source(f"{_GH}/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"),
            Source(f"{_HF}/fastrtc/kokoro-onnx/resolve/main/voices-v1.0.bin"),
        ),
    ),
    ModelSpec(
        "parakeet",
        PARAKEET_NAME,
        (
            Source(f"{_GH}/k2-fsa/sherpa-onnx/releases/download/asr-models/{PARAKEET_NAME}.tar.bz2"),
            Source(f"{_HF}/csukuangfj/{PARAKEET_NAME}/resolve/main", _PARAKEET_FILES),
        ),
        required=_PARAKEET_FILES,
    ),
    ModelSpec(  # voice cloning (tts.engine = "clone"), opt-in; weights are CC-BY-4.0 (see LICENSE)
        "pocket",
        POCKET_NAME,
        # HF first: the GitHub tarball extracts to sherpa-onnx-pocket-tts-int8-2026-01-26/, not POCKET_NAME.
        (Source(f"{_HF}/csukuangfj2/sherpa-onnx-pocket-tts-int8-2026-01-26/resolve/main", _POCKET_FILES),),
        required=_POCKET_FILES[:-1],
        optional=True,
    ),
)
SPEC_BY_KEY = {s.key: s for s in SPECS}


def models_dir() -> Path:
    return config.MODELS_DIR


def path(key: str) -> Path:
    """Installed location of a model (file or directory) under MODELS_DIR."""
    return models_dir() / SPEC_BY_KEY[key].target


def is_present(spec: ModelSpec) -> bool:
    p = models_dir() / spec.target
    if spec.is_dir:
        return p.is_dir() and all(any(p.glob(pat)) for pat in spec.required)
    return p.is_file() and p.stat().st_size > 0


def missing_models(keys: list[str] | None = None) -> list[str]:
    """Keys of models that are not (fully) installed (optional ones only when named in ``keys``)."""
    specs = [SPEC_BY_KEY[k] for k in keys] if keys else [s for s in SPECS if not s.optional]
    return [s.key for s in specs if not is_present(s)]


ProgressCb = Callable[[str, int, int | None], None]  # (key, bytes_done, bytes_total)


def _download(url: str, dest: Path, key: str, progress_cb: ProgressCb | None, chunk: int = 1 << 20) -> None:
    """Download to ``dest`` via ``dest.part`` with HTTP Range resume; atomic rename."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "macd-model-downloader"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT_S)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:  # already complete
            os.replace(part, dest)
            return
        raise
    with resp:
        if have and resp.status != 206:  # server ignored Range: restart
            have = 0
        length = resp.headers.get("Content-Length")
        total = (int(length) + have) if length is not None else None
        done = have
        with open(part, "ab" if have else "wb") as f:
            while buf := resp.read(chunk):
                f.write(buf)
                done += len(buf)
                if progress_cb:
                    progress_cb(key, done, total)
    if total is not None and done < total:
        raise OSError(f"incomplete download of {url}: {done}/{total} bytes")
    os.replace(part, dest)


def _safe_members(tar: tarfile.TarFile, root: Path) -> list[tarfile.TarInfo]:
    root = root.resolve()
    members = []
    for m in tar.getmembers():
        name = m.name
        if os.path.isabs(name) or name.startswith("\\") or ".." in Path(name).parts:
            raise ValueError(f"unsafe path in archive: {name!r}")
        if m.issym() or m.islnk():
            raise ValueError(f"links not allowed in archive: {name!r}")
        if not (m.isfile() or m.isdir()):
            continue
        if not (root / name).resolve().is_relative_to(root):
            raise ValueError(f"path escapes target: {name!r}")
        members.append(m)
    return members


def safe_extract(archive: Path, dest_dir: Path) -> None:
    """Extract a tarball rejecting absolute paths, '..', links and special files."""
    with tarfile.open(archive, "r:*") as tar:
        members = _safe_members(tar, dest_dir)
        tar.extractall(dest_dir, members=members, filter="data")


def _install_archive(spec: ModelSpec, src: Source, root: Path, progress_cb: ProgressCb | None) -> None:
    archive = root / (spec.target + ".tar.bz2")
    _download(src.url, archive, spec.key, progress_cb)
    with tempfile.TemporaryDirectory(dir=root, prefix=".extract-") as tmp:
        safe_extract(archive, Path(tmp))
        extracted = Path(tmp) / spec.target
        if not extracted.is_dir():
            raise ValueError(f"archive {archive.name} did not contain {spec.target}/")
        final = root / spec.target
        if final.exists():
            shutil.rmtree(final)
        os.replace(extracted, final)
    archive.unlink(missing_ok=True)


def _install_files(spec: ModelSpec, src: Source, root: Path, progress_cb: ProgressCb | None) -> None:
    target = root / spec.target
    target.mkdir(parents=True, exist_ok=True)
    for name in src.files:
        dest = target / name
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        _download(f"{src.url}/{name}", dest, spec.key, progress_cb)


def _install(spec: ModelSpec, src: Source, root: Path, progress_cb: ProgressCb | None) -> None:
    if src.files:
        _install_files(spec, src, root, progress_cb)
    elif src.archive:
        _install_archive(spec, src, root, progress_cb)
    else:
        _download(src.url, root / spec.target, spec.key, progress_cb)


def download_models(progress_cb: ProgressCb | None = None, keys: list[str] | None = None) -> list[str]:
    """Download (and extract) missing models, trying each source in order. Returns installed keys."""
    root = models_dir()
    root.mkdir(parents=True, exist_ok=True)
    installed: list[str] = []
    for key in missing_models(keys):
        spec = SPEC_BY_KEY[key]
        errors: list[str] = []
        for src in spec.sources:
            log.info("downloading model %s from %s", key, src.url)
            try:
                _install(spec, src, root, progress_cb)
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                errors.append(f"{src.url}: {e}")
                log.warning("source failed for %s (%s); trying next", key, e)
                continue
            if is_present(spec):
                installed.append(key)
                break
            errors.append(f"{src.url}: incomplete")
        else:
            raise OSError(f"could not download model {key}:\n  " + "\n  ".join(errors))
    return installed


def _cli_progress(key: str, done: int, total: int | None) -> None:
    mb = done / 1e6
    if total:
        sys.stderr.write(f"\r{key}: {mb:7.1f} / {total / 1e6:.1f} MB ({100 * done / total:5.1f}%)")
    else:
        sys.stderr.write(f"\r{key}: {mb:7.1f} MB")
    sys.stderr.flush()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    keys = [k for k in (argv if argv is not None else sys.argv[1:]) if k in SPEC_BY_KEY] or None
    missing = missing_models(keys)
    if not missing:
        print(f"All models present in {models_dir()}")
        return 0
    print(f"Downloading {', '.join(missing)} into {models_dir()}")
    download_models(_cli_progress, keys)
    sys.stderr.write("\n")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
