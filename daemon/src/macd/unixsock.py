"""Private AF_UNIX sockets shared by the control and audio servers (ARCHITECTURE §11, C14)."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import socket
import stat
from pathlib import Path

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


def peer_uid(sock: socket.socket) -> int:
    """Effective uid of the peer of a connected unix socket (getpeereid(2))."""
    uid, gid = ctypes.c_uint32(), ctypes.c_uint32()
    if _libc.getpeereid(sock.fileno(), ctypes.byref(uid), ctypes.byref(gid)) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return uid.value


def bind_private_unix_socket(path: Path, backlog: int = 16) -> socket.socket:
    """Listening AF_UNIX socket at `path`: directory 0700, socket 0600, stale file removed."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if path.exists() or path.is_symlink():
        if not stat.S_ISSOCK(path.lstat().st_mode):
            raise FileExistsError(f"{path} exists and is not a socket")
        path.unlink()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old = os.umask(0o077)  # process-wide, restored immediately
    try:
        sock.bind(str(path))
    finally:
        os.umask(old)
    os.chmod(path, 0o600)
    sock.listen(backlog)
    sock.setblocking(False)
    return sock
