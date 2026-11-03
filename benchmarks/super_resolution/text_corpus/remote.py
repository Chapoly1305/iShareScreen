"""Explicit SSH/SFTP transport for the training host, with known-host checks.

This optional Paramiko path avoids macOS account/DNS lookup failures in the
system ssh binary. It never changes system accounts or disables host checking.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path, PurePosixPath
import posixpath
import stat
import sys
import threading
import time


def connect(args):
    import paramiko

    client = paramiko.SSHClient()
    client.load_host_keys(str(args.known_hosts))
    client.connect(
        args.host,
        username=args.user,
        key_filename=str(args.key),
        allow_agent=True,
        look_for_keys=False,
        timeout=15,
        banner_timeout=15,
        auth_timeout=15,
    )
    client.get_transport().set_keepalive(20)
    return client


def upload(args):
    root = args.root.resolve()
    files = Path(args.files).read_text().splitlines()
    remote = PurePosixPath(args.destination)
    if not remote.is_absolute() or len(remote.parts) < 4:
        raise ValueError("Choose an explicit isolated task directory")
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not (root / name).is_file():
            raise ValueError("Unsafe or missing upload path: " + name)
    client = connect(args)
    thread = threading.local()
    handles = []
    handle_lock = threading.Lock()

    def send(name):
        if not hasattr(thread, "sftp"):
            thread.sftp = client.open_sftp()
            thread.directories = set()
            with handle_lock:
                handles.append(thread.sftp)
        sftp = thread.sftp
        target = str(remote / name)
        parent = posixpath.dirname(target)
        parts = PurePosixPath(parent).parts
        current = ""
        for part in parts:
            current = posixpath.join(current, part)
            if current in thread.directories:
                continue
            try:
                sftp.mkdir(current)
            except OSError:
                sftp.stat(current)
            thread.directories.add(current)
        temporary = target + f".upload-{os.getpid()}"
        sftp.put(str(root / name), temporary, confirm=True)
        sftp.posix_rename(temporary, target)
        stat = (root / name).stat()
        sftp.utime(target, (stat.st_atime, stat.st_mtime))
        return stat.st_size

    total = 0
    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for n, size in enumerate(pool.map(send, files), 1):
                total += size
                if n % 500 == 0:
                    print(
                        "UPLOAD",
                        n,
                        "/",
                        len(files),
                        "MiB",
                        round(total / 2**20, 1),
                        "elapsed_s",
                        round(time.time() - started),
                        flush=True,
                    )
        print("UPLOAD COMPLETE", len(files), "files", total, "bytes", flush=True)
    finally:
        for handle in handles:
            handle.close()
        client.close()


def command(args):
    import select

    client = connect(args)
    try:
        _, stdout, stderr = client.exec_command(args.command)
        channel = stdout.channel
        while True:
            if channel.recv_ready():
                sys.stdout.buffer.write(channel.recv(65536))
                sys.stdout.buffer.flush()
            if channel.recv_stderr_ready():
                sys.stderr.buffer.write(channel.recv_stderr(65536))
                sys.stderr.buffer.flush()
            if (
                channel.exit_status_ready()
                and not channel.recv_ready()
                and not channel.recv_stderr_ready()
            ):
                break
            select.select([channel], [], [], 0.1)
        raise SystemExit(channel.recv_exit_status())
    finally:
        client.close()


def download(args):
    root = args.root.resolve()
    remote = PurePosixPath(args.destination)
    files = Path(args.files).read_text().splitlines()
    if not remote.is_absolute() or len(remote.parts) < 4:
        raise ValueError("Explicit remote task directory required")
    for name in files:
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe download path")
        (root / name).resolve().relative_to(root)
    client = connect(args)
    thread = threading.local()
    handles = []
    lock = threading.Lock()

    def receive(name):
        if not hasattr(thread, "sftp"):
            thread.sftp = client.open_sftp()
            with lock:
                handles.append(thread.sftp)
        source = str(remote / name)
        target = root / name
        sftp = thread.sftp
        info = sftp.lstat(source)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Download source is not a regular file: " + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError("Download destination already exists: " + str(target))
        temporary = target.with_name(target.name + f".download-{os.getpid()}")
        sftp.get(source, str(temporary))
        if temporary.stat().st_size != info.st_size:
            raise ValueError("Truncated download: " + name)
        temporary.replace(target)
        os.utime(target, (info.st_atime, info.st_mtime))
        return info.st_size

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            sizes = list(pool.map(receive, files))
        print("DOWNLOAD COMPLETE", len(sizes), "files", sum(sizes), "bytes", flush=True)
    finally:
        for handle in handles:
            handle.close()
        client.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["command", "upload", "download"])
    p.add_argument("--host", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--key", type=Path, default=Path.home() / ".ssh/id_ed25519")
    p.add_argument("--known-hosts", type=Path, default=Path.home() / ".ssh/known_hosts")
    p.add_argument("--command")
    p.add_argument("--root", type=Path)
    p.add_argument("--files", type=Path)
    p.add_argument("--destination")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    dict(command=command, upload=upload, download=download)[args.action](args)


if __name__ == "__main__":
    main()
