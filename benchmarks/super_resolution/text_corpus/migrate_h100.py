"""Migrate the complete project without changing frozen remote experiments.

Git history and the dirty working tree are preserved separately. Existing H100
data is reused only after SHA256 matches; new files are streamed to the fresh
checkout. Local data and credentials remain in place. Public download jobs move
to H100 before the longer historical/raw recording transfer finishes.
"""

import argparse
from collections import defaultdict
import hashlib
import gzip
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import threading
import time
from types import SimpleNamespace

from benchmarks.super_resolution.text_corpus.remote import connect
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256

TARGET = "/home/jchen73/iShareScreen"
STAGE = "/home/jchen73/isharescreen-migration.20261021"


def execute(client, command):
    _, out, err = client.exec_command(command)
    value = out.read()
    error = err.read()
    code = out.channel.recv_exit_status()
    if code:
        raise RuntimeError(f"Remote exit {code}: " + error.decode()[-3000:])
    return value


def py(client, source):
    return execute(client, "python3 -c " + shlex.quote(source))


def send_tar(client, rows, root):
    if not rows:
        return
    incoming, out, err = client.exec_command(
        "tar --no-same-owner -xzf - -C " + shlex.quote(root)
    )
    with gzip.GzipFile(
        filename="", fileobj=incoming, mode="wb", compresslevel=1
    ) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", dereference=True) as archive:
            for row in rows:
                source = Path(row["source"])
                if (
                    source.stat().st_size != row["bytes"]
                    or sha256(source) != row["sha256"]
                ):
                    raise ValueError(
                        "Local source changed during migration: " + str(source)
                    )
                archive.add(source, arcname=row["target"], recursive=False)
    incoming.flush()
    incoming.channel.shutdown_write()
    code = out.channel.recv_exit_status()
    if code:
        raise RuntimeError("Remote extraction failed: " + err.read().decode()[-2000:])


def record(source, target):
    source = source.resolve()
    return dict(
        source=str(source),
        target=target,
        bytes=source.stat().st_size,
        sha256=sha256(source),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--resume-bootstrap", action="store_true")
    a = p.parse_args()
    root = a.root.resolve()
    out = root / "benchmarks/super_resolution/artifacts/h100-migration-v1"
    out.mkdir(exist_ok=True)
    started = time.time()
    client = connect(
        SimpleNamespace(
            host="10.193.124.20",
            user="jchen73",
            key=Path.home() / ".ssh/id_ed25519",
            known_hosts=Path.home() / ".ssh/known_hosts",
        )
    )

    def status(phase, **more):
        atomic_json(
            out / "status.json",
            dict(
                phase=phase,
                target=TARGET,
                pid=os.getpid(),
                elapsed_s=time.time() - started,
                updated=time.time(),
                **more,
            ),
        )

    try:
        status("migrating_git_and_worktree")
        files = (
            subprocess.check_output(
                ["git", "ls-files", "-c", "-o", "--exclude-standard", "-z"], cwd=root
            )
            .decode()
            .split("\0")
        )
        code = [
            record(root / f, f)
            for f in sorted(set(files))
            if f
            and not f.startswith(("tmp/", ".agents/", ".codex/"))
            and (root / f).is_file()
        ]
        atomic_json(out / "code-manifest.json", code)
        if a.resume_bootstrap:
            expected = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            py(
                client,
                f"import pathlib,subprocess; r=pathlib.Path({TARGET!r}); assert not (r/'.migration/environment-launch.json').exists(); assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()=={expected!r}",
            )
        else:
            execute(client, "mkdir " + shlex.quote(STAGE))
        with client.open_sftp() as sftp:
            sftp.put(str(out / "repository.bundle"), STAGE + "/repository.bundle")
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=root, text=True
        ).strip()
        if not a.resume_bootstrap:
            execute(
                client,
                "git clone --branch "
                + shlex.quote(branch)
                + " "
                + shlex.quote(STAGE + "/repository.bundle")
                + " "
                + shlex.quote(TARGET),
            )
        send_tar(client, code, TARGET)
        execute(client, "mkdir -p " + shlex.quote(TARGET + "/.migration"))
        with client.open_sftp() as sftp:
            sftp.put(
                str(out / "code-manifest.json"),
                TARGET + "/.migration/code-manifest.json",
            )
        bootstrap = f"""import subprocess,os,json,time,pathlib
r=pathlib.Path({TARGET!r});log=(r/'.migration/environment.log').open('xb')
p=subprocess.Popen(['/home/jchen73/cs682-isharescreen/.venv/bin/python',str(r/'benchmarks/super_resolution/text_corpus/h100_setup.py'),'--root',str(r)],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(r/'.migration/environment-launch.json').write_text(json.dumps(dict(pid=p.pid,started=time.time())))
"""
        py(client, bootstrap)
        status("handoff_public_downloads")
        base = root / "benchmarks/super_resolution/artifacts/public-text-downloads-v1"
        handoff = []
        for path in base.rglob("*"):
            if (
                not path.is_file()
                or path.suffix in (".partial", ".assembling")
                or path.name.endswith(".log")
            ):
                continue
            rel = path.relative_to(base)
            # Preserve complete TextZoom, all source indexes, and completed slow
            # IBM byte ranges. Other public archives are fetched on H100 by their
            # frozen upstream hash, avoiding a large one-off laptop upload.
            keep = (
                rel.parts[0] == "textzoom"
                or (rel.parts[0] == "doclaynet" and path.suffix == ".bin")
                or len(rel.parts) == 1
                or rel.parts[0] == "notices"
            )
            if keep and path.stat().st_size < 2**30:
                handoff.append(record(path, str(path.relative_to(root))))
        for path in base.glob("doclaynet/*.parts/source.json"):
            handoff.append(record(path, str(path.relative_to(root))))
        handoff = list({r["target"]: r for r in handoff}.values())
        send_tar(client, handoff, TARGET)
        # Do not let stale local process receipts masquerade as active H100 jobs.
        launch = f"""import os,pathlib,subprocess,json,time
r=pathlib.Path({TARGET!r});d=r/'benchmarks/super_resolution/artifacts/public-text-downloads-v1';logs=r/'.migration';jobs=[]
env=dict(os.environ,PYTHONPATH=str(r)+':'+str(r/'src'))
python='/home/jchen73/cs682-isharescreen/.venv/bin/python'
for name,module,args in [('doclaynet','public_downloads',['--dataset','doclaynet','--output',str(d)]),('pubtabnet','public_downloads',['--dataset','pubtabnet','--output',str(d)]),('additional','public_download_queue',['--root',str(d)])]:
 log=(logs/(name+'-download.log')).open('xb')
 child=subprocess.Popen([python,'-u','-m','benchmarks.super_resolution.text_corpus.'+module]+args,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 jobs.append(dict(name=name,pid=child.pid))
(logs/'downloads-launched.json').write_text(json.dumps(dict(jobs=jobs,started=time.time()),indent=2))
print(json.dumps(jobs))
"""
        print(py(client, launch).decode(), flush=True)
        status("inventorying_historical_data")
        rows = []
        artifacts = root / "benchmarks/super_resolution/artifacts"
        for path in artifacts.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(artifacts)
            if (
                rel.parts[0] in ("public-text-downloads-v1", "h100-migration-v1")
                or "__pycache__" in rel.parts
            ):
                continue
            rows.append(record(path, str(path.relative_to(root))))
        for prefix, source in [
            (
                ".cache/super_resolution",
                Path.home() / ".cache/isharescreen/super_resolution",
            ),
            ("datasets/cs682_dataset", Path("/Volumes/Tools/cs682_dataset")),
        ]:
            for path in source.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    rows.append(
                        record(path, str(Path(prefix) / path.relative_to(source)))
                    )
        ocr = Path("/opt/homebrew/share/tessdata/eng.traineddata")
        if ocr.is_file():
            rows.append(record(ocr, "tools-runtime/tessdata/eng.traineddata"))
        atomic_json(out / "data-manifest.json", rows)
        with client.open_sftp() as sftp:
            sftp.put(
                str(out / "data-manifest.json"),
                TARGET + "/.migration/data-manifest.json",
            )
        # One remote scan, matching names/sizes before hashing candidates. Copies
        # are independent files; the old source trees remain unchanged.
        reuse = f"""import pathlib,os,json,hashlib,shutil,time
r=pathlib.Path({TARGET!r});rows=json.loads((r/'.migration/data-manifest.json').read_text());wanted={{(pathlib.Path(x['target']).name,x['bytes']) for x in rows}};index={{}};seen=set()
roots=[pathlib.Path('/home/jchen73/cs682-isharescreen')]+list(pathlib.Path('/home/jchen73').glob('isharescreen-*'))
for base in roots:
 if not base.is_dir() or str(base)=={STAGE!r}:continue
 for here,dirs,files in os.walk(base,followlinks=True):
  real=os.path.realpath(here)
  if real in seen:dirs[:]=[];continue
  seen.add(real);dirs[:]=[d for d in dirs if d not in ['.venv','.git','__pycache__','code']]
  for name in files:
   p=pathlib.Path(here)/name
   try:key=(name,p.stat().st_size)
   except OSError:continue
   if key in wanted:index.setdefault(key,[]).append(str(p))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
missing=[];reused=[];cache={{}}
for row in rows:
 dest=r/row['target'];matched=None
 for candidate in index.get((dest.name,row['bytes']),[]):
  if candidate not in cache:cache[candidate]=sha(candidate)
  if cache[candidate]==row['sha256']:matched=candidate;break
 if matched:
  dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(matched,dest);reused.append(dict(target=row['target'],source=matched,bytes=row['bytes']))
 else:missing.append(row)
(r/'.migration/reused.json').write_text(json.dumps(reused));(r/'.migration/missing.json').write_text(json.dumps(missing))
print(json.dumps(dict(reused_files=len(reused),reused_bytes=sum(x['bytes'] for x in reused),missing_files=len(missing),missing_bytes=sum(x['bytes'] for x in missing))))
"""
        print(py(client, reuse).decode(), flush=True)
        with client.open_sftp() as sftp:
            with sftp.file(TARGET + "/.migration/missing.json") as stream:
                missing = json.loads(stream.read())
        status(
            "transferring_remaining_data",
            files=len(missing),
            bytes=sum(r["bytes"] for r in missing),
        )
        send_tar(client, missing, TARGET)
        verify = f"""import pathlib,json,hashlib,time
r=pathlib.Path({TARGET!r});rows=json.loads((r/'.migration/code-manifest.json').read_text())+json.loads((r/'.migration/data-manifest.json').read_text())
for row in rows:
 p=r/row['target'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 if p.stat().st_size!=row['bytes'] or h.hexdigest()!=row['sha256']:raise ValueError('Migration hash mismatch: '+row['target'])
packed=r/'benchmarks/super_resolution/artifacts/text-corpus-v1/packed-v1'
if not packed.exists():packed.symlink_to('/home/jchen73/isharescreen-text-corpus-v1.20261021/data/packed-v1',target_is_directory=True)
receipt=dict(files=len(rows),bytes=sum(x['bytes'] for x in rows),all_sha256_matched=True,finished=time.time())
(r/'.migration/DATA-COMPLETED.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
"""
        proof = json.loads(py(client, verify))
        atomic_json(out / "DATA-COMPLETED.json", proof)
        status("data_migrated_environment_validation_pending", **proof)
    except BaseException as error:
        status("failed", error=repr(error))
        raise
    finally:
        client.close()


if __name__ == "__main__":
    main()
