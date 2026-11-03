"""Stream frozen training inputs between the two user-authorized GPU hosts.

No raw input is downloaded to the laptop or changed at the source. Archive
members are an explicit relative-file allowlist; the receiver must be fresh.
"""

import argparse
import json
from pathlib import Path
import shlex
import time
from types import SimpleNamespace

from benchmarks.super_resolution.text_corpus import remote
from benchmarks.super_resolution.training_state import atomic_json


def execute(client, command):
    _, out, err = client.exec_command(command)
    result = out.read()
    error = err.read()
    code = out.channel.recv_exit_status()
    if code:
        raise RuntimeError(f"Remote command exit {code}: {error.decode()[-2000:]}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    options = dict(
        key=Path.home() / ".ssh/id_ed25519",
        known_hosts=Path.home() / ".ssh/known_hosts",
    )
    source = remote.connect(
        SimpleNamespace(host="10.193.124.20", user="jchen73", **options)
    )
    target = remote.connect(
        SimpleNamespace(host="10.193.124.90", user="chen", **options)
    )
    previous = "/home/jchen73/isharescreen-text-h100-recovery.20261021"
    destination = "/home/chen/isharescreen-text-input.20261021"
    campaign = "/home/chen/isharescreen-official-transfer.20261021"
    started = time.time()
    compressed = 0

    def status(phase, **kwargs):
        atomic_json(
            a.output / "status.json",
            dict(
                phase=phase,
                compressed_bytes=compressed,
                elapsed_s=time.time() - started,
                updated=time.time(),
                **kwargs,
            ),
        )

    try:
        code = (
            """import json,pathlib
r=pathlib.Path(%r)
files=[str(p.relative_to(r)) for d in ['code','assets'] for p in (r/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
m=json.loads((r/'data/packed-v1/manifest.json').read_text())
files+=['data/packed-v1/manifest.json']+['data/packed-v1/'+v['path'] for v in m['files'].values()]
files+=['data/hevc-v2/pairs.json','data/evaluation-protocol/protocol.json','data/evaluation-protocol/source-compatibility.json']
pairs={p['id']:p for p in json.loads((r/'data/hevc-v2/pairs.json').read_text())['pairs']}
for c in json.loads((r/'data/evaluation-protocol/protocol.json').read_text())['cases']:
 if pairs[c['view']]['requested_bps'] in [256000,600000,1000000,2000000]:files+=['data/'+c['clean_crop'],'data/'+pairs[c['view']]['lr']]
assert all(not pathlib.PurePosixPath(p).is_absolute() and '..' not in pathlib.PurePosixPath(p).parts for p in files)
print(json.dumps(sorted(set(files))))
"""
            % previous
        )
        files = json.loads(execute(source, "python3 -c " + shlex.quote(code)))
        atomic_json(a.output / "files.json", files)
        listing = "/home/jchen73/isharescreen-transfer-input-list-v2.20261021.txt"
        with source.open_sftp() as sftp:
            with sftp.file(listing, "wx") as stream:
                stream.write("\n".join(files) + "\n")
        execute(target, "mkdir " + shlex.quote(destination))
        status("streaming", files=len(files))
        _, out, srcerr = source.exec_command(
            "tar --dereference -I "
            + shlex.quote("zstd -T2 -1")
            + " -cf - -C "
            + shlex.quote(previous)
            + " -T "
            + shlex.quote(listing)
        )
        incoming, dstout, dsterr = target.exec_command(
            "tar --no-same-owner -I zstd -xf - -C " + shlex.quote(destination)
        )
        tick = time.time()
        while True:
            block = out.read(2**20)
            if not block:
                break
            incoming.write(block)
            compressed += len(block)
            if time.time() - tick > 15:
                incoming.flush()
                status("streaming", files=len(files))
                print(
                    "STREAM",
                    round(compressed / 2**20, 1),
                    "compressed MiB",
                    round(time.time() - started),
                    "s",
                    flush=True,
                )
                tick = time.time()
        incoming.flush()
        incoming.channel.shutdown_write()
        src_code = out.channel.recv_exit_status()
        dst_code = dstout.channel.recv_exit_status()
        if src_code or dst_code:
            raise RuntimeError(
                "Archive transfer failed: "
                + repr(
                    (src_code, dst_code, srcerr.read()[-1000:], dsterr.read()[-1000:])
                )
            )
        status("installing_code")
        # Only these reviewed experiment files replace the older frozen runner.
        overlay = ["train.py", "evaluate.py", "official.py", "transfer_campaign.py"]
        prefix = "benchmarks/super_resolution/text_corpus/"
        execute(target, "mkdir -p " + shlex.quote(campaign + "/code/" + prefix))
        with target.open_sftp() as sftp:
            for name in overlay:
                sftp.put(
                    str(Path(__file__).with_name(name)),
                    campaign + "/code/" + prefix + name,
                    confirm=True,
                )
            for name in ["official-state.pt", "official-state.json"]:
                sftp.put(
                    str(a.output.parent / name),
                    destination + "/assets/" + name,
                    confirm=True,
                )
        bootstrap = """from pathlib import Path
import shutil
source=Path(%r);target=Path(%r)
for p in source.rglob('*'):
 if p.is_file() and '__pycache__' not in p.parts:
  q=target/p.relative_to(source)
  if not q.exists():q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
""" % (destination + "/code", campaign + "/code")
        execute(target, "python3 -c " + shlex.quote(bootstrap))
        python = "/home/chen/venvs/vllm/bin/python"
        env = (
            "PYTHONPATH="
            + shlex.quote(campaign + "/code:" + campaign + "/code/src")
            + " "
        )
        base = (
            env
            + shlex.quote(python)
            + " -u -m benchmarks.super_resolution.text_corpus.transfer_campaign "
        )
        print(
            execute(
                target,
                base
                + "prepare --root "
                + shlex.quote(campaign)
                + " --previous "
                + shlex.quote(destination)
                + " --epochs 100",
            ).decode(),
            flush=True,
        )
        print(
            execute(
                target,
                base
                + "start --root "
                + shlex.quote(campaign)
                + " --python "
                + shlex.quote(python)
                + " --gpu-uuid GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a",
            ).decode(),
            flush=True,
        )
        atomic_json(
            a.output / "COMPLETED.json",
            dict(
                source=previous,
                destination=destination,
                campaign=campaign,
                files=len(files),
                compressed_bytes=compressed,
                finished=time.time(),
            ),
        )
        status("launched")
    except BaseException as error:
        status("failed", error=repr(error))
        raise
    finally:
        source.close()
        target.close()


if __name__ == "__main__":
    main()
