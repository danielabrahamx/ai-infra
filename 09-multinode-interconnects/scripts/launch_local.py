"""torchrun substitute for single-host runs where the elastic launcher fails.

Needed on Windows builds of PyTorch without libuv: torchrun's rendezvous
store requests libuv and errors out. This sets the same env vars torchrun
sets, then runs the target script once per rank.
Usage: python scripts/launch_local.py -n 2 scripts/sweep.py --backend gloo ...
"""
import os, socket, subprocess, sys


def main():
    n = int(sys.argv[2]) if sys.argv[1] == "-n" else 2
    cmd = sys.argv[3:]
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    procs = []
    for r in range(n):
        env = dict(os.environ, RANK=str(r), LOCAL_RANK=str(r), WORLD_SIZE=str(n),
                   LOCAL_WORLD_SIZE=str(n), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                   USE_LIBUV="0")
        procs.append(subprocess.Popen([sys.executable, *cmd], env=env))
    sys.exit(max(p.wait() for p in procs))


if __name__ == "__main__":
    main()
