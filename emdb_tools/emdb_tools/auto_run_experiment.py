import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
import threading
from datetime import datetime

stop_requested = False


def handle_signal(signum, frame):
    global stop_requested
    stop_requested = True
    print("Interrupted.", flush=True)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

parser = argparse.ArgumentParser(
    prog=os.path.basename(sys.argv[0]),
    description="Run a ROS command multiple times, each time in a new directory.",
    add_help=False,
)
parser.add_argument("-c", dest="cmd", required=True, help='Command to run (e.g. "roslaunch pkg file.launch")')
parser.add_argument("-n", dest="n", type=int, default=0, help="Number of runs. 0 = infinite. (default 0)")
parser.add_argument("-b", dest="base_dir", default="./runs", help="Base directory for creating subfolders (default ./runs)")
parser.add_argument("-s", dest="setup", default="", help="Optional: setup.bash file to source before each run")
parser.add_argument("-i", dest="sleep", type=float, default=1.0, help="Seconds to wait between runs (default 1)")
parser.add_argument("-h", action="help", help="Show this help message and exit")
parser.add_argument("-v",dest="verbose", action="store_true", help="Verbose: print command output to console and save to logs")
args = parser.parse_args()

os.makedirs(args.base_dir, exist_ok=True)

run = 1
while (args.n == 0 and not stop_requested) or (args.n != 0 and run <= args.n and not stop_requested):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.base_dir, f"run_{run}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting run #{run} in {run_dir}", flush=True)

    cmd_to_run = args.cmd
    # If an existing setup was provided, run in a shell that sources it first
    if args.setup and os.path.isfile(args.setup):
        # Use bash -lc "source 'setup' && exec CMD" so the command inherits the environment
        cmd_shell = f"source {shlex.quote(args.setup)} && exec {cmd_to_run}"
    else:
        cmd_shell = cmd_to_run

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    with open(os.path.join(run_dir, "logs/command.log"), "w", encoding="utf-8") as clog, \
         open(os.path.join(run_dir, "logs/stdout.log"), "w", encoding="utf-8") as sout, \
         open(os.path.join(run_dir, "logs/stderr.log"), "w", encoding="utf-8") as serr:
        clog.write(f"Command: {cmd_to_run}\n")
        clog.write(f"Start: {start_time}\n")
        # Run in bash to allow expansions and for source to work
        if args.verbose:
            # Stream output live to console and save to log files
            proc = subprocess.Popen(["bash", "-lc", cmd_shell], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=run_dir)

            def stream_pipe(pipe, out_file, is_err=False):
                try:
                    for line in iter(pipe.readline, ''):
                        if not line:
                            break
                        if is_err:
                            sys.stderr.write(line)
                            sys.stderr.flush()
                        else:
                            sys.stdout.write(line)
                            sys.stdout.flush()
                        out_file.write(line)
                        out_file.flush()
                finally:
                    try:
                        pipe.close()
                    except Exception:
                        pass

            t_out = threading.Thread(target=stream_pipe, args=(proc.stdout, sout, False), daemon=True)
            t_err = threading.Thread(target=stream_pipe, args=(proc.stderr, serr, True), daemon=True)
            t_out.start()
            t_err.start()
            rc = proc.wait()
            t_out.join()
            t_err.join()
        else:
            # Save output to files only
            proc = subprocess.run(["bash", "-lc", cmd_shell], stdout=sout, stderr=serr, cwd=run_dir)
            rc = proc.returncode

        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        clog.write(f"End: {end_time}\n")
        clog.write(f"Exit code: {rc}\n")

    run += 1
    if stop_requested:
        break
    time.sleep(args.sleep)