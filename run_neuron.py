#!/usr/bin/env python3
"""
ZEUS Miner Auto-Update + Self-Heal Controller
Aligned 100% with start_miner.sh (PM2: zeus_miner)
"""

import os
import subprocess
import time
import argparse
import sys

# Restart interval in hours (self-heal)
RESTART_INTERVAL_HOURS = 3


def run_cmd(cmd):
    """Run shell command and return output."""
    return subprocess.getoutput(cmd)


def should_update(local_commit, remote_commit):
    return local_commit != remote_commit


def restart_miner():
    """Restart miner using your start_miner.sh script."""
    print("Restarting ZEUS miner with start_miner.sh ...")
    os.system("chmod +x ./start_miner.sh")
    os.system("./start_miner.sh")


def auto_update_logic():
    current_branch = run_cmd("git rev-parse --abbrev-ref HEAD")
    local_commit = run_cmd("git rev-parse HEAD")

    os.system("git fetch")
    remote_commit = run_cmd(f"git rev-parse origin/{current_branch}")

    if should_update(local_commit, remote_commit):
        print("🔄 Update detected! Local repo is behind.")
        print(f"Local:  {local_commit}")
        print(f"Remote: {remote_commit}")

        reset_cmd = f"git reset --hard {remote_commit}"
        print("Applying update...")
        os.system(reset_cmd)

        # optional setup.sh
        if os.path.exists("./setup.sh"):
            print("Running setup.sh ...")
            os.system("chmod +x ./setup.sh")
            os.system("./setup.sh")

        print("Update applied. Restarting miner...")
        restart_miner()
    else:
        print("✔ Repo is up-to-date.")


def run_loop(enable_update, enable_self_heal):
    last_restart = time.time()

    while True:
        time.sleep(120)  # check every 2 minutes

        if enable_update:
            auto_update_logic()

        if enable_self_heal:
            elapsed = time.time() - last_restart
            if elapsed >= RESTART_INTERVAL_HOURS * 3600:
                print("🛠 Self-heal duration reached — restarting miner...")
                restart_miner()
                last_restart = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ZEUS Miner with auto-update + self-heal")

    parser.add_argument("--miner", action="store_true")
    parser.add_argument("--self-heal", action="store_true")
    parser.add_argument("--no-auto-update", action="store_true")

    args = parser.parse_args()

    if not args.miner:
        print(f"Usage: python {sys.argv[0]} --miner [--self-heal] [--no-auto-update]")
        sys.exit(1)

    # Always start miner first
    restart_miner()

    # If update or self-heal enabled → run loop
    if not args.no_auto_update or args.self_heal:
        run_loop(enable_update=not args.no_auto_update, enable_self_heal=args.self_heal)
