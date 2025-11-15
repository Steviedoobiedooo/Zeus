#!/usr/bin/env python3
"""
ZEUS Miner Auto-Update + Health-Based Self-Heal
Aligned with start_miner.sh (PM2 process: zeus_miner)
"""

import os
import subprocess
import time
import argparse
import sys
import json

MINER_PROCESS_NAME = "zeus_miner"


def run_cmd(cmd):
    return subprocess.getoutput(cmd)


def restart_miner():
    print("🔁 Restarting ZEUS miner with start_miner.sh ...")
    os.system("chmod +x ./start_miner.sh")
    os.system("./start_miner.sh")


def is_miner_healthy():
    """
    Check PM2 status of zeus_miner.
    Healthy = present and status == 'online'.
    """
    try:
        out = run_cmd("pm2 jlist")
        data = json.loads(out)
        for proc in data:
            if proc.get("name") == MINER_PROCESS_NAME:
                status = proc.get("pm2_env", {}).get("status")
                print(f"PM2 status for {MINER_PROCESS_NAME}: {status}")
                return status == "online"
        print(f"{MINER_PROCESS_NAME} not found in PM2 list.")
        return False
    except Exception as e:
        print(f"Error checking PM2 status: {e}")
        # if we can't even parse status, assume unhealthy
        return False


def auto_update_if_needed():
    current_branch = run_cmd("git rev-parse --abbrev-ref HEAD")
    local_commit = run_cmd("git rev-parse HEAD")

    os.system("git fetch")
    remote_commit = run_cmd(f"git rev-parse origin/{current_branch}")

    if local_commit != remote_commit:
        print("🔄 Update detected! Local repo is behind.")
        print(f"Local:  {local_commit}")
        print(f"Remote: {remote_commit}")

        os.system(f"git reset --hard {remote_commit}")

        if os.path.exists("./setup.sh"):
            print("Running setup.sh ...")
            os.system("chmod +x ./setup.sh")
            os.system("./setup.sh")

        print("Update applied. Restarting miner...")
        restart_miner()
    else:
        print("✔ Repo is up-to-date.")


def run_loop(enable_update, enable_self_heal):
    while True:
        time.sleep(120)  # every 2 minutes

        if enable_update:
            auto_update_if_needed()

        if enable_self_heal:
            if not is_miner_healthy():
                print("❗ Miner unhealthy or not online in PM2. Triggering restart...")
                restart_miner()
            else:
                print("✅ Miner healthy, no restart needed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ZEUS Miner with auto-update + health-based self-heal"
    )

    parser.add_argument("--miner", action="store_true")
    parser.add_argument("--self-heal", action="store_true")
    parser.add_argument("--no-auto-update", action="store_true")

    args = parser.parse_args()

    if not args.miner:
        print(f"Usage: python {sys.argv[0]} --miner [--self-heal] [--no-auto-update]")
        sys.exit(1)

    # Start miner once
    restart_miner()

    # Loop only if at least one feature active
    if not args.no_auto-update or args.self_heal:
        run_loop(enable_update=not args.no_auto_update, enable_self_heal=args.self_heal)
