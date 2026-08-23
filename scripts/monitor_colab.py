#!/usr/bin/env python3
"""Monitor Colab pretraining via SSH — checks every hour, downloads checkpoints."""
import subprocess, time, os, sys

SSH_HOST = "colab"
REMOTE_REPO = "/content/pg"
REMOTE_WEIGHTS = f"{REMOTE_REPO}/weights"
LOCAL_BACKUP = "/home/fedora/Documents/projects/pisto-gpt-64m/colab-backup"
CHECK_INTERVAL = 3600  # 1 hour

os.makedirs(LOCAL_BACKUP, exist_ok=True)

def ssh_run(cmd, timeout=15):
    """Run a command on Colab via SSH, return stdout or None if failed."""
    try:
        r = subprocess.run(
            ["ssh", SSH_HOST, cmd],
            capture_output=True, text=True, timeout=timeout
        )
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None

def check_status():
    """Check if training is alive and get last output."""
    # Check process
    out = ssh_run("pgrep -f pretrain.py || echo DEAD")
    alive = out and "DEAD" not in out.strip()

    # Get last training line
    r = ssh_run(f"tail -8 {REMOTE_REPO}/pretrain.out 2>/dev/null")
    last_lines = r.strip().split("\n") if r else []

    # Check session
    sess = subprocess.run(["colab", "sessions"], capture_output=True, text=True)
    session_alive = "pisto" in sess.stdout

    return alive, session_alive, last_lines

def download_checkpoints():
    """Download all checkpoints via scp."""
    try:
        r = subprocess.run(
            ["scp", f"{SSH_HOST}:{REMOTE_WEIGHTS}/*", LOCAL_BACKUP + "/"],
            capture_output=True, text=True, timeout=120
        )
        return r.returncode == 0
    except Exception:
        return False

def main():
    print(f"Monitor started. Checking every {CHECK_INTERVAL//60} min. Backups -> {LOCAL_BACKUP}")
    start = time.time()
    while True:
        elapsed_h = (time.time() - start) / 3600
        alive, session_alive, lines = check_status()

        print(f"\n[{elapsed_h:.1f}h] alive={alive} session={session_alive}")
        for line in lines[-3:]:
            print(f"   {line[:100]}")

        if not alive or not session_alive:
            print("⚠ TRAINING OR SESSION DIED")
            ok = download_checkpoints()
            print(f"   Last checkpoint download: {'✓' if ok else '✗'}")
            if not session_alive:
                print("   VM gone — cannot resume via SSH. Run notebook in browser to resume from Drive.")
                break
        else:
            # Download periodically as backup
            ok = download_checkpoints()
            print(f"   Checkpoint backup: {'✓' if ok else '✗ (will retry next check)'}")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped by user.")
