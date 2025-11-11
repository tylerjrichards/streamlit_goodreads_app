import subprocess
import sys


def test_check_install_runs_ok():
    # Run the smoke script as a subprocess to ensure it exits 0 in CI after deps installed
    res = subprocess.run([sys.executable, "check_install.py"], capture_output=True)
    out = res.stdout.decode(errors="ignore")
    err = res.stderr.decode(errors="ignore")
    assert res.returncode == 0, f"check_install failed (code={res.returncode})\nSTDOUT:\n{out}\nSTDERR:\n{err}"
