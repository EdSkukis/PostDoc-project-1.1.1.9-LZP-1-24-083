import time, shutil, os
from pathlib import Path

STORAGE = Path("./storage")
def clean():
    for f in STORAGE.glob("*"):
        if f.is_dir() and (time.time() - f.stat().st_mtime > 86400):
            shutil.rmtree(f)
if __name__ == "__main__": clean()