"""
Step 4: Download pretrained Common Crawl Hindi FastText word vectors.
Source: https://fasttext.cc/docs/en/crawl-vectors.html
Downloads the text-format vectors (.vec.gz) for comparison.
"""
import sys, os, gzip, shutil, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *


def download_with_progress(url, dest, label="Downloading"):
    """Download a file with progress display."""
    import time
    
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            mb_done = count * block_size / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  {label}: {pct}% ({mb_done:.0f}/{mb_total:.0f} MB)", end="", flush=True)
        else:
            mb_done = count * block_size / (1024 * 1024)
            print(f"\r  {label}: {mb_done:.0f} MB downloaded", end="", flush=True)
    
    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()  # newline after progress


def download_pretrained():
    print("[Step 4] Downloading pretrained Common Crawl Hindi FastText vectors...")
    print(f"  Source: https://fasttext.cc/docs/en/crawl-vectors.html")

    if os.path.exists(PRETRAINED_VEC_PATH):
        size_mb = os.path.getsize(PRETRAINED_VEC_PATH) / (1024 * 1024)
        print(f"  Already exists: {PRETRAINED_VEC_PATH} ({size_mb:.1f} MB)")
        return PRETRAINED_VEC_PATH

    # Download .vec.gz file (text format - lighter than binary)
    if not os.path.exists(PRETRAINED_VEC_GZ):
        print(f"  Downloading from {PRETRAINED_VEC_URL}...")
        print("  (This is ~1.2 GB compressed, ~4 GB uncompressed)")
        print("  Please be patient...")
        download_with_progress(PRETRAINED_VEC_URL, PRETRAINED_VEC_GZ, "cc.hi.300.vec.gz")
        size_mb = os.path.getsize(PRETRAINED_VEC_GZ) / (1024 * 1024)
        print(f"  Downloaded {size_mb:.1f} MB")
    else:
        print(f"  .gz already downloaded: {PRETRAINED_VEC_GZ}")

    # Extract
    print("  Extracting (this may take a few minutes)...")
    with gzip.open(PRETRAINED_VEC_GZ, 'rb') as f_in:
        with open(PRETRAINED_VEC_PATH, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    size_mb = os.path.getsize(PRETRAINED_VEC_PATH) / (1024 * 1024)
    print(f"  Extracted to {PRETRAINED_VEC_PATH} ({size_mb:.1f} MB)")

    # Optionally remove .gz to save space
    # os.remove(PRETRAINED_VEC_GZ)

    return PRETRAINED_VEC_PATH


if __name__ == "__main__":
    download_pretrained()
