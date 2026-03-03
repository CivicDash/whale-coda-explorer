"""Telecharge les fichiers WAV du dataset DSWP depuis HuggingFace en batch."""
import os
import time
import urllib.request
import sys

DSWP_DIR = os.path.join(os.path.dirname(__file__), "data", "dswp")
BASE_URL = "https://huggingface.co/datasets/orrp/DSWP/resolve/main"

def get_wav_files():
    files = sorted([f for f in os.listdir(DSWP_DIR) if f.endswith('.wav')])
    return files

def is_lfs_pointer(filepath):
    try:
        with open(filepath, 'rb') as f:
            header = f.read(50)
        return b'git-lfs' in header
    except Exception:
        return True

def download_file(filename, retries=3):
    filepath = os.path.join(DSWP_DIR, filename)
    url = f"{BASE_URL}/{filename}"
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, filepath)
            size = os.path.getsize(filepath)
            if size < 1000 or is_lfs_pointer(filepath):
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                return False
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                print(f"  Erreur {filename}: {e}")
                return False
    return False

def main():
    files = get_wav_files()
    need_download = [f for f in files if is_lfs_pointer(os.path.join(DSWP_DIR, f))]
    
    print(f"Dataset DSWP: {len(files)} fichiers total")
    print(f"A telecharger: {len(need_download)} fichiers")
    
    if not need_download:
        print("Tout est deja telecharge !")
        return
    
    success = 0
    errors = 0
    batch_size = 20
    
    for i, filename in enumerate(need_download):
        downloaded = download_file(filename)
        if downloaded:
            success += 1
        else:
            errors += 1
        
        if (i + 1) % 10 == 0:
            pct = (i + 1) / len(need_download) * 100
            print(f"  [{pct:.0f}%] {i+1}/{len(need_download)} — {success} OK, {errors} erreurs")
        
        if (i + 1) % batch_size == 0:
            time.sleep(2)
    
    print(f"\nTermine: {success} telecharges, {errors} erreurs sur {len(need_download)}")

if __name__ == "__main__":
    main()
