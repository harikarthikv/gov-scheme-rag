import os
import shutil
from huggingface_hub import snapshot_download

def setup():
    # Create directories
    os.makedirs("data/bharatschemes", exist_ok=True)
    os.makedirs("data/gov_myscheme", exist_ok=True)
    os.makedirs("processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Download datasets
    print("Downloading bharatschemes...")
    snapshot_download(
        repo_id="satyajitdas/bharatschemes-v1",
        repo_type="dataset",
        local_dir="data/bharatschemes",
        local_dir_use_symlinks=False
    )
    
    print("Downloading gov_myscheme...")
    snapshot_download(
        repo_id="shrijayan/gov_myscheme",
        repo_type="dataset",
        local_dir="data/gov_myscheme",
        local_dir_use_symlinks=False
    )
    
    print("Done downloading datasets.")

if __name__ == "__main__":
    setup()
