"""Upload this repo to the Hugging Face Hub as a model repo.

Usage:
    export HF_TOKEN=hf_xxx            # a WRITE token from huggingface.co/settings/tokens
    python scripts/upload_to_hf.py --repo-id <username>/Elysium-X-20
Nothing is uploaded unless you run this yourself.
"""
import argparse, os, sys
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo-id", required=True, help="e.g. itsppm76/Elysium-X-20")
ap.add_argument("--private", action="store_true")
args = ap.parse_args()
token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("Set HF_TOKEN to a write token first.")
api = HfApi(token=token)
api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
api.upload_folder(folder_path=root, repo_id=args.repo_id, repo_type="model",
                  ignore_patterns=[".git*", "__pycache__", "*.egg-info", ".pytest_cache", "dist", "build"],
                  commit_message="Elysium X 20 v0.1.0")
print(f"Done: https://huggingface.co/{args.repo_id}")
