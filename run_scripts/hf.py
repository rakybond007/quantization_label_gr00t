from huggingface_hub import HfApi, Repository

# 1. Log in (this will open a browser prompt on first run)
api = HfApi()
# api.login()  

# 2. Create the dataset repo (skip if it already exists)
repo_id = "kimtaey/libero-gr00t-delta"
try:
    api.create_repo(repo_id=repo_id, repo_type="dataset")
except Exception:
    pass  # already exists

# 3. Clone it locally and copy files
repo = Repository(local_dir="libero-gr00t-delta", clone_from=repo_id, repo_type="dataset")
repo.git_pull()
repo.lfs_track(["*.bin", "*.pt", "*.jpg", "*.png"])  # adjust for your filetypes

import shutil, os
src = "/virtual_lab/sjw_alinlab/taeyoung/LVLA/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"
for fn in os.listdir(src):
    shutil.copy(os.path.join(src, fn), "libero-gr00t-delta/")

# 4. Push back up
repo.git_add(auto_lfs_track=True)
repo.git_commit("Add local cache files")
repo.git_push()