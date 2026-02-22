from huggingface_hub import snapshot_download

local_dir = snapshot_download(repo_id="facebook/bart-large-cnn")
print(local_dir)  # folder with pytorch_model.bin, config.json, etc.
