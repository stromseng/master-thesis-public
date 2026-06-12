# IDUN Setup Guide

## 1. Connect to IDUN

```bash
ssh idun
```


## 2. Set up SSH key for GitHub

Generate an SSH key on IDUN:
```bash
ssh-keygen -t ed25519 -C "your-email@stud.ntnu.no"
```

Press Enter to accept the default file location and optionally set a passphrase.

Display your public key:
```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the output and add it to GitHub:
1. Go to [github.com/settings/keys](https://github.com/settings/keys)
2. Click "New SSH key"
3. Paste your key and save

Test the connection:
```bash
ssh -T git@github.com
```

## 3. Clone the repository

For the IDUN CLI to work properly, clone the repo into `/cluster/home/<USERNAME>/repos/master-thesis`:

```bash
mkdir -p ~/repos
cd ~/repos
git clone git@github.com:stromseng/master-thesis.git
```

Configure Git:
```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@stud.ntnu.no"
```

## 4. Set up UV

```bash
cd ~/repos/master-thesis/code/python
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## 5. Run conversion script

```bash
cd ~/repos/master-thesis/code/python
uv run python -m scripts.idun.convert_pdfs
```
