## Remote hosting for SALSA

This app is a Streamlit service. On a remote Linux machine, the simplest production-friendly setup is:

1. Install the app under `/data/appls/machine_learning/SALSA`
2. Create the Conda environment and run the existing setup script
3. Run Streamlit under `systemd`
4. Expose it either:
   - directly on `http://<host>:8501`, or
   - behind Nginx on a friendlier URL such as `https://salsa.example.org`

### 1. Copy the code to the target location

```bash
mkdir -p /data/appls/machine_learning
cd /data/appls/machine_learning
git clone --recurse-submodules <your-repo-url> SALSA
cd SALSA
```

If you are moving the current checkout on this machine instead of cloning:

```bash
rsync -a /data/wschertzer/SALSA/ /data/appls/machine_learning/SALSA/
```

### 2. Create the environment

```bash
cd /data/appls/machine_learning/SALSA
conda env create -f environment.yml || conda env update -f environment.yml --prune
conda activate SALSA
bash src/setup-scripts/setup.sh
```

If your checkpoints were pulled with Git LFS, make sure the model files are present under `huggingface/` and `src/huggingface/`.

### 3. Create an environment file

Use [`deploy/env/salsa.env.example`](./env/salsa.env.example) as a template:

```bash
cp deploy/env/salsa.env.example /data/appls/machine_learning/SALSA/.env.production
```

Then edit values for your server hostname and LLM provider.

Important remote-access note:

- If users will use the chemical drawing tool, set `SALSA_JSME_BIND_HOST=0.0.0.0`
- Set `SALSA_JSME_PUBLIC_URL` to a browser-reachable URL
- If you proxy through Nginx, use a subpath such as `https://your-host.example.org/jsme-assets`

### 4. Test the app manually

```bash
cd /data/appls/machine_learning/SALSA
conda activate SALSA
set -a
source .env.production
set +a
python -m streamlit run src/ui/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

If that works, browse to:

- `http://<server-host>:8501`

### 5. Run it as a systemd service

Copy the unit file and adjust `User`, `Group`, and the Conda path:

```bash
sudo cp deploy/systemd/salsa.service /etc/systemd/system/salsa.service
sudo systemctl daemon-reload
sudo systemctl enable --now salsa
sudo systemctl status salsa
```

Logs:

```bash
journalctl -u salsa -f
```

### 6. Expose it through a URL

#### Option A: direct URL on port 8501

This is the fastest option. Point DNS at the server and open the firewall:

- `http://your-host.example.org:8501`

If you want HTTPS or a cleaner URL without a port, use Option B.

#### Option B: Nginx reverse proxy

Use [`deploy/nginx/salsa.conf`](./nginx/salsa.conf) as a starting point.

That config proxies:

- `/` to the Streamlit app on `127.0.0.1:8501`
- `/jsme-assets/` to the helper server on `127.0.0.1:8765`

After placing the config in Nginx, reload:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

Then the app can live at:

- `https://salsa.example.org`

and the chemical editor helper assets can live at:

- `https://salsa.example.org/jsme-assets`

### 7. Firewall and DNS

At minimum, allow inbound traffic to:

- `8501/tcp` if exposing Streamlit directly
- `80/tcp` and `443/tcp` if using Nginx
- `8765/tcp` only if you expose the JSME helper directly instead of reverse proxying it

### 8. Recommended production notes

- Keep Streamlit bound to `127.0.0.1` if you are using Nginx
- Bind Streamlit to `0.0.0.0` only when exposing port `8501` directly
- Prefer OpenAI or a server-local Ollama endpoint configured through `.env.production`
- Keep app data and workspaces on a disk with enough space for papers, extracted figures, and model artifacts
- If multiple users will run jobs, consider a dedicated workspace root such as `/data/appls/machine_learning/SALSA/workspaces`

### Service verification checklist

```bash
systemctl status salsa
ss -ltnp | grep 8501
ss -ltnp | grep 8765
curl -I http://127.0.0.1:8501
```
