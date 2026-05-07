## Remote hosting for AEM_OCR

This app is a Streamlit service. On a remote Linux machine, the simplest production-friendly setup is:

1. Install the app under `/data/appls/machine_learning/AEM_OCR`
2. Create the Conda environment and run the existing setup script
3. Run Streamlit under `systemd`
4. Expose it either:
   - directly on `http://<host>:8501`, or
   - behind Nginx on a friendlier URL such as `https://aem-ocr.example.org`

### 1. Copy the code to the target location

```bash
mkdir -p /data/appls/machine_learning
cd /data/appls/machine_learning
git clone --recurse-submodules <your-repo-url> AEM_OCR
cd AEM_OCR
```

If you are moving the current checkout on this machine instead of cloning:

```bash
rsync -a /data/wschertzer/AEM_OCR/ /data/appls/machine_learning/AEM_OCR/
```

### 2. Create the environment

```bash
cd /data/appls/machine_learning/AEM_OCR
conda env create -f environment.yml || conda env update -f environment.yml --prune
conda activate AEM_OCR
bash src/setup-scripts/setup.sh
```

If your checkpoints were pulled with Git LFS, make sure the model files are present under `huggingface/` and `src/huggingface/`.

### 3. Create an environment file

Use [`deploy/env/aem-ocr.env.example`](./env/aem-ocr.env.example) as a template:

```bash
cp deploy/env/aem-ocr.env.example /data/appls/machine_learning/AEM_OCR/.env.production
```

Then edit values for your server hostname and LLM provider.

Important remote-access note:

- If users will use the chemical drawing tool, set `AEM_OCR_JSME_BIND_HOST=0.0.0.0`
- Set `AEM_OCR_JSME_PUBLIC_URL` to a browser-reachable URL
- If you proxy through Nginx, use a subpath such as `https://your-host.example.org/jsme-assets`

### 4. Test the app manually

```bash
cd /data/appls/machine_learning/AEM_OCR
conda activate AEM_OCR
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
sudo cp deploy/systemd/aem-ocr.service /etc/systemd/system/aem-ocr.service
sudo systemctl daemon-reload
sudo systemctl enable --now aem-ocr
sudo systemctl status aem-ocr
```

Logs:

```bash
journalctl -u aem-ocr -f
```

### 6. Expose it through a URL

#### Option A: direct URL on port 8501

This is the fastest option. Point DNS at the server and open the firewall:

- `http://your-host.example.org:8501`

If you want HTTPS or a cleaner URL without a port, use Option B.

#### Option B: Nginx reverse proxy

Use [`deploy/nginx/aem-ocr.conf`](./nginx/aem-ocr.conf) as a starting point.

That config proxies:

- `/` to the Streamlit app on `127.0.0.1:8501`
- `/jsme-assets/` to the helper server on `127.0.0.1:8765`

After placing the config in Nginx, reload:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

Then the app can live at:

- `https://aem-ocr.example.org`

and the chemical editor helper assets can live at:

- `https://aem-ocr.example.org/jsme-assets`

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
- If multiple users will run jobs, consider a dedicated workspace root such as `/data/appls/machine_learning/AEM_OCR/workspaces`

### Service verification checklist

```bash
systemctl status aem-ocr
ss -ltnp | grep 8501
ss -ltnp | grep 8765
curl -I http://127.0.0.1:8501
```
