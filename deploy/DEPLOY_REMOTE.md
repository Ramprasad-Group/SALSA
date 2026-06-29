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

Georgia Tech outbound email:

- Set `SALSA_SMTP_HOST=outbound.mail.gatech.edu`, `SALSA_SMTP_PORT=25`, and `SALSA_SMTP_TLS=0`
- Set `SALSA_SMTP_LOCAL_HOSTNAME` to the hostname registered on Georgia Tech's Authorized Senders list
- Set `SALSA_SMTP_FROM` to an authorized sender address for the relay, such as `no-reply@polymergenome.org`
- Campus relay connections do not use SMTP username/password authentication
- Request Authorized Sender access from Georgia Tech OIT if the relay rejects the host
- Password-reset messages contain only the one-time token, not a reset URL
- Public account creation produces a pending request; approved admins receive the submitted name, username, email, institution, and request time by email
- Pending users cannot sign in until an admin approves them from the SALSA Admin page

Publisher full-text access:

- SALSA does not use Georgia Tech IP access, browser cookies, or publisher landing pages
- DOI redirects provide location metadata only; they do not authorize automated full-text downloads
- Elsevier requires `ELSEVIER_API_KEY`
- Wiley requires `WILEY_API_KEY`
- Springer Nature open-access JATS requires `SPRINGER_NATURE_API_KEY`
- RSC requires an RSC-issued `RSC_API_KEY`; SALSA uses the RSC publications endpoint directly
- IEEE Open Access or licensed full text requires an approved `IEEE_API_KEY`
- PubMed Central and PLOS open-access retrieval require no key; `NCBI_API_KEY` is optional for higher request limits
- APS open-access articles are retrieved through APS Harvest without a user credential
- IOP articles are discovered through Crossref text-mining links without a user credential, but IOP blocks systematic platform downloads and directs TDM users to arrange SFTP delivery through `contentsupport@ioppublishing.org`
- ACS, ECS, AIP, Hindawi, and Informa papers must currently be supplied as authorized PDF uploads

For `polymergenome.storm.gatech.edu`, the authorization request should identify:

- Hostname: `polymergenome.storm.gatech.edu`
- Canonical hostname: `mselweb0001.storm.gatech.edu`
- Source IP: `143.215.76.191`
- Relay: `outbound.mail.gatech.edu`, port `25`
- Sender: `no-reply@polymergenome.org`
- Rejection when unauthorized: `550 5.7.1 Your access to submit messages to this e-mail system has been rejected.`
- A successful SMTP `EHLO`, `MAIL FROM`, and `RCPT TO` does not prove authorization; the relay may reject the message at the final `DATA` step

### 4. Test the app manually

```bash
cd /data/appls/machine_learning/SALSA
conda activate SALSA
set -a
source .env.production
set +a
python -m streamlit run src/ui/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.fileWatcherType auto
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

Production disables Streamlit source-file watching in
`deploy/systemd/salsa.service` so edits to this working tree do not reload the
production app and kill in-process runs. After testing changes, restart
production deliberately:

```bash
sudo systemctl restart salsa
```

Logs:

```bash
journalctl -u salsa -f
```

### 5a. Run a separate development app

Use the development service when changing or testing code. It listens on port
`8602` by default and stores data under `src/ui_runs_dev`, separate from
production workspaces and auth.

```bash
cp deploy/env/salsa-dev.env.example .env.development
sudo cp deploy/systemd/salsa-dev.service /etc/systemd/system/salsa-dev.service
sudo systemctl daemon-reload
sudo systemctl enable --now salsa-dev
sudo systemctl status salsa-dev
```

Open the development app at `http://<server-host>:8602` or proxy it separately.
Because the dev service uses Streamlit file watching, source edits reload only
the development app. They should not interrupt production runs.

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

#### Docker-managed certificate renewal

When Nginx and Certbot run from `/root/pgV2/docker-compose.yml`, the host's
standard `certbot.timer` does not renew certificates in the Compose-mounted
`/root/pgV2/certbot/conf` directory. Install the dedicated timer instead:

```bash
sudo cp deploy/systemd/pgv2-certbot-renew.service /etc/systemd/system/
sudo cp deploy/systemd/pgv2-certbot-renew.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pgv2-certbot-renew.timer
```

The timer checks twice daily. Certbot renews only certificates near expiry;
after a successful check, the service validates and reloads the Docker Nginx
container so renewed certificates take effect.

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
