# Vendored third-party code

## `streamlit_drawable_canvas`

Vendored from [`streamlit-drawable-canvas`](https://github.com/andfanilo/streamlit-drawable-canvas)
**0.9.3**, with SALSA-specific patches to the compiled frontend bundle.

### Why it is vendored

The patches live in a webpack-minified bundle
(`frontend/build/static/js/main.80185090.salsa-patched.js`). Previously they were
applied by hand directly inside `site-packages`, so any `make setup`,
`conda env update`, or `pip install --force-reinstall` silently reverted the
component to stock upstream and figure canvases stopped rendering. Vendoring makes
the patched build part of the repo, so every checkout and every deployment behaves
identically.

`index.html` loads `main.80185090.salsa-patched.js` instead of the stock
`main.80185090.chunk.js`. Source maps are not shipped (they are ~4 MB and are never
served).

### What was changed relative to upstream 0.9.3

All changes are inside the background-image `useEffect` in the main bundle:

1. **Raw `drawImage` instead of `fabric.setBackgroundImage`.** Upstream loads the
   background through Fabric's async image loader onto the drawing canvas; that
   left the background blank on remount. The patched build paints the image
   directly onto the dedicated `#backgroundimage-canvas` context and additionally
   sets it as a CSS `background-image` on `#canvas` as a belt-and-braces fallback.
2. **`streamlitUrl` origin resolution.** Relative media URLs are resolved against
   the origin Streamlit passes to the component iframe, so the canvas works behind
   the Nginx reverse proxy in `deploy/nginx/salsa.conf`.
3. **Absolute/`data:` URL passthrough** *(SALSA)*. The origin from (2) is no longer
   prefixed onto URLs that are already absolute. SALSA now sends backgrounds as
   `data:` URIs, and `"https://host" + "data:image/..."` would be an invalid URL.
4. **`onerror` handling** *(SALSA)*. Upstream sets `img.src` with no error path, so
   a failed background load left a silently blank canvas — the single most common
   "the figure didn't render" symptom. The patched build retries once with a
   cache-busting query parameter (relative URLs only), then paints a visible
   "Background image failed to load" placeholder.

Patches (1) and (2) predate this vendoring and are baked into the committed
bundle; they were made by hand against the minified source and are not scripted.
Patches (3) and (4) are reproducible via
[`src/setup-scripts/patch_drawable_canvas.py`](../../setup-scripts/patch_drawable_canvas.py),
which is idempotent and also has a `--check` mode used by the test suite.

### Regenerating from upstream

```bash
pip download streamlit-drawable-canvas==0.9.3 --no-deps -d /tmp/sdc
# unpack, copy frontend/build/ here excluding *.map, reapply patches 1-2 by hand,
# rename the main chunk to *.salsa-patched.js and update index.html, then:
python src/setup-scripts/patch_drawable_canvas.py
```

Upstream is licensed MIT; see the upstream repository for the full license text.
