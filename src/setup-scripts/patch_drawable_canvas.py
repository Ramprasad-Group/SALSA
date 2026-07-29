"""Reapply SALSA's `data:`-URL and `onerror` patches to the vendored drawable-canvas bundle.

The vendored bundle at
``src/ui/vendor/streamlit_drawable_canvas/frontend/build/static/js/main.*.salsa-patched.js``
is already patched, so running this script normally reports "already patched" and
exits 0.  It exists so the patch is reproducible if the bundle is ever regenerated
from upstream.

See ``src/ui/vendor/README.md`` for the full list of deviations from upstream.

Usage:
    python src/setup-scripts/patch_drawable_canvas.py [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BUNDLE_GLOB = "main.*.salsa-patched.js"
BUNDLE_DIR = (
    Path(__file__).resolve().parents[1]
    / "ui"
    / "vendor"
    / "streamlit_drawable_canvas"
    / "frontend"
    / "build"
    / "static"
    / "js"
)

# The tail of upstream's background-image effect: resolve an origin from the
# `streamlitUrl` query param, then unconditionally prefix it onto the URL.
UNPATCHED_TAIL = (
    'var n=null!==(t=function(){var t=new URLSearchParams(window.location.search)'
    '.get("streamlitUrl");if(null==t)return null;try{return new URL(t).origin}'
    'catch(e){return null}}())&&void 0!==t?t:"";e.src=n+h'
)

# SALSA replacement: skip the origin prefix for absolute/data: URLs, and add the
# error path upstream is missing (a failed load otherwise leaves a blank canvas).
PATCHED_TAIL = (
    'var n=null!==(t=function(){var t=new URLSearchParams(window.location.search)'
    '.get("streamlitUrl");if(null==t)return null;try{return new URL(t).origin}'
    'catch(e){return null}}())&&void 0!==t?t:"";'
    'var salsaAbs=0===h.indexOf("data:")||0===h.indexOf("http://")||0===h.indexOf("https://"),'
    'salsaSrc=salsaAbs?h:n+h,salsaRetried=!1;'
    'e.onerror=function(){'
    'if(!salsaRetried&&!salsaAbs){salsaRetried=!0,e.src=salsaSrc+(salsaSrc.indexOf("?")>-1?"&":"?")+"salsaRetry=1";return}'
    'try{var c=R.getContext(),v=c.canvas;'
    'c.clearRect(0,0,v.width,v.height),'
    'c.fillStyle="#f6f6f6",c.fillRect(0,0,v.width,v.height),'
    'c.strokeStyle="#c8c8c8",c.strokeRect(.5,.5,v.width-1,v.height-1),'
    'c.fillStyle="#b00020",c.font="13px sans-serif",c.textAlign="center",c.textBaseline="middle",'
    'c.fillText("Background image failed to load",v.width/2,v.height/2),'
    'R.renderAll()}catch(err){}};'
    'e.src=salsaSrc'
)


def find_bundle() -> Path:
    matches = sorted(BUNDLE_DIR.glob(BUNDLE_GLOB))
    if not matches:
        raise SystemExit(f"No {BUNDLE_GLOB} found under {BUNDLE_DIR}")
    if len(matches) > 1:
        raise SystemExit(f"Expected one bundle, found {len(matches)}: {matches}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the bundle is not patched, instead of patching it.",
    )
    args = parser.parse_args()

    bundle = find_bundle()
    source = bundle.read_text()

    if PATCHED_TAIL in source:
        print(f"already patched: {bundle}")
        return 0

    if args.check:
        print(f"NOT patched: {bundle}", file=sys.stderr)
        return 1

    if source.count(UNPATCHED_TAIL) != 1:
        raise SystemExit(
            f"Could not locate the upstream background-image loader in {bundle}. "
            "The bundle likely changed; update UNPATCHED_TAIL in this script."
        )

    bundle.write_text(source.replace(UNPATCHED_TAIL, PATCHED_TAIL))
    print(f"patched: {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
