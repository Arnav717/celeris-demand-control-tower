"""FastAPI backend for the Celeris planning control tower.

The UI (static/index.html) is a single page; every view calls one JSON endpoint with the
shared filters (family, sku, region, channel, customer) as repeated query parameters.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from celeris.dashboard.data import make_filters, store

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Celeris Control Tower", docs_url="/api/docs", redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

Multi = Annotated[list[str] | None, Query()]


class Selection:
    """The shared filters, parsed once per request."""

    def __init__(self, family: Multi = None, sku: Multi = None, region: Multi = None,
                 channel: Multi = None, customer: Multi = None) -> None:
        self.F = make_filters(family=family, sku=sku, region=region, channel=channel, customer=customer)


Sel = Annotated[Selection, Depends()]


@app.on_event("startup")
def _warm() -> None:
    store()


def _versioned(match: re.Match) -> str:
    """Append the file's mtime so browsers fetch an asset again after it changes."""
    path = match.group(1)
    f = STATIC / path
    v = int(f.stat().st_mtime) if f.exists() else 0
    return f'"/static/{path}?v={v}"'


@app.get("/", include_in_schema=False)
def index() -> HTMLResponse:
    html = (STATIC / "index.html").read_text()
    html = re.sub(r'"/static/([^"?]+)"', _versioned, html)
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.get("/api/meta")
def meta() -> JSONResponse:
    return JSONResponse(store().meta())


@app.get("/api/overview")
def overview(sel: Sel) -> JSONResponse:
    return JSONResponse(store().overview(sel.F))


@app.get("/api/health")
def health(sel: Sel) -> JSONResponse:
    return JSONResponse(store().health(sel.F))


@app.get("/api/explorer")
def explorer(sel: Sel) -> JSONResponse:
    return JSONResponse(store().explorer(sel.F))


@app.get("/api/forecast")
def forecast(sel: Sel) -> JSONResponse:
    return JSONResponse(store().forecast(sel.F))


@app.get("/api/lifecycle")
def lifecycle(sel: Sel) -> JSONResponse:
    return JSONResponse(store().lifecycle(sel.F))


@app.get("/api/stockouts")
def stockouts(sel: Sel) -> JSONResponse:
    return JSONResponse(store().stockouts(sel.F))


@app.get("/api/anomalies")
def anomalies(sel: Sel, category: str = "", atype: str = "") -> JSONResponse:
    return JSONResponse(store().anomalies(sel.F, category, atype))


@app.get("/api/drivers")
def drivers(family: str = "", indicator: str = "") -> JSONResponse:
    return JSONResponse(store().drivers(family, indicator))


@app.get("/api/whatif")
def whatif(sel: Sel, indicator: str = "ISM_Manufacturing_PMI", shock: float = -3.0) -> JSONResponse:
    if not -50 <= shock <= 50:
        raise HTTPException(400, "shock out of range")
    if indicator not in store().macro.columns:
        raise HTTPException(400, f"unknown indicator {indicator!r}")
    return JSONResponse(store().whatif(sel.F, indicator, round(shock, 2)))


@app.get("/api/procurement")
def procurement(sel: Sel, window: int = 6) -> JSONResponse:
    return JSONResponse(store().procurement(sel.F, max(1, min(19, window))))


@app.get("/api/exceptions")
def exceptions(sel: Sel, flag: str = "") -> JSONResponse:
    return JSONResponse(store().exceptions(sel.F, flag))


@app.get("/api/download/{kind}.csv")
def download(kind: str, sel: Sel, window: int = 6) -> StreamingResponse:
    try:
        df = store().download(kind, sel.F, max(1, min(19, window)))
    except KeyError:
        raise HTTPException(404, f"unknown download {kind!r}") from None
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="celeris_{kind}.csv"'})


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    return {"ok": True}
