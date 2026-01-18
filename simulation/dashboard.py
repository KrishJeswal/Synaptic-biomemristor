from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import os
import re
import socketserver
import time
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_title(s: str) -> str:
    return html.escape(s, quote=True)


def _fmt_value(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) >= 1000 or (abs(v) > 0 and abs(v) < 0.001):
            return f"{v:.4e}"
        return f"{v:.6g}"
    return html.escape(str(v))


def _df_preview_html(csv_path: Path, max_rows: int = 12) -> str:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"<div class='note error'>Could not read {html.escape(str(csv_path))}: {html.escape(str(e))}</div>"

    if df.empty:
        return "<div class='note'>No rows.</div>"

    df2 = df.head(max_rows)
    return df2.to_html(index=False, classes="table", border=0, escape=True)


def _img_tag(img_path: Path, embed: bool) -> str:
    if not img_path.exists():
        return f"<div class='note error'>Missing image: {html.escape(str(img_path))}</div>"

    if not embed:
        rel = img_path.as_posix()
        return f"<img class='plot' src='{html.escape(rel)}' alt='{html.escape(img_path.name)}' />"

    mime, _ = mimetypes.guess_type(str(img_path))
    if not mime:
        mime = "image/png"
    b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return f"<img class='plot' src='data:{mime};base64,{b64}' alt='{html.escape(img_path.name)}' />"


def _metrics_table_html(metrics: Dict[str, Any], title: str) -> str:
    if not metrics:
        return "<div class='note'>No metrics found.</div>"

    rows = []
    for k in sorted(metrics.keys()):
        rows.append(f"<tr><td class='k'>{_safe_title(k)}</td><td class='v'>{_fmt_value(metrics[k])}</td></tr>")

    return f"""
    <div class="card">
      <div class="card-title">{_safe_title(title)}</div>
      <table class="kv">
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    """


def _collect_files(data_dir: Path) -> Dict[str, List[Path]]:
    plots_dir = data_dir / "plots"
    raw_dir = data_dir / "raw"
    proc_dir = data_dir / "processed"

    out = {
        "plots": sorted(plots_dir.glob("*.png")),
        "raw_csv": sorted(raw_dir.glob("*.csv")),
        "proc_json": sorted(proc_dir.glob("*.json")),
        "proc_csv": sorted(proc_dir.glob("*.csv")),
    }
    return out


def _pick_latest_by_mtime(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _group_assets(files: Dict[str, List[Path]]) -> Dict[str, Any]:
    plots = files["plots"]
    proc_json = files["proc_json"]
    raw_csv = files["raw_csv"]

    def by_name(pattern: str, items: List[Path]) -> List[Path]:
        rx = re.compile(pattern)
        return [p for p in items if rx.search(p.name)]

    grouped = {
        "iv": {
            "plots": by_name(r"^iv_curve\.png$", plots),
            "hysteresis": by_name(r"^hysteresis_.*\.png$", plots),
            "raw": by_name(r"^iv_.*\.csv$", raw_csv),
            "metrics": by_name(r"^iv_metrics_.*\.json$", proc_json),
        },
        "pulse": {
            "plots": by_name(r"^pulse_.*\.png$", plots),
            "raw": by_name(r"^pulse_.*\.csv$", raw_csv),
            "metrics": [],  # none in v3 (fine)
        },
        "endurance": {
            "plots": by_name(r"^endurance_.*\.png$", plots),
            "raw": by_name(r"^endurance_.*\.csv$", raw_csv),
            "metrics": by_name(r"^endurance_metrics_.*\.json$", proc_json),
        },
        "retention": {
            "plots": by_name(r"^retention_.*\.png$", plots),
            "raw": by_name(r"^retention_.*\.csv$", raw_csv),
            "metrics": by_name(r"^retention_metrics_.*\.json$", proc_json),
        },
    }

    for k in grouped:
        grouped[k]["latest_plot"] = _pick_latest_by_mtime(grouped[k]["plots"])
        grouped[k]["latest_raw"] = _pick_latest_by_mtime(grouped[k]["raw"])
        grouped[k]["latest_metrics"] = _pick_latest_by_mtime(grouped[k]["metrics"])

    return grouped


def _build_html(root: Path, embed_images: bool) -> str:
    data_dir = root / "data"
    files = _collect_files(data_dir)
    grouped = _group_assets(files)

    now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def section_header(title: str, subtitle: str = "") -> str:
        return f"""
        <div class="section-head">
          <h2>{_safe_title(title)}</h2>
          {f"<div class='sub'>{_safe_title(subtitle)}</div>" if subtitle else ""}
        </div>
        """

    def plot_block(title: str, img_path: Optional[Path]) -> str:
        if not img_path:
            return "<div class='note'>No plot found.</div>"
        return f"""
        <div class="card">
          <div class="card-title">{_safe_title(title)} <span class="muted">({html.escape(img_path.name)})</span></div>
          <div class="plot-wrap">{_img_tag(img_path, embed_images)}</div>
        </div>
        """

    def many_plots_block(title: str, paths: List[Path]) -> str:
        if not paths:
            return "<div class='note'>No plots found.</div>"
        cards = []
        for p in sorted(paths, key=lambda x: x.name):
            cards.append(f"""
              <div class="card">
                <div class="card-title">{_safe_title(title)} <span class="muted">({html.escape(p.name)})</span></div>
                <div class="plot-wrap">{_img_tag(p, embed_images)}</div>
              </div>
            """)
        return "<div class='grid'>" + "".join(cards) + "</div>"

    def metrics_block(title: str, metrics_path: Optional[Path]) -> str:
        if not metrics_path:
            return "<div class='note'>No metrics JSON found.</div>"
        try:
            m = _read_json(metrics_path)
        except Exception as e:
            return f"<div class='note error'>Could not read {html.escape(metrics_path.name)}: {html.escape(str(e))}</div>"
        return _metrics_table_html(m, title=f"{title} metrics")

    def raw_preview_block(title: str, raw_path: Optional[Path]) -> str:
        if not raw_path:
            return "<div class='note'>No raw CSV found.</div>"
        return f"""
        <div class="card">
          <div class="card-title">{_safe_title(title)} <span class="muted">({html.escape(raw_path.name)})</span></div>
          <div class="table-wrap">{_df_preview_html(raw_path)}</div>
        </div>
        """

    iv = grouped["iv"]
    pulse = grouped["pulse"]
    endu = grouped["endurance"]
    ret = grouped["retention"]

    os.chdir(str(root))

    html_out = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Synaptic Memristor Dashboard</title>
  <style>
    :root {{
      --bg: #0b0d12;
      --card: #121622;
      --text: #e9ecf1;
      --muted: #9aa3b2;
      --accent: #7aa2ff;
      --border: #1f2637;
      --error: #ff6b6b;
    }}
    body {{
      margin: 0; padding: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }}
    header {{
      position: sticky; top: 0;
      background: rgba(11,13,18,0.85);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--border);
      padding: 14px 18px;
      z-index: 10;
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 18px; }}
    .title {{ font-size: 20px; font-weight: 700; }}
    .meta {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
    nav a {{
      color: var(--muted);
      text-decoration: none;
      margin-right: 14px;
      font-size: 13px;
    }}
    nav a:hover {{ color: var(--text); }}
    .section-head {{ margin-top: 26px; margin-bottom: 12px; }}
    h2 {{ margin: 0; font-size: 18px; }}
    .sub {{ color: var(--muted); margin-top: 4px; font-size: 13px; }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }}
    @media (min-width: 920px) {{
      .grid {{ grid-template-columns: 1fr 1fr; }}
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px 12px 10px 12px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.22);
    }}
    .card-title {{
      font-size: 13px;
      color: var(--text);
      font-weight: 600;
      margin-bottom: 10px;
    }}
    .muted {{ color: var(--muted); font-weight: 500; }}
    .plot-wrap {{
      background: rgba(255,255,255,0.02);
      border: 1px dashed rgba(255,255,255,0.06);
      border-radius: 12px;
      padding: 8px;
      overflow: hidden;
    }}
    img.plot {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 10px;
    }}
    .note {{
      background: rgba(122,162,255,0.08);
      border: 1px solid rgba(122,162,255,0.2);
      padding: 10px 12px;
      border-radius: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .note.error {{
      background: rgba(255,107,107,0.08);
      border: 1px solid rgba(255,107,107,0.25);
      color: #ffb2b2;
    }}
    table.kv {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    table.kv td {{
      padding: 8px 6px;
      border-top: 1px solid rgba(255,255,255,0.06);
      vertical-align: top;
    }}
    table.kv td.k {{
      width: 55%;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 12px;
    }}
    table.kv td.v {{
      width: 45%;
      color: var(--text);
      text-align: right;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 12px;
    }}
    .table-wrap {{
      overflow: auto;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.06);
    }}
    table.table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      background: rgba(255,255,255,0.02);
    }}
    table.table th, table.table td {{
      padding: 8px 8px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      white-space: nowrap;
    }}
    table.table th {{
      text-align: left;
      color: var(--muted);
      font-weight: 600;
    }}
    footer {{
      color: var(--muted);
      font-size: 12px;
      padding: 24px 0 40px 0;
    }}
    .chip {{
      display: inline-block;
      border: 1px solid rgba(255,255,255,0.10);
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 12px;
      color: var(--muted);
      margin-left: 8px;
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <div class="title">Synaptic Memristor — Results Dashboard <span class="chip">{'embedded images' if embed_images else 'linked images'}</span></div>
      <div class="meta">Generated: {html.escape(now_str)} · Source: <span class="muted">data/plots</span>, <span class="muted">data/processed</span>, <span class="muted">data/raw</span></div>
      <div style="margin-top:10px;">
        <nav>
          <a href="#all">All graphs</a>
          <a href="#iv">IV + Hysteresis</a>
          <a href="#pulse">Pulse</a>
          <a href="#endurance">Endurance</a>
          <a href="#retention">Retention</a>
          <a href="#raw">Raw previews</a>
        </nav>
      </div>
    </div>
  </header>

  <div class="wrap">

    <a id="all"></a>
    {section_header("All graphs", "Everything at a glance.")}
    <div class="grid">
      {plot_block("IV curve (latest)", iv["latest_plot"])}
      {plot_block("Hysteresis (latest)", _pick_latest_by_mtime(iv["hysteresis"]))}
      {plot_block("Pulse response (latest)", pulse["latest_plot"])}
      {plot_block("Endurance (latest)", endu["latest_plot"])}
      {plot_block("Retention (latest)", ret["latest_plot"])}
    </div>

    <a id="iv"></a>
    {section_header("IV + Hysteresis", "IV curve + bidirectional sweep hysteresis (area between branches).")}
    <div class="grid">
      {many_plots_block("IV plot", iv["plots"])}
      {many_plots_block("Hysteresis plot", iv["hysteresis"])}
      {metrics_block("IV", iv["latest_metrics"])}
      {raw_preview_block("IV raw preview", iv["latest_raw"])}
    </div>

    <a id="pulse"></a>
    {section_header("Pulse", "STP → LTP style pulse response (training pulses).")}
    <div class="grid">
      {many_plots_block("Pulse plot", pulse["plots"])}
      {raw_preview_block("Pulse raw preview", pulse["latest_raw"])}
      <div class="note">
        Pulse currently has no processed metrics JSON in v3 (that’s ok). If you later add metrics,
        put them into <span class="muted">data/processed</span> and the dashboard can be extended to load them.
      </div>
    </div>

    <a id="endurance"></a>
    {section_header("Endurance", "Alternating SET/RESET cycling with READ measurements over cycles.")}
    <div class="grid">
      {many_plots_block("Endurance plot", endu["plots"])}
      {metrics_block("Endurance", endu["latest_metrics"])}
      {raw_preview_block("Endurance raw preview", endu["latest_raw"])}
    </div>

    <a id="retention"></a>
    {section_header("Retention", "Program once, then READ over increasing delays (time).")}
    <div class="grid">
      {many_plots_block("Retention plot", ret["plots"])}
      {metrics_block("Retention", ret["latest_metrics"])}
      {raw_preview_block("Retention raw preview", ret["latest_raw"])}
    </div>

    <a id="raw"></a>
    {section_header("Raw datasets present", "All CSVs found in data/raw (for quick sanity checks).")}
    <div class="card">
      <div class="card-title">Raw files</div>
      <table class="table">
        <thead>
          <tr><th>File</th><th>Size</th><th>Modified</th></tr>
        </thead>
        <tbody>
          {''.join(
            f"<tr><td>{html.escape((p.relative_to(root)).as_posix())}</td>"
            f"<td>{p.stat().st_size} bytes</td>"
            f"<td>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime))}</td></tr>"
            for p in files['raw_csv']
          )}
        </tbody>
      </table>
    </div>

    <footer>
      Tip: If images don’t show when opening the HTML directly, run
      <span class="muted">python simulation/dashboard.py --serve</span>
      or regenerate with <span class="muted">--embed-images</span>.
    </footer>

  </div>
</body>
</html>
"""
    return html_out


def _write_dashboard(root: Path, embed_images: bool) -> Path:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "dashboard.html"
    html_text = _build_html(root, embed_images=embed_images)
    out_path.write_text(html_text, encoding="utf-8")
    return out_path


def _serve_repo(root: Path, port: int = 8000) -> None:
    os.chdir(str(root))

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        print(f"Serving repo at http://127.0.0.1:{port}")
        print(f"Open: http://127.0.0.1:{port}/data/dashboard.html")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-images", action="store_true", help="Embed images into the HTML (portable single file).")
    parser.add_argument("--serve", action="store_true", help="Serve the repo so the dashboard + images load reliably.")
    parser.add_argument("--port", type=int, default=8000, help="Port for --serve (default: 8000).")
    args = parser.parse_args()

    root = _project_root()
    out = _write_dashboard(root, embed_images=args.embed_images)
    print(f"Dashboard written to: {out}")
    if args.serve:
        _serve_repo(root, port=args.port)


if __name__ == "__main__":
    main()
