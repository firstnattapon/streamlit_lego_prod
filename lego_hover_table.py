"""lego_hover_table.py — ตารางที่ชี้เมาส์ดูที่มาของตัวเลขได้ (HTML ล้วน ไม่มี streamlit I/O)

st.dataframe วาดด้วย canvas จึงทำ tooltip รายเซลล์ที่มีตัวเลขเฉพาะแถวไม่ได้ (มีแค่ help ของหัวคอลัมน์)
โมดูลนี้สร้างตาราง HTML + JS เล็ก ๆ ให้ฝังด้วย ``streamlit.components.v1.html``:
  - เซลล์ในคอลัมน์ที่มีคำอธิบาย (``tip_cols``) แสดง tooltip ตอนชี้เมาส์หรือกด Tab โฟกัส
  - ข้อความทุกส่วน (ค่าในเซลล์, tooltip) ถูก escape — ข้อมูลมาจาก Firebase ห้ามเชื่อ
  - JS ใช้ textContent เท่านั้น ไม่มี innerHTML
"""
from __future__ import annotations

import html
import math
import numbers

import numpy as np
import pandas as pd

MAX_HEIGHT = 520       # px ของกรอบตาราง — เกินนี้เลื่อนในกรอบ
MIN_HEIGHT = 300       # เตี้ยกว่านี้ tooltip หลายบรรทัดไม่มีที่แสดง


def hover_table_height(n_rows: int) -> int:
    """ความสูง iframe ที่พอดีกับจำนวนแถว (หัวตาราง ~46px + แถวละ ~30px)"""
    return max(MIN_HEIGHT, min(MAX_HEIGHT, 46 + 30 * int(n_rows)))


def _missing(value) -> bool:
    return value is None or (pd.api.types.is_scalar(value)
                             and not isinstance(value, str) and bool(pd.isna(value)))


def _cell_text(value, money: bool) -> str:
    if _missing(value):
        return ""
    if money:
        try:
            number = float(value)
            if math.isfinite(number):
                return f"{number:.2f}"
        except (TypeError, ValueError):
            pass
    return str(value)


def _is_number(value) -> bool:
    return (isinstance(value, numbers.Number)
            and not isinstance(value, (bool, np.bool_)) and not _missing(value))


_CSS = """
:root{--fg:#1f2430;--bg:transparent;--head:#eef0f4;--line:#d9dce3;--tipcol:#f4f8ff;
--hl:#dbe8ff;--row:#f6f7fa;--tipbg:#1f2430;--tipfg:#f5f6f8;--tipti:#ffd479;--accent:#3b6fd4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--fg:#e6e8ee;--head:#2a2e39;
--line:#3b4050;--tipcol:#1c2433;--hl:#2c4270;--row:#222632;--tipbg:#0d0f14;--tipfg:#eceef3;
--tipti:#ffd479;--accent:#7fa6f5}}
:root[data-theme="dark"]{--fg:#e6e8ee;--head:#2a2e39;--line:#3b4050;--tipcol:#1c2433;--hl:#2c4270;
--row:#222632;--tipbg:#0d0f14;--tipfg:#eceef3;--tipti:#ffd479;--accent:#7fa6f5}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.4 system-ui,-apple-system,"Segoe UI",
"Noto Sans Thai",sans-serif}
.wrap{max-height:%(max_height)dpx;overflow:auto;border:1px solid var(--line);border-radius:6px}
table{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%%}
th,td{padding:5px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{position:sticky;top:0;z-index:2;background:var(--head);text-align:left;font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums}
th.tip::after{content:" ⓘ";color:var(--accent)}
td.tip{background:var(--tipcol);cursor:help;text-decoration:underline dotted var(--accent);
text-underline-offset:3px}
tbody tr:hover td{background:var(--row)}
tbody tr:hover td.tip,td.tip:hover,td.tip:focus{background:var(--hl);outline:none}
td.tip:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
#tip{position:fixed;display:none;z-index:9;max-width:min(560px,94vw);padding:8px 11px;
border-radius:6px;background:var(--tipbg);color:var(--tipfg);pointer-events:none;
box-shadow:0 4px 18px rgba(0,0,0,.35);font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,
"Noto Sans Thai",monospace;white-space:pre-wrap}
#tip .t{color:var(--tipti);font-weight:700;margin-bottom:3px}
"""

_JS = """
(function(){
  var tip=document.getElementById('tip'), wrap=document.querySelector('.wrap'), cur=null;
  function hide(){cur=null;tip.style.display='none';}
  function place(x,y){
    var pad=14, r=tip.getBoundingClientRect(), w=window.innerWidth, h=window.innerHeight;
    var left=x+pad, top=y+pad;
    if(left+r.width>w-4) left=Math.max(4,x-r.width-pad);
    if(top+r.height>h-4) top=Math.max(4,y-r.height-pad);
    tip.style.left=left+'px'; tip.style.top=top+'px';
  }
  function show(td,x,y){
    cur=td;
    var lines=td.getAttribute('data-tip').split('\\n');
    var head=document.createElement('div'); head.className='t'; head.textContent=lines.shift();
    var body=document.createElement('div'); body.textContent=lines.join('\\n');
    tip.replaceChildren(head,body);
    tip.style.display='block'; place(x,y);
  }
  wrap.addEventListener('mousemove',function(e){
    var td=e.target.closest&&e.target.closest('td[data-tip]');
    if(!td){hide();return;}
    if(td!==cur) show(td,e.clientX,e.clientY); else place(e.clientX,e.clientY);
  });
  wrap.addEventListener('mouseleave',hide);
  wrap.addEventListener('scroll',hide);
  wrap.addEventListener('focusin',function(e){
    var td=e.target.closest&&e.target.closest('td[data-tip]');
    if(td){var r=td.getBoundingClientRect(); show(td,r.left,r.bottom);}
  });
  wrap.addEventListener('focusout',hide);
})();
"""


def build_hover_table_html(frame: pd.DataFrame, tips: list[dict[str, str]],
                           tip_cols, money_cols=(), theme: str | None = None,
                           max_height: int = MAX_HEIGHT) -> str:
    """ตาราง HTML เต็มหน้า (สำหรับ components.html)

    ``tips[i][col]`` คือข้อความ tooltip ของแถวลำดับที่ i (ตามตำแหน่ง ไม่ใช่ index) คอลัมน์ col
    บรรทัดแรกของข้อความเป็นหัวข้อ ที่เหลือเป็นเนื้อหา; ไม่มีข้อความ -> เซลล์ธรรมดา
    ``theme`` = "light"/"dark" บังคับสี; None = ตาม prefers-color-scheme ของเบราว์เซอร์
    """
    tip_cols = set(tip_cols)
    money_cols = set(money_cols)
    frame = frame.reset_index(drop=True)
    columns = list(frame.columns)

    head = "".join(
        f'<th class="tip" title="ชี้ที่ตัวเลขในคอลัมน์นี้เพื่อดูที่มา">{html.escape(str(c))}</th>'
        if c in tip_cols else f"<th>{html.escape(str(c))}</th>" for c in columns)

    body_rows = []
    for i in range(len(frame)):
        cells = []
        for col in columns:
            raw = frame.at[i, col]
            text = html.escape(_cell_text(raw, col in money_cols))
            classes = ["num"] if (col in money_cols or _is_number(raw)) else []
            attr = ""
            tip = tips[i].get(col) if i < len(tips) else None
            if col in tip_cols and tip:
                classes.append("tip")
                attr = f' data-tip="{html.escape(tip, quote=True)}" tabindex="0"'
            klass = f' class="{" ".join(classes)}"' if classes else ""
            cells.append(f"<td{klass}{attr}>{text}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    theme_attr = f' data-theme="{theme}"' if theme in ("light", "dark") else ""
    return (
        f'<!doctype html><html lang="th"{theme_attr}><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<style>{_CSS % {'max_height': int(max_height) - 4}}</style></head><body>"
        f'<div class="wrap"><table><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table></div>'
        f'<div id="tip" role="tooltip"></div><script>{_JS}</script></body></html>'
    )
