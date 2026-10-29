# LEGO Streamlit Dashboard: PROD

repo นี้คือสำเนาของ [streamlit_lego](https://github.com/firstnattapon/streamlit_lego) สำหรับ `lego-tick-prod`
หน้าเว็บจึงมีป้าย PROD และอ่าน RTDB `lego-firebase-prod` แทน DB ของ UAT

> ⚠️ **repo นี้เป็น public แต่ dashboard แสดงข้อมูลบัญชีจริง**
> - ห้าม commit secrets ใดๆ (`.gitignore` กัน `.streamlit/secrets.toml` และไฟล์ service account ไว้แล้ว)
> - ตั้ง app บน Streamlit Community Cloud เป็น **private**: Share → เชิญเฉพาะ email ที่อนุญาต
>   ถ้า app เป็น public ใครมี URL ก็เห็นยอดเงินและ order ของบัญชีจริง

## ตั้งค่า PROD

ใส่ใน Streamlit Cloud → app → Settings → Secrets (หรือ `.streamlit/secrets.toml` ในเครื่อง):

```toml
FIREBASE_DB_URL = "https://lego-firebase-prod.asia-southeast1.firebasedatabase.app"
FIREBASE_SA_JSON = """{ ... service account JSON ... }"""
```

- service account ควรเป็นตัวแยกสำหรับ dashboard และมีแค่ role `roles/firebasedatabase.viewer` (อ่านอย่างเดียว) ในโปรเจกต์ `lego-firebase`
- rules ของ PROD ปิดการอ่านสาธารณะ แต่ Admin SDK อ่านผ่าน IAM จึงไม่ติด rules
- ถ้า `FIREBASE_DB_URL` ไม่มี `lego-firebase-prod` หน้าเว็บจะขึ้นกรอบแดงเตือนว่าอาจเป็น DB ของ UAT

สร้าง service account และ key ใน Cloud Shell:

```bash
gcloud iam service-accounts create lego-dash-prod --project=lego-firebase --display-name="LEGO PROD dashboard (read-only)"
gcloud projects add-iam-policy-binding lego-firebase --member="serviceAccount:lego-dash-prod@lego-firebase.iam.gserviceaccount.com" --role="roles/firebasedatabase.viewer"
gcloud iam service-accounts keys create lego-dash-prod.json --iam-account=lego-dash-prod@lego-firebase.iam.gserviceaccount.com
```

นำเนื้อหาของ `lego-dash-prod.json` ไปใส่ใน `FIREBASE_SA_JSON` แล้วลบไฟล์ทิ้ง (`rm lego-dash-prod.json`)

---

อัปเดตการตรวจข้อมูลต้นฉบับและ baseline ซื้อครั้งแรก:
[บันทึก audit 16 กันยายน 2026](docs/PERSISTED_LEDGER_AUDIT_20260916.md)

แดชบอร์ดแบบ read-only สำหรับอ่าน committed rows จาก Firebase Realtime Database
และหน้าเรียนรู้ Rebalancing 101 ที่จำลอง DNA gate ได้โดยไม่ต้องเชื่อม Firebase
พร้อมคู่มือ HTML แบบโต้ตอบ (แท็บ Web Apps) และ cheat sheet แผนภาพ Mermaid (แท็บ Cheat Sheet)

> สถานะ candidate นี้ยังไม่พร้อมใช้ Production โปรดอ่าน
> [RELEASE_STATUS_TH.md](RELEASE_STATUS_TH.md) ก่อนใช้งาน

## ไฟล์หลัก

- `streamlit_app.py` — หน้า Live Dashboard, Rebalancing 101, Web Apps และ Cheat Sheet
- `web_apps/` — คู่มือ/playground แบบ HTML ล้วน แสดงใน iframe ผ่าน `components.html`
  (รายละเอียดด้านล่าง)
- `cheat_sheet/` — แผนภาพ Mermaid `.mmd` + ภาพ PNG ที่ render ไว้ + `summary.md` (สรุป 10 ข้อ + ประโยคเดียว) ·
  `tools/render_cheat_sheet.py` สร้างภาพใหม่ (รายละเอียดด้านล่าง)
- `lego_dash_core.py` — การจัดตาราง ตรวจ integrity และคำนวณ ledger สำหรับการแสดงผล
- `lego_hover_table.py` — ตาราง HTML ที่ชี้เมาส์ดูที่มาของตัวเลขได้ (ฝังด้วย `components.html`)
- `dna_engine.py` — ตัวถอด DNA แบบเดียวกับ backend
- `requirements.txt` — dependency ที่ pin แล้ว
- `docs/` — คู่มือและแผนการเรียนรู้เพิ่มเติม

## เริ่มใช้งานในเครื่อง

ต้องใช้ Python 3.12

```powershell
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

แท็บ Rebalancing 101, Web Apps และ Cheat Sheet ใช้งานได้ทันทีโดยไม่ต้องมี credential

## แท็บ Cheat Sheet

มี 2 แท็บย่อย: **📝 Summary** (เปิดเป็นแท็บแรก) แสดงสรุป 10 ข้อ + ประโยคเดียวจาก `cheat_sheet/summary.md`
(แก้ข้อความที่ไฟล์นี้ได้โดยไม่ต้องแตะโค้ด) และ **🔀 Diagrams** แสดงแผนภาพตามด้านล่าง

แผนภาพ Mermaid ของขั้นตอน **main rebalancing**, **rollover**
(`beta += (Fix_c − Fix_c_new) * ln(P_acted / P₀)`; beta บวกที่ R เท่านั้น ไม่บวกที่ Aₙ)
และ **Option Scale** (จ่าย K ก่อน → เก็บ E → ขยาย `Fix_c` ด้วยกำไรที่ล็อกแล้ว ย่อจาก
`app_v4_system.mmd` ของ repo super_app ที่มีแผนผังเต็ม)
เก็บเป็นไฟล์ `.mmd` ใน `cheat_sheet/`
(Mermaid มาตรฐาน — วางใน code block ```` ```mermaid ```` ของ Markdown หรือ Mermaid Live Editor ได้)
ส่วนแท็บแสดงภาพ PNG ที่ render ไว้ล่วงหน้า (เปิดทันที ไม่ต้องมี internet ไม่โหลดไลบรารีเพิ่ม)
พร้อมซอร์สให้คัดลอกและปุ่มดาวน์โหลด

ชื่อภาพมี hash ของ `.mmd` (`rollover.<sha8>.png`) ถ้าแก้ `.mmd` แล้วยังไม่ render ใหม่
แท็บจะเตือนและ `test_cheat_sheet.py` จะไม่ผ่าน — แก้แผนภาพแล้วสร้างภาพใหม่ (ต้องมี Node.js;
ครั้งแรก puppeteer จะโหลด Chromium):

```powershell
python tools/render_cheat_sheet.py
```

## แท็บ Web Apps

คู่มือแบบโต้ตอบเป็นไฟล์ HTML เดี่ยวที่ไม่โหลดอะไรจากภายนอก เก็บใน `web_apps/`
แล้วแสดงด้วย `render_web_app()` ใน `streamlit_app.py` (iframe แยก ไม่เรียก Firebase
และไม่อ่าน secrets) หน้า HTML รองรับ `:root[data-theme]` จึงสีตรงกับธีม Streamlit

| ไฟล์ | ที่มา |
| --- | --- |
| `rebalancing101.html`, `rebalancing_playground.html` | คัดลอกทุกไบต์จากแท็บ Web Apps ของ `pages/Manual.py` ใน repo leg-code (commit `c797709`) |

เพิ่มคู่มือใหม่: วางไฟล์ HTML ใน `web_apps/` แล้วเพิ่มชื่อใน `WEB_APPS` (ตัวเลือกในแท็บ
Web Apps) — `test_web_apps.py` จะตรวจให้อัตโนมัติว่าทุกหน้าใน `web_apps/` อยู่ในรายการ
และยัง self-contained

## ดูที่มาของตัวเลขในตาราง 17 คอลัมน์

ในแท็บ Live Dashboard ชี้เมาส์ (หรือกด Tab) ที่ตัวเลขคอลัมน์ Rₙ, ΔAₙ, Aₙ, Eₙ เพื่อดูสูตรพร้อมแทนค่าจริงของแถวนั้น เช่น P_acted มาจากแถว act ไหน และเหตุผลที่แถว pass แช่แข็ง ตารางปัด 2 ตำแหน่ง ส่วน tooltip แสดง 4 ตำแหน่งเพื่อตรวจทาน ปิดสวิตช์เหนือตารางเพื่อกลับไปใช้ตารางแบบเรียงได้

## เชื่อม Firebase สำหรับ Live Dashboard

สร้าง `.streamlit/secrets.toml` เฉพาะในเครื่องหรือกำหนดค่าใน Streamlit Community
Cloud โดยห้าม commit ไฟล์ secret:

```toml
FIREBASE_SA_JSON = """{ ... service account JSON ... }"""
FIREBASE_DB_URL = "https://lego-firebase-prod.asia-southeast1.firebasedatabase.app"
```

service account ควรมีสิทธิ์อ่านข้อมูลเท่าที่จำเป็น แดชบอร์ดอ่าน path ต่อไปนี้:

- `webull_lego_rows`
- `webull_lego_state`
- `webull_lego_order_audit`
- `webull_lego_warnings`

## ทดสอบ

```powershell
python -m pip install -r requirements.txt pytest
python -m py_compile *.py
python -m pytest -q
```

แดชบอร์ดนี้ไม่ส่งคำสั่งซื้อขาย `READY_BUY` และ `READY_SELL` เป็น decision intent
ไม่ใช่หลักฐานว่า order ถูก fill แล้ว

