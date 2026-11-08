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

## ดู PAPER (ไม่บังคับ)

`LEGO_MODE=paper` ของ lego เก็บจำนวนหุ้นและ fill ปลอมไว้ใน database แยกชื่อ `lego-firebase-paper` จึงไม่ปนกับ `lego-firebase-prod`
ถ้าจะดูใน dashboard ให้เพิ่ม secret นี้:

```toml
FIREBASE_DB_URL_PAPER = "https://lego-firebase-paper.asia-southeast1.firebasedatabase.app"
```

- sidebar จะมีตัวเลือก **ฐานข้อมูล: PROD / PAPER (จำลอง)** (ค่าเริ่มต้นคือ PROD)
- ใช้ service account เดิม เพราะ `roles/firebasedatabase.viewer` ระดับ project อ่านได้ทุก database
- ตอนเลือก PAPER ทั้งหน้าจะมีป้ายสีส้ม **PAPER** และ chain ที่ backend ตีตราว่า paper จะมีป้ายนี้เสมอไม่ว่าอยู่ database ไหน
- ไม่ตั้ง secret นี้ = หน้า PROD อย่างเดียวเหมือนเดิม

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
- `lego_live_data.py` — ชั้นข้อมูลของแท็บ Live: อ่าน Firebase ครั้งเดียวเป็น snapshot, memo ผลคำนวณ,
  สถานะ สด/ค้าง/ผิดพลาด (pure ไม่ import streamlit — `test_live_data.py`)
- `lego_hover_table.py` — ตาราง HTML ที่ชี้เมาส์ดูที่มาของตัวเลขได้ (ฝังด้วย `components.html`)
- `lego_charts.py` — กราฟ Plotly ทั้งหมด (pure ไม่ import streamlit จึง test ได้ตรง ๆ — `test_charts.py`)
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

## แท็บ Live Dashboard

```
(•) AAPL_new  ( ) QBTS_x  ( ) AAPL_old          ← เลือก Chain 1 คลิก (ซ่อนเมื่อมี chain เดียว)
AAPL · Live     [สด] อัปเดต 14:05 UTC · 3 นาทีที่แล้ว · chain AAPL_new     [⟳ รีเฟรช]
วัน (UTC) [2026-08-02 ▾]  แสดง 3 จาก 6 แถว      ← ตัวกรองวัน ใช้ร่วมทั้งสองแท็บ
(คำเตือน: PAPER · engine เขียน ledger ผิด · broker fee PENDING)
[ข้อมูล] [กราฟ]
  ข้อมูล: การ์ด 5 metric → ตาราง 17 คอลัมน์ (สวิตช์ ดูที่มา) → Order audit → ▸ Integrity E1–E8 → ▸ State pointer
  กราฟ:  Rₙ / Aₙ / Eₙ ของ chain ตามวัน → Eₙ ผลรวมทุกตัว ตามเวลา
```

- **อ่าน Firebase ครั้งเดียวต่อการโหลดหน้า** (4 node) แล้วส่ง snapshot เข้า `st.fragment`: สลับ Chain / แท็บ / วัน /
  สวิตช์ / เปิด Integrity = rerun เฉพาะส่วนนี้บน snapshot เดิม ไม่อ่าน Firebase ซ้ำและไม่คำนวณซ้ำ (ผลคำนวณ memo บน snapshot)
- **รีเฟรช**: ปุ่มมุมขวา = อ่าน Firebase ใหม่ทั้ง 4 node — หน้าไม่รีเฟรชเอง (ไม่มี timer) ชี้เมาส์ที่ปุ่มเห็นเวลาที่ดึงข้อมูลล่าสุด
- **สถานะ**: สด = มีแถว/`updated_at` ใหม่ภายใน 2 slot (ความยาว slot อ่านจากระยะห่างแถวท้าย ๆ ของ chain, ไม่มีใช้ 15 นาที) ·
  ค้าง = นานกว่านั้น พร้อมเหตุผล "ตลาดปิด" เมื่ออยู่นอก จ–ศ 09:30–16:00 เวลา New York (ไม่มีปฏิทินวันหยุด จึงเป็นการประมาณ) ·
  ผิดพลาด = โหลดข้อมูลไม่ได้ หรือ System health มีรายการ ACTIVE
- **Chain ใน URL**: `?chain=<chain_key>` เปิด chain นั้นตรง ๆ (แชร์ลิงก์ได้) ค่าที่ไม่มีอยู่จริงถูกเมินและลบจาก URL
  ค่าเริ่มต้นคือ chain ที่ `updated_at` ใน state pointer ล่าสุด
- **ข้อมูล/กราฟ** รันเฉพาะแท็บที่เปิด · **Integrity E1–E8** คำนวณเมื่อเปิด expander เท่านั้น (chain v3 ที่ PASS เยอะใช้เวลา
  O(n²) — ราว 7 วินาทีที่ 500 แถว และ 28 วินาทีที่ 1,040 แถว ในการวัดของ PR นี้) แล้วจำผลไว้จนกว่าจะกดรีเฟรช
- **ไม่ล่มทั้งหน้า**: อ่าน `webull_lego_rows` / `webull_lego_state` ไม่ได้ = กรอบแดง + ปุ่มรีเฟรช (อีก 3 แท็บใช้ได้ตามปกติ);
  `webull_lego_order_audit` / `webull_lego_warnings` อ่านไม่ได้ = คำเตือนแล้วแสดงส่วนที่เหลือ; ช่องที่เป็น null แสดง "—"
- State pointer แสดงเฉพาะ chain ที่เลือก (ค่าซ้อนแสดงเป็น JSON)

## ดูที่มาของตัวเลขในตาราง 17 คอลัมน์

ในแท็บ Live Dashboard ชี้เมาส์ (หรือกด Tab) ที่ตัวเลขคอลัมน์ Rₙ, ΔAₙ, Aₙ, Eₙ เพื่อดูสูตรพร้อมแทนค่าจริงของแถวนั้น เช่น P_acted มาจากแถว act ไหน และเหตุผลที่แถว pass แช่แข็ง ตารางปัด 2 ตำแหน่ง ส่วน tooltip แสดง 4 ตำแหน่งเพื่อตรวจทาน ปิดสวิตช์ **ดูที่มา** เหนือตารางเพื่อกลับไปใช้ตารางแบบเรียงได้

### แถวแรกของ chain (n = 0) เริ่มจาก 0

ที่ n = 0: Rₙ = FIX_C × ln(P₀ / P₀) = 0 และ ΔAₙ = FIX_C × (P₀ / P₀ − 1) = 0 จึง Aₙ = Eₙ = 0 — fill แรกมี slippage (quote → fill) แต่ไม่นับเป็นกำไร/ขาดทุน
และ P_acted ของ act ถัดไปคือ P₀ เหมือน chain ที่แถวแรกไม่มี fill (act แรกเทียบ P₀) ตารางแถวแรกจึงเป็น `0.00` ทั้งสี่คอลัมน์ และ tooltip แสดงสูตรแทนค่า เช่น `1,500.0000 × (10.1900 / 10.1900 − 1)`

ถ้า DB ของ chain นั้นบันทึก ΔA₀ ≠ 0 จาก slippage ของ fill แรกไว้ (พบใน chain ที่เริ่มด้วย LEAPS) dashboard แสดงแบบเริ่มจาก 0:
หัก ΔA₀ ที่ DB บันทึกออกจาก Aₙ/Eₙ ของแถวถัดไป และคิด ΔAₙ ของ act ถัดไปใหม่จาก P₀ (DB เทียบราคา fill แรก) แล้วปรับ Aₙ/Eₙ หลังจากนั้นตามผลรวมใหม่
ตัวเลขทุกแถวจึงเท่ากับ chain เดียวกันที่แถวแรกเป็น PASS ที่ P₀ (ไม่มี fill)
หน้าไม่ขึ้นป้ายแจ้งเรื่องนี้ (ปิดแล้ว) — **ไม่แก้ DB**: ชี้เมาส์ที่ตัวเลขเพื่อดูค่าที่ DB บันทึกและจำนวนที่ตัดออก และ Integrity check ด้านล่างยังตรวจค่าที่ DB บันทึกจริง
เมื่อ backend แก้ให้ fill แรกเป็น funding (ΔA₀ = 0) กฎนี้จะไม่มีผลเอง รายละเอียดใน [docs/PERSISTED_LEDGER_AUDIT_20260916.md](docs/PERSISTED_LEDGER_AUDIT_20260916.md)

### LEAPS แทนหุ้น (`LEGO_LEAPS` ฝั่ง bot)

chain ที่ bot ใช้ LEAPS: คอลัมน์ 7 "จำนวนถือครอง (หุ้น)" = หุ้นจริง + LEAPS (หุ้นจำลอง, 1 สัญญา = 100) ตาราง 17 คอลัมน์ไม่เปลี่ยน ค่าแยกมาจาก field แบน `real_holdings` / `leaps_shares` (ต่อท้ายหลัง 17 คอลัมน์) — ชี้เมาส์ที่คอลัมน์ 7 เห็น "หุ้นจริง X + LEAPS Y = Z", SELL ได้ไม่เกินหุ้นจริง, แถวที่ LEAPS เปลี่ยน (`leaps_shares_changed_from`) และค่าหลัง fill; caption ใต้หัวตารางสรุปแถวล่าสุด. แถว `PASS_SELL_EXCEEDS_REAL` = ต้องขาย ≥ หุ้นจริง จึงไม่ส่ง order. chain ที่ไม่ใช้ LEAPS แสดงผลเหมือนเดิมทุกตัวอักษร

## ตัวกรองวัน (UTC) และ Eₙ ผลรวมทุกตัว

เปลี่ยนแค่การแสดงผล สูตรคำนวณเหมือนเดิมทุกอย่าง

**ตัวกรองวัน (UTC)**: selectbox "วัน (UTC)" อยู่ใต้หัวข้อของแท็บ Live Dashboard ใช้ร่วมทั้งแท็บ ข้อมูล และ กราฟ

- ค่าเริ่มต้นคือวันล่าสุดที่ chain มีแถว (ตัวแรกของรายการ) ไม่ใช่วันนี้ตามปฏิทิน เช่น ก่อนตลาด US เปิด (13:30 UTC = 20:30 น. เวลาไทย) จะเห็น session ล่าสุด
- ตัวเลือกสุดท้ายคือ "ทุกวัน (ทั้ง chain)"
- กลับไปวันล่าสุดเองเมื่อสลับ PROD/PAPER เปลี่ยน chain หรือมีวันใหม่ ส่วนวันที่เลือกเองยังค้างอยู่ระหว่างกด widget อื่นและสลับแท็บ
- มีผลกับกราฟ ตาราง 17 คอลัมน์ และ Order audit (order ผูกกับแถวด้วย `run_id` วันของ order จึงเป็นวันของแถวนั้น)
- Rₙ / ΔAₙ / Aₙ / Eₙ และ tooltip ยังคำนวณจากทั้ง chain ก่อนกรอง ตัวกรองแค่เลือกแถวที่แสดง: tooltip ของแถวแรกในวันยังอ้าง P_acted / Aₙ₋₁ จากวันก่อนได้ถูกต้อง
- metric, caption LEAPS, Integrity check และคำเตือน broker fee PENDING (money fence) ใช้ทั้ง chain เสมอ
  ถ้า fee ของ fill วันก่อนยัง PENDING คำเตือนก็ยังขึ้นตอนดูวันล่าสุด

**metric "Eₙ สะสม (USD) ผลรวมทุกตัว"**: การ์ดสุดท้ายในแท็บ ข้อมูล ข้าง "Eₙ สะสม (USD)" ซึ่งยังเป็นค่าของ chain ที่เลือก

- ผลบวก Eₙ ของ chain ล่าสุดของแต่ละหุ้น โดย Eₙ ของแต่ละตัวคือค่าที่ metric "Eₙ สะสม (USD)" แสดงเมื่อเลือก chain นั้น
- chain ล่าสุดของหุ้น = chain ที่ `updated_at` ใน state pointer ล่าสุด (ไม่มี state ใช้เวลาแถวล่าสุด) chain เก่าของหุ้นเดียวกัน เช่นหลังเปลี่ยน FIX_C / DIFF / DNA จึงไม่ถูกนับ
- หน้า PROD ไม่นับ chain PAPER (state pointer หรือแถวมี `paper = true`) ส่วนหน้า PAPER นับทุก chain ในฐานข้อมูล paper
- ชี้เมาส์ที่ไอคอน ? ข้างชื่อ metric เพื่อดูรายการแต่ละหุ้น สมการผลรวม และ chain ที่ไม่นับ
- ไม่มี chain ให้นับ หรือมี Eₙ ที่อ่านไม่ได้ จะแสดง "—"

**กราฟ "Eₙ ผลรวมทุกตัว ตามเวลา"**: อยู่ท้ายแท็บ กราฟ (`excess_total_timeline`)

- นับ chain ชุดเดียวกับ metric ผลรวม (ใช้ `_counted_chains` ร่วมกัน) และ Eₙ จากสูตรเดียวกัน ค่าปลายกราฟจึงเท่ากับ metric
- เส้นหนาคือผลรวม เส้นบางคือ Eₙ ของแต่ละหุ้น แกน x เป็นเวลา (UTC) มีปุ่ม 1D / 1W / 1M / ทั้งหมด และตาราง Eₙ ล่าสุดรายหุ้นใน expander
- Eₙ เป็นค่าสะสมอยู่แล้ว: ผลรวม ณ เวลา t = ผลบวก Eₙ ล่าสุดของแต่ละ chain ที่ ≤ t (ไม่ cumsum ซ้ำ) ก่อนแถวแรกของ chain นับเป็น 0
- ไม่ขึ้นกับตัวกรองวัน แถวที่เวลาอ่านไม่ได้ไม่อยู่ในกราฟนี้

## กราฟ (Plotly)

กราฟทุกจุดสร้างจาก `lego_charts.py` แสดงด้วย `st.plotly_chart(theme=None)` สีเลือกแยกสำหรับธีม light/dark
(ตามธีม Streamlit) ชี้เมาส์เห็นค่าทุกเส้น ณ จุดเดียวกันเป็น USD ลากเพื่อซูม ดับเบิลคลิกเพื่อรีเซ็ต และกดปุ่มกล้องเพื่อบันทึก PNG

- **Rₙ / Aₙ / Eₙ** (Live Dashboard ตามตัวกรองวัน และ Rebalancing 101): Rₙ เป็นเส้นประอ้างอิง, Aₙ มีจุดที่แถว act
  (Live: ΔAₙ ≠ 0, 101: gate 1; ซ่อนจุดเมื่อมี act เกิน 60 รอบ), Eₙ มีพื้นที่ใต้เส้น และมี label ค่าล่าสุดที่ปลายเส้น
- หุ้นเกิน 8 ตัวในกราฟผลรวม ตัวที่ |Eₙ| เล็กสุดถูกรวมเป็น "อื่น ๆ"
- เส้นที่มีเกิน 1,500 จุด (`MAX_LINE_POINTS`) วาดเฉพาะจุดตัวแทนที่ห่างเท่ากัน + จุดแรก/ท้าย + แถว act
  (`downsample_positions`) — ค่าที่วาดคือค่าจริงของแถวนั้น ตาราง/metric ไม่เปลี่ยน

## เชื่อม Firebase สำหรับ Live Dashboard

สร้าง `.streamlit/secrets.toml` เฉพาะในเครื่องหรือกำหนดค่าใน Streamlit Community
Cloud โดยห้าม commit ไฟล์ secret:

```toml
FIREBASE_SA_JSON = """{ ... service account JSON ... }"""
FIREBASE_DB_URL = "https://lego-firebase-prod.asia-southeast1.firebasedatabase.app"
```

service account ควรมีสิทธิ์อ่านข้อมูลเท่าที่จำเป็น แดชบอร์ดอ่าน path ต่อไปนี้ (ครั้งละ 4 node ต่อการโหลดหน้า/กดรีเฟรช):

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

