import os
import sqlite3
import uuid
import io
import json
import csv
import zipfile
import html as html_mod
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import FastAPI, Form, UploadFile, File, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from PIL import Image
from google import genai
from google.genai import types
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import asyncio

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "super-secret-fallback"))

oauth = OAuth()
oauth.register(
    name='oidc',
    server_metadata_url=os.getenv('OIDC_DISCOVERY_URL'),
    client_id=os.getenv('OIDC_CLIENT_ID'),
    client_secret=os.getenv('OIDC_CLIENT_SECRET'),
    client_kwargs={'scope': 'openid email profile'}
)

DATA_DIR = "/app/data"
IMG_DIR = f"{DATA_DIR}/images"
DB_PATH = f"{DATA_DIR}/prompts.db"
os.makedirs(IMG_DIR, exist_ok=True)

SMTP_HOST     = os.getenv("SMTP_HOST", "")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM     = os.getenv("SMTP_FROM", "")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS prompts
                 (id TEXT PRIMARY KEY, title TEXT, prompt TEXT, author TEXT, tags TEXT, 
                  image_path TEXT, user_email TEXT, is_shared INTEGER)''')
    
    c.execute("PRAGMA table_info(prompts)")
    columns = [col[1] for col in c.fetchall()]
    if 'copy_count' not in columns:
        c.execute("ALTER TABLE prompts ADD COLUMN copy_count INTEGER DEFAULT 0")
    if 'forked_from' not in columns:
        c.execute("ALTER TABLE prompts ADD COLUMN forked_from TEXT")

    c.execute('''CREATE TABLE IF NOT EXISTS prompt_history
                 (history_id TEXT PRIMARY KEY, prompt_id TEXT, title TEXT, prompt TEXT, 
                  author TEXT, tags TEXT, image_path TEXT, edited_by TEXT, 
                  edited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
                  
    c.execute('''CREATE TABLE IF NOT EXISTS favorites
                 (user_email TEXT, prompt_id TEXT, UNIQUE(user_email, prompt_id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS collections
                 (id TEXT PRIMARY KEY, name TEXT, user_email TEXT,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS collection_prompts
                 (collection_id TEXT, prompt_id TEXT,
                 UNIQUE(collection_id, prompt_id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS share_links
                 (token TEXT PRIMARY KEY, prompt_id TEXT, created_by TEXT,
                 expires_at TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    c.execute("PRAGMA table_info(share_links)")
    sl_cols = [col[1] for col in c.fetchall()]
    if 'created_at' not in sl_cols:
        c.execute("ALTER TABLE share_links ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

    c.execute('''CREATE TABLE IF NOT EXISTS comments
                 (id TEXT PRIMARY KEY, prompt_id TEXT, parent_id TEXT,
                 author_email TEXT, author_name TEXT, body TEXT,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    c.execute("PRAGMA table_info(comments)")
    comment_cols = [col[1] for col in c.fetchall()]
    if comment_cols:
        if 'parent_id' not in comment_cols:
            c.execute("ALTER TABLE comments ADD COLUMN parent_id TEXT")
        if 'author_email' not in comment_cols:
            c.execute("ALTER TABLE comments ADD COLUMN author_email TEXT")
        if 'author_name' not in comment_cols:
            c.execute("ALTER TABLE comments ADD COLUMN author_name TEXT")
        if 'body' not in comment_cols:
            c.execute("ALTER TABLE comments ADD COLUMN body TEXT")

    c.execute('''CREATE TABLE IF NOT EXISTS comment_votes
                 (comment_id TEXT, user_email TEXT, UNIQUE(comment_id, user_email))''')

    conn.commit()
    conn.close()

init_db()

def is_admin(user: dict) -> bool:
    if not user:
        return False
    
    # Helper to check if "admin" is inside a string (or comma-separated string) or a list
    def has_admin(val):
        if isinstance(val, str):
            return val.lower() == 'admin' or 'admin' in [v.strip().lower() for v in val.split(',')]
        if isinstance(val, list):
            return any(str(r).lower() == 'admin' for r in val)
        return False

    # 1. Check standard OIDC claims (Added singular 'role' and 'Role')
    for key in ['roles', 'groups', 'Roles', 'Groups', 'role', 'Role']:
        if has_admin(user.get(key)):
            return True
            
    # 2. Check Keycloak specific nested claims
    realm_access = user.get('realm_access', {})
    if isinstance(realm_access, dict) and has_admin(realm_access.get('roles')):
        return True
            
    # 3. Fallback: check manually configured admin emails via Env Var
    admin_emails = [e.strip().lower() for e in os.getenv('ADMIN_EMAILS', '').split(',') if e.strip()]
    if user.get('email', '').lower() in admin_emails:
        return True
        
    return False

def _send_email_sync(to: str, subject: str, body_html: str):
    if not (SMTP_HOST and SMTP_FROM and to):
        return
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = SMTP_FROM
        msg['To'] = to
        msg.attach(MIMEText(body_html, 'html'))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            if SMTP_PORT != 465:
                server.starttls()
            if SMTP_USER and SMTP_PASSWORD:
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, to, msg.as_string())
    except Exception as e:
        print(f"SMTP Error: {e}")

app.mount("/images", StaticFiles(directory=IMG_DIR, html=True), name="images")

async def optimize_and_save_image(upload_file: UploadFile) -> str:
    image_data = await upload_file.read()
    try:
        img = Image.open(io.BytesIO(image_data))
        img.verify()
        img = Image.open(io.BytesIO(image_data))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file format.")
    
    if img.mode in ("RGBA", "P"): img = img.convert("RGB")
    img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    
    filename = f"{uuid.uuid4()}.webp"
    file_path = os.path.join(IMG_DIR, filename)
    img.save(file_path, "webp", quality=80, optimize=True)
    return filename

def extract_metadata_from_image(image_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load() 
        info = img.info
        if 'parameters' in info: return str(info['parameters'])
        if 'prompt' in info:
            if isinstance(info['prompt'], str): return info['prompt']
            else: return json.dumps(info['prompt'], indent=2)
        if 'Description' in info: return str(info['Description'])
            
        exif = img.getexif()
        if exif:
            for tag in [270, 37510]:
                if tag in exif:
                    val = exif[tag]
                    if isinstance(val, bytes):
                        if val.startswith(b'ASCII\x00\x00\x00') or val.startswith(b'UNICODE\x00'):
                            val = val[8:]
                        val = val.decode('utf-8', errors='ignore')
                    val = str(val).strip()
                    if val and val != "None": return val
    except Exception: pass
    return ""

# --- AUTH ROUTES ---
@app.get('/login')
async def login(request: Request):
    redirect_uri = request.url_for('auth_callback')
    return await oauth.oidc.authorize_redirect(request, str(redirect_uri))

@app.get('/auth')
async def auth_callback(request: Request):
    token = await oauth.oidc.authorize_access_token(request)
    user = token.get('userinfo')
    if user: request.session['user'] = dict(user)
    return RedirectResponse('/')

@app.get('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
    return RedirectResponse('/')

# --- API ROUTES ---
@app.get("/api/prompts")
def get_prompts(request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401, detail="Unauthorized")
    
    user_email = str(user.get('email', 'unknown'))
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT p.*, CASE WHEN f.prompt_id IS NOT NULL THEN 1 ELSE 0 END as is_favorite,
               parent.title as forked_from_title,
               GROUP_CONCAT(cp.collection_id) as collection_ids
        FROM prompts p
        LEFT JOIN favorites f ON p.id = f.prompt_id AND f.user_email = ?
        LEFT JOIN prompts parent ON p.forked_from = parent.id
        LEFT JOIN collection_prompts cp ON p.id = cp.prompt_id
            AND cp.collection_id IN (SELECT id FROM collections WHERE user_email = ?)
        WHERE p.is_shared = 1 OR p.user_email = ?
        GROUP BY p.id
        ORDER BY p.rowid DESC
    """, (user_email, user_email, user_email))
    rows = [dict(row) for row in c.fetchall()]
    conn.close()

    for row in rows:
        row['is_mine'] = (row['user_email'] == user_email)
        row['is_favorite'] = bool(row['is_favorite'])
        raw_cids = row.get('collection_ids')
        row['collection_ids'] = str(raw_cids).split(',') if raw_cids else []
    return rows

@app.get("/api/prompts/{prompt_id}/history")
def get_prompt_history(prompt_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM prompt_history WHERE prompt_id = ? ORDER BY edited_at DESC", (prompt_id,))
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows

@app.post("/api/prompts/{prompt_id}/rollback/{history_id}")
async def rollback_prompt(prompt_id: str, history_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    user_email = str(user.get('email', 'unknown'))

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,))
    current = c.fetchone()
    if not current:
        conn.close()
        raise HTTPException(status_code=404, detail="Prompt not found.")

    c.execute("SELECT * FROM prompt_history WHERE history_id = ? AND prompt_id = ?", (history_id, prompt_id))
    hist = c.fetchone()
    if not hist:
        conn.close()
        raise HTTPException(status_code=404, detail="History entry not found.")

    c.execute("""INSERT INTO prompt_history (history_id, prompt_id, title, prompt, author, tags, image_path, edited_by)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
              (str(uuid.uuid4()), prompt_id, current['title'], current['prompt'],
               current['author'], current['tags'], current['image_path'], user_email))

    c.execute("""UPDATE prompts SET title = ?, prompt = ?, author = ?, tags = ?, image_path = ? WHERE id = ?""",
              (hist['title'], hist['prompt'], hist['author'], hist['tags'], hist['image_path'], prompt_id))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.post("/api/prompts/{prompt_id}/share-link")
async def create_share_link(prompt_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    user_email = str(user.get('email', 'unknown'))

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
    try:
        expires_in_hours = int(body.get("expires_in_hours", 24))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid expires_in_hours.")
    if not (1 <= expires_in_hours <= 720):
        raise HTTPException(status_code=400, detail="expires_in_hours must be 1–720.")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT id FROM prompts WHERE id = ? AND (user_email = ? OR is_shared = 1)", (prompt_id, user_email))
    if not c.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Prompt not found.")

    token = str(uuid.uuid4())
    expires_at = (datetime.utcnow() + timedelta(hours=expires_in_hours)).strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO share_links (token, prompt_id, created_by, expires_at) VALUES (?, ?, ?, ?)",
              (token, prompt_id, user_email, expires_at))
    conn.commit()
    conn.close()
    return {"token": token, "expires_at": expires_at}

@app.get("/api/prompts/{prompt_id}/share-links")
def list_share_links(prompt_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    user_email = str(user.get('email', 'unknown'))
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM share_links WHERE prompt_id = ? AND created_by = ? ORDER BY created_at DESC",
              (prompt_id, user_email))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

@app.delete("/api/share-links/{token}")
def revoke_share_link(token: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    user_email = str(user.get('email', 'unknown'))
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM share_links WHERE token = ? AND created_by = ?", (token, user_email))
    conn.commit()
    conn.close()
    return {"status": "revoked"}

@app.post("/api/prompts")
async def add_prompt(request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    user_email = str(user.get('email', 'unknown'))
    
    form = await request.form()
    title = form.get("title")
    prompt = form.get("prompt")
    author = form.get("author")
    tags = form.get("tags")
    is_shared = form.get("is_shared", "false")
    forked_from = form.get("forked_from") or None
    new_images = form.getlist("new_images")
    media_order_raw = form.get("media_order", "[]")
    
    try: media_order = json.loads(media_order_raw)
    except: media_order = []

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM prompts WHERE prompt = ?", (prompt.strip(),))
    if c.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="This exact prompt already exists!")

    saved_filenames = []
    for img in new_images:
        if hasattr(img, "filename") and hasattr(img, "read"):
            if getattr(img, "filename", ""):
                saved_filenames.append(await optimize_and_save_image(img))
            
    final_images = []
    for item in media_order:
        if item.startswith("existing:"):
            final_images.append(item.split(":", 1)[1])
        elif item.startswith("new:"):
            idx = int(item.split(":")[1])
            if idx < len(saved_filenames): final_images.append(saved_filenames[idx])

    if not final_images:
        conn.close()
        raise HTTPException(status_code=400, detail="At least one image is required.")

    image_path_json = json.dumps(final_images)
    shared_int = 1 if is_shared == "true" else 0
    c.execute("""INSERT INTO prompts (id, title, prompt, author, tags, image_path, user_email, is_shared, forked_from) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (str(uuid.uuid4()), title, prompt.strip(), author, tags, image_path_json, user_email, shared_int, forked_from))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.put("/api/prompts/{prompt_id}")
async def edit_prompt(prompt_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    user_email = str(user.get('email', 'unknown'))

    form = await request.form()
    title = form.get("title")
    prompt = form.get("prompt")
    author = form.get("author")
    tags = form.get("tags")
    is_shared = form.get("is_shared", "false")
    new_images = form.getlist("new_images")
    media_order_raw = form.get("media_order", "[]")
    
    try: media_order = json.loads(media_order_raw)
    except: media_order = []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,))
    current_row = c.fetchone()
    if not current_row:
        conn.close()
        raise HTTPException(status_code=404)

    c.execute("SELECT id FROM prompts WHERE prompt = ? AND id != ?", (prompt.strip(), prompt_id))
    if c.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="Prompt text exists already.")

    c.execute("""INSERT INTO prompt_history (history_id, prompt_id, title, prompt, author, tags, image_path, edited_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
              (str(uuid.uuid4()), prompt_id, current_row['title'], current_row['prompt'], current_row['author'], current_row['tags'], current_row['image_path'], user_email))

    saved_filenames = []
    for img in new_images:
        if hasattr(img, "filename") and hasattr(img, "read"):
            if getattr(img, "filename", ""):
                saved_filenames.append(await optimize_and_save_image(img))
            
    final_images = []
    for item in media_order:
        if item.startswith("existing:"): final_images.append(item.split(":", 1)[1])
        elif item.startswith("new:"):
            idx = int(item.split(":")[1])
            if idx < len(saved_filenames): final_images.append(saved_filenames[idx])

    if not final_images:
        conn.close()
        raise HTTPException(status_code=400, detail="At least one image is required.")

    image_path_json = json.dumps(final_images)
    is_owner = (current_row['user_email'] == user_email)
    shared_int = 1 if is_shared == "true" else 0 if is_owner else current_row['is_shared']

    c.execute("""UPDATE prompts SET title = ?, prompt = ?, author = ?, tags = ?, image_path = ?, is_shared = ? WHERE id = ?""", 
              (title, prompt.strip(), author, tags, image_path_json, shared_int, prompt_id))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.delete("/api/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    user_email = str(user.get('email', 'unknown'))
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_email FROM prompts WHERE id = ?", (prompt_id,))
    row = c.fetchone()
    
    # Check if user is owner or admin
    if not row or (row[0] != user_email and not is_admin(user)):
        conn.close()
        raise HTTPException(status_code=403, detail="Not authorized to delete this prompt.")
        
    c.execute("SELECT image_path FROM prompts WHERE id = ?", (prompt_id,))
    cur_img = c.fetchone()
    
    files_to_delete = set()
    if cur_img and cur_img[0]:
        try: files_to_delete.update(json.loads(str(cur_img[0])))
        except: files_to_delete.add(str(cur_img[0]))
    c.execute("SELECT image_path FROM prompt_history WHERE prompt_id = ?", (prompt_id,))
    for h_row in c.fetchall():
        if h_row[0]:
            try: files_to_delete.update(json.loads(str(h_row[0])))
            except: files_to_delete.add(str(h_row[0]))
            
    for f in files_to_delete:
        c.execute("SELECT count(*) FROM prompts WHERE image_path LIKE ?", (f'%{f}%',))
        usage_count = c.fetchone()[0]
        if usage_count <= 1:
            fp = os.path.join(IMG_DIR, f)
            if os.path.exists(fp): os.remove(fp)

    c.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
    c.execute("DELETE FROM prompt_history WHERE prompt_id = ?", (prompt_id,))
    c.execute("DELETE FROM favorites WHERE prompt_id = ?", (prompt_id,)) 
    conn.commit()
    conn.close()
    return {"status": "deleted"}

# --- BULK API ROUTES ---
@app.post("/api/prompts/bulk/delete")
async def bulk_delete(request: Request, prompt_ids: str = Form(...)):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    user_email = str(user.get('email', 'unknown'))
    user_is_admin = is_admin(user)
    
    ids = [i.strip() for i in prompt_ids.split(',') if i.strip()]
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    for pid in ids:
        c.execute("SELECT user_email, image_path FROM prompts WHERE id = ?", (pid,))
        row = c.fetchone()
        
        if row and (row[0] == user_email or user_is_admin): 
            files_to_delete = set()
            if row[1]:
                try: files_to_delete.update(json.loads(str(row[1])))
                except: files_to_delete.add(str(row[1]))
            c.execute("SELECT image_path FROM prompt_history WHERE prompt_id = ?", (pid,))
            for h_row in c.fetchall():
                if h_row[0]:
                    try: files_to_delete.update(json.loads(str(h_row[0])))
                    except: files_to_delete.add(str(h_row[0]))
                    
            for f in files_to_delete:
                c.execute("SELECT count(*) FROM prompts WHERE image_path LIKE ?", (f'%{f}%',))
                usage_count = c.fetchone()[0]
                if usage_count <= 1:
                    fp = os.path.join(IMG_DIR, f)
                    if os.path.exists(fp): os.remove(fp)

            c.execute("DELETE FROM prompts WHERE id = ?", (pid,))
            c.execute("DELETE FROM prompt_history WHERE prompt_id = ?", (pid,))
            c.execute("DELETE FROM favorites WHERE prompt_id = ?", (pid,)) 
            
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.post("/api/prompts/bulk/tag")
async def bulk_tag(request: Request, prompt_ids: str = Form(...), new_tag: str = Form(...)):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    user_email = str(user.get('email', 'unknown'))
    user_is_admin = is_admin(user)
    
    ids = [i.strip() for i in prompt_ids.split(',') if i.strip()]
    clean_tag = new_tag.strip()
    if not clean_tag: return {"status": "success"}
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    for pid in ids:
        c.execute("SELECT * FROM prompts WHERE id = ?", (pid,))
        current = c.fetchone()
        if current and (current['user_email'] == user_email or current['is_shared'] == 1 or user_is_admin):
            tags_str = str(current['tags'] or "")
            current_tags = [t.strip() for t in tags_str.split(',') if t.strip()]
            
            if clean_tag not in current_tags:
                current_tags.append(clean_tag)
                new_tags_str = ", ".join(current_tags)
                
                c.execute("""INSERT INTO prompt_history (history_id, prompt_id, title, prompt, author, tags, image_path, edited_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                          (str(uuid.uuid4()), pid, current['title'], current['prompt'], current['author'], current['tags'], current['image_path'], user_email))
                
                c.execute("UPDATE prompts SET tags = ? WHERE id = ?", (new_tags_str, pid))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.post("/api/prompts/bulk/auto-tag")
async def bulk_auto_tag(request: Request, prompt_ids: str = Form(...), language: str = Form("English")):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    user_email = str(user.get('email', 'unknown'))
    user_is_admin = is_admin(user)
    
    ids = [i.strip() for i in prompt_ids.split(',') if i.strip()]
    if not ids: return {"status": "success"}
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    client = genai.Client()
    
    for pid in ids:
        c.execute("SELECT * FROM prompts WHERE id = ?", (pid,))
        current = c.fetchone()
        
        if current and (current['user_email'] == user_email or current['is_shared'] == 1 or user_is_admin) and current['prompt']:
            prompt_text = str(current['prompt'])
            instruction = f"""
            Analyze this prompt for AI image generation and create 3 to 6 relevant, short tags (e.g., 3d, cyberpunk, portrait).
            Output EXACTLY a comma-separated list of tags in lowercase. No markdown, no further explanations.
            The tags MUST be written in the following language: {language}.
            Prompt: {prompt_text}
            """
            try:
                response = client.models.generate_content(
                    model='gemma-3-27b-it',
                    contents=instruction
                )
                new_tags_raw = response.text.strip().replace('\n', '').replace('"', '')
                new_tags = [t.strip() for t in new_tags_raw.split(',') if t.strip()]
                
                tags_str = str(current['tags'] or "")
                current_tags = [t.strip() for t in tags_str.split(',') if t.strip()]
                
                added = False
                for nt in new_tags:
                    if nt not in current_tags:
                        current_tags.append(nt)
                        added = True
                        
                if added:
                    new_tags_str = ", ".join(current_tags)
                    c.execute("""INSERT INTO prompt_history (history_id, prompt_id, title, prompt, author, tags, image_path, edited_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                              (str(uuid.uuid4()), pid, current['title'], current['prompt'], current['author'], current['tags'], current['image_path'], user_email))
                    c.execute("UPDATE prompts SET tags = ? WHERE id = ?", (new_tags_str, pid))
                    conn.commit()
            except Exception as e:
                print(f"Error auto-tagging prompt {pid}: {e}")
                
    conn.close()
    return {"status": "success"}

@app.post("/api/prompts/{prompt_id}/copy")
async def increment_copy(prompt_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE prompts SET copy_count = COALESCE(copy_count, 0) + 1 WHERE id = ?", (prompt_id,))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.post("/api/prompts/{prompt_id}/favorite")
async def toggle_favorite(prompt_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    user_email = str(user.get('email', 'unknown'))
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT 1 FROM favorites WHERE user_email = ? AND prompt_id = ?", (user_email, prompt_id))
    is_fav = c.fetchone()
    if is_fav:
        c.execute("DELETE FROM favorites WHERE user_email = ? AND prompt_id = ?", (user_email, prompt_id))
        status = False
    else:
        c.execute("INSERT INTO favorites (user_email, prompt_id) VALUES (?, ?)", (user_email, prompt_id))
        status = True
    conn.commit()
    conn.close()
    return {"is_favorite": status}

@app.post("/api/extract-metadata")
async def extract_metadata(request: Request, image: Optional[UploadFile] = File(None), existing_image: Optional[str] = Form(None)):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    
    image_data = None
    if image and hasattr(image, "filename") and image.filename:
        image_data = await image.read()
    elif existing_image:
        safe_name = os.path.basename(existing_image)
        file_path = os.path.join(IMG_DIR, safe_name)
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                image_data = f.read()
                
    if not image_data:
        raise HTTPException(status_code=400, detail="No image provided")
        
    return {"extracted_prompt": extract_metadata_from_image(image_data)}

@app.post("/api/tags/auto")
async def auto_generate_tags(request: Request, prompt: str = Form(...), language: str = Form("English")):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    if not prompt: return {"tags": ""}

    try:
        client = genai.Client()
        instruction = f"""
        Analyze this prompt for AI image generation and create 3 to 6 relevant, short tags (e.g., 3d, cyberpunk, portrait).
        Output EXACTLY a comma-separated list of tags in lowercase. No markdown, no further explanations.
        The tags MUST be written in the following language: {language}.
        Prompt: {prompt}
        """

        response = client.models.generate_content(
            model='gemma-3-27b-it',
            contents=instruction
        )
        tags = response.text.strip().replace('\n', '').replace('"', '')
        return {"tags": tags}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/title/auto")
async def auto_generate_title(request: Request, prompt: str = Form(...), language: str = Form("English")):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    if not prompt: return {"title": ""}

    try:
        client = genai.Client()
        instruction = f"""
        Create a short, descriptive title (3 to 8 words) for the following AI image generation prompt.
        The title should capture the main subject and mood of the image.
        Output ONLY the title text, nothing else. No quotes, no markdown, no explanation.
        The title MUST be written in the following language: {language}.
        Prompt: {prompt}
        """

        response = client.models.generate_content(
            model='gemma-3-27b-it',
            contents=instruction
        )
        title = response.text.strip().replace('\n', '').replace('"', '')
        return {"title": title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/prompts/{prompt_id}/comments")
async def get_comments(prompt_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    user_email = str(user.get('email', 'unknown'))
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT c.*,
               COUNT(v.comment_id) as downvote_count,
               MAX(CASE WHEN v.user_email = ? THEN 1 ELSE 0 END) as user_downvoted
        FROM comments c
        LEFT JOIN comment_votes v ON c.id = v.comment_id
        WHERE c.prompt_id = ?
        GROUP BY c.id
        ORDER BY c.created_at ASC
    """, (user_email, prompt_id))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    
    top_level = [r for r in rows if not r['parent_id']]
    replies_map = {}
    for r in rows:
        if r['parent_id']:
            replies_map.setdefault(r['parent_id'], []).append(r)
            
    for row in top_level:
        row['replies'] = replies_map.get(row['id'], [])
        row['is_mine'] = (row['author_email'] == user_email)
        row['user_downvoted'] = bool(row['user_downvoted'])
        for reply in row['replies']:
            reply['is_mine'] = (reply['author_email'] == user_email)
            reply['user_downvoted'] = bool(reply['user_downvoted'])
            reply['replies'] = []
    return top_level

@app.post("/api/prompts/{prompt_id}/comments")
async def add_comment(prompt_id: str, request: Request, background_tasks: BackgroundTasks,
                      body: str = Form(...), parent_id: Optional[str] = Form(None)):
    try:
        user = request.session.get('user')
        if not user: raise HTTPException(status_code=401, detail="Unauthorized")
        
        user_email = str(user.get('email', 'unknown'))
        
        body = body.strip()
        if not body: raise HTTPException(status_code=400, detail="Comment cannot be empty.")
        
        parent_id = parent_id.strip() if parent_id else None

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("SELECT title, user_email FROM prompts WHERE id = ?", (prompt_id,))
        prompt_row = c.fetchone()
        if not prompt_row:
            conn.close()
            raise HTTPException(status_code=404, detail="Prompt not found.")

        parent_row = None
        if parent_id:
            c.execute("SELECT author_email FROM comments WHERE id = ? AND prompt_id = ?", (parent_id, prompt_id))
            parent_row = c.fetchone()
            if not parent_row:
                conn.close()
                raise HTTPException(status_code=404, detail="Parent comment not found.")

        comment_id = str(uuid.uuid4())
        author_name = str(user.get('name') or user.get('preferred_username') or user_email)
        
        c.execute("INSERT INTO comments (id, prompt_id, parent_id, author_email, author_name, body) VALUES (?, ?, ?, ?, ?, ?)",
                  (comment_id, prompt_id, parent_id, user_email, author_name, body))
        conn.commit()
        conn.close()

        prompt_title = str(prompt_row['title'] or 'Untitled')
        commenter = author_name
        base_url = str(request.base_url).rstrip('/')
        prompt_link = f"{base_url}/?open_comments={prompt_id}"

        if parent_row and parent_row['author_email'] and parent_row['author_email'] != user_email:
            notify_to = str(parent_row['author_email'])
            subject = f'New reply to your comment on "{prompt_title}"'
            html_body = (f'<p><b>{html_mod.escape(commenter)}</b> replied to your comment on '
                         f'<b>"{html_mod.escape(prompt_title)}"</b>:</p>'
                         f'<blockquote style="border-left:3px solid #f59e0b;padding-left:12px;color:#aaa">'
                         f'{html_mod.escape(str(body))}</blockquote>'
                         f'<p><a href="{prompt_link}" style="display:inline-block;padding:10px 15px;background:#eab308;color:#000;text-decoration:none;border-radius:5px;font-weight:bold;margin-top:10px;">View Reply</a></p>')
            
            background_tasks.add_task(_send_email_sync, notify_to, subject, html_body)
            
        elif not parent_id and prompt_row['user_email'] and prompt_row['user_email'] != user_email:
            notify_to = str(prompt_row['user_email'])
            subject = f'New comment on your prompt "{prompt_title}"'
            html_body = (f'<p><b>{html_mod.escape(commenter)}</b> commented on your prompt '
                         f'<b>"{html_mod.escape(prompt_title)}"</b>:</p>'
                         f'<blockquote style="border-left:3px solid #f59e0b;padding-left:12px;color:#aaa">'
                         f'{html_mod.escape(str(body))}</blockquote>'
                         f'<p><a href="{prompt_link}" style="display:inline-block;padding:10px 15px;background:#eab308;color:#000;text-decoration:none;border-radius:5px;font-weight:bold;margin-top:10px;">View Comment</a></p>')
                         
            background_tasks.add_task(_send_email_sync, notify_to, subject, html_body)

        return {"status": "ok", "id": comment_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

@app.post("/api/comments/{comment_id}/downvote")
async def toggle_downvote(comment_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    user_email = str(user.get('email', 'unknown'))
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT 1 FROM comment_votes WHERE comment_id = ? AND user_email = ?",
              (comment_id, user_email))
    if c.fetchone():
        c.execute("DELETE FROM comment_votes WHERE comment_id = ? AND user_email = ?",
                  (comment_id, user_email))
        voted = False
    else:
        c.execute("INSERT INTO comment_votes (comment_id, user_email) VALUES (?, ?)",
                  (comment_id, user_email))
        voted = True
    conn.commit()
    conn.close()
    return {"downvoted": voted}

@app.delete("/api/comments/{comment_id}")
async def delete_comment(comment_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    user_email = str(user.get('email', 'unknown'))
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT author_email FROM comments WHERE id = ?", (comment_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Comment not found.")
        
    if row[0] != user_email and not is_admin(user):
        conn.close()
        raise HTTPException(status_code=403, detail="Not authorized.")
        
    c.execute("DELETE FROM comments WHERE id = ? OR parent_id = ?", (comment_id, comment_id))
    c.execute("DELETE FROM comment_votes WHERE comment_id = ?", (comment_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}

@app.post("/api/tags/merge")
async def merge_tags(request: Request, old_tag: str = Form(...), new_tag: str = Form(...)):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    old_tag = old_tag.strip()
    new_tag = new_tag.strip()
    if not old_tag or not new_tag: raise HTTPException(status_code=400)
        
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT id, tags FROM prompts")
    rows = c.fetchall()
    
    for row in rows:
        if not row['tags']: continue
        current_tags = [t.strip() for t in str(row['tags']).split(',')]
        if old_tag in current_tags:
            updated_tags = []
            for t in current_tags:
                if t == old_tag:
                    if new_tag not in updated_tags: updated_tags.append(new_tag)
                else:
                    if t not in updated_tags: updated_tags.append(t)
            c.execute("UPDATE prompts SET tags = ? WHERE id = ?", (", ".join(updated_tags), row['id']))
    conn.commit()
    conn.close()
    return {"status": "success"}

# --- COLLECTIONS API ---
@app.get("/api/collections")
def get_collections(request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT col.id, col.name, col.created_at, COUNT(cp.prompt_id) as prompt_count
        FROM collections col
        LEFT JOIN collection_prompts cp ON col.id = cp.collection_id
        WHERE col.user_email = ?
        GROUP BY col.id
        ORDER BY col.created_at ASC
    """, (user['email'],))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows

@app.post("/api/collections")
async def create_collection(request: Request, name: str = Form(...)):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    name = name.strip()
    if not name: raise HTTPException(status_code=400, detail="Name is required.")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    new_id = str(uuid.uuid4())
    c.execute("INSERT INTO collections (id, name, user_email) VALUES (?, ?, ?)", (new_id, name, user['email']))
    conn.commit()
    conn.close()
    return {"id": new_id, "name": name, "prompt_count": 0}

@app.put("/api/collections/{collection_id}")
async def rename_collection(collection_id: str, request: Request, name: str = Form(...)):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    name = name.strip()
    if not name: raise HTTPException(status_code=400, detail="Name is required.")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_email FROM collections WHERE id = ?", (collection_id,))
    row = c.fetchone()
    if not row or row[0] != user['email']:
        conn.close()
        raise HTTPException(status_code=403)
    c.execute("UPDATE collections SET name = ? WHERE id = ?", (name, collection_id))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.delete("/api/collections/{collection_id}")
async def delete_collection(collection_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_email FROM collections WHERE id = ?", (collection_id,))
    row = c.fetchone()
    if not row or row[0] != user['email']:
        conn.close()
        raise HTTPException(status_code=403)
    c.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
    c.execute("DELETE FROM collection_prompts WHERE collection_id = ?", (collection_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}

@app.post("/api/collections/{collection_id}/prompts")
async def add_to_collection(collection_id: str, request: Request, prompt_id: str = Form(...)):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_email FROM collections WHERE id = ?", (collection_id,))
    row = c.fetchone()
    if not row or row[0] != user['email']:
        conn.close()
        raise HTTPException(status_code=403)
    c.execute("INSERT OR IGNORE INTO collection_prompts (collection_id, prompt_id) VALUES (?, ?)", (collection_id, prompt_id))
    conn.commit()
    conn.close()
    return {"status": "success"}

@app.delete("/api/collections/{collection_id}/prompts/{prompt_id}")
async def remove_from_collection(collection_id: str, prompt_id: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_email FROM collections WHERE id = ?", (collection_id,))
    row = c.fetchone()
    if not row or row[0] != user['email']:
        conn.close()
        raise HTTPException(status_code=403)
    c.execute("DELETE FROM collection_prompts WHERE collection_id = ? AND prompt_id = ?", (collection_id, prompt_id))
    conn.commit()
    conn.close()
    return {"status": "success"}

# --- EXPORT & IMPORT API ---
@app.get("/api/export/{format}")
def export_data(format: str, request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT * FROM prompts WHERE is_shared = 1 OR user_email = ?", (user['email'],))
    prompts_rows = [dict(r) for r in c.fetchall()]
    
    prompt_ids = set([r['id'] for r in prompts_rows])
    c.execute("SELECT * FROM prompt_history")
    all_history = [dict(r) for r in c.fetchall()]
    history_rows = [h for h in all_history if h['prompt_id'] in prompt_ids]
    
    conn.close()
    
    export_payload = {
        "prompts": prompts_rows,
        "history": history_rows
    }

    if format == "json":
        return Response(content=json.dumps(export_payload, indent=2), media_type="application/json", headers={"Content-Disposition": "attachment; filename=nanobanana_backup.json"})
    elif format == "csv":
        output = io.StringIO()
        if prompts_rows:
            writer = csv.DictWriter(output, fieldnames=prompts_rows[0].keys())
            writer.writeheader()
            writer.writerows(prompts_rows)
        return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=nanobanana_backup.csv"})
    elif format == "zip":
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("backup.json", json.dumps(export_payload, indent=2))
            
            images_added = set()
            
            for row in prompts_rows + history_rows:
                try:
                    imgs = json.loads(row['image_path'])
                    if not isinstance(imgs, list): imgs = [row['image_path']]
                except:
                    imgs = [row['image_path']]
                
                for img in imgs:
                    if img and img not in images_added:
                        img_full_path = os.path.join(IMG_DIR, img)
                        if os.path.exists(img_full_path):
                            zip_file.write(img_full_path, arcname=f"images/{img}")
                            images_added.add(img)
                            
        return Response(content=zip_buffer.getvalue(), media_type="application/zip", headers={"Content-Disposition": "attachment; filename=nanobanana_full_backup.zip"})
        
    raise HTTPException(status_code=400)

@app.post("/api/import")
async def import_data(request: Request, file: UploadFile = File(...)):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=401)
    
    filename = file.filename.lower()
    data_payload = {}
    
    if filename.endswith(".zip"):
        try:
            with zipfile.ZipFile(file.file) as z:
                json_filename = "backup.json" if "backup.json" in z.namelist() else "prompts.json"
                if json_filename not in z.namelist():
                    raise HTTPException(status_code=400, detail="Invalid ZIP: Missing backup.json")
                
                json_data = json.loads(z.read(json_filename))
                if isinstance(json_data, dict): data_payload = json_data
                else: data_payload = {"prompts": json_data, "history": []}
                
                for file_info in z.infolist():
                    if file_info.filename.startswith("images/") and not file_info.is_dir():
                        safe_filename = os.path.basename(file_info.filename)
                        if safe_filename:
                            target_path = os.path.join(IMG_DIR, safe_filename)
                            if not os.path.exists(target_path):
                                with open(target_path, "wb") as f_out:
                                    f_out.write(z.read(file_info.filename))
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file.")
    else:
        content = await file.read()
        if filename.endswith(".json"):
            try:
                json_data = json.loads(content.decode("utf-8"))
                if isinstance(json_data, dict): data_payload = json_data
                else: data_payload = {"prompts": json_data, "history": []}
            except: raise HTTPException(status_code=400, detail="Invalid JSON file.")
        elif filename.endswith(".csv"):
            try:
                reader = csv.DictReader(content.decode("utf-8").splitlines())
                data_payload = {"prompts": list(reader), "history": []}
            except: raise HTTPException(status_code=400, detail="Invalid CSV file.")
        else:
            raise HTTPException(status_code=400, detail="Only ZIP, JSON and CSV files are supported.")
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    added_prompts = 0
    added_history = 0
    
    prompts_to_insert = data_payload.get("prompts", [])
    history_to_insert = data_payload.get("history", [])
    id_map = {} 
    
    for row in prompts_to_insert:
        prompt_text = str(row.get("prompt", "")).strip()
        if not prompt_text: continue
        
        old_id = row.get("id")
        
        c.execute("SELECT id FROM prompts WHERE prompt = ?", (prompt_text,))
        existing = c.fetchone()
        if existing: 
            if old_id: id_map[old_id] = existing[0] 
            continue 
        
        new_id = str(uuid.uuid4())
        if old_id: id_map[old_id] = new_id
        
        title = row.get("title", "Imported Prompt")
        author = row.get("author", user['email'])
        tags = row.get("tags", "")
        image_path = row.get("image_path", "[]")
        user_email = row.get("user_email", user['email'])
        is_shared = int(row.get("is_shared", 0))
        copy_count = int(row.get("copy_count", 0))
        forked_from = row.get("forked_from", None)
        
        if forked_from and forked_from in id_map:
            forked_from = id_map[forked_from]
            
        c.execute("""INSERT INTO prompts 
                     (id, title, prompt, author, tags, image_path, user_email, is_shared, copy_count, forked_from) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (new_id, title, prompt_text, author, tags, image_path, user_email, is_shared, copy_count, forked_from))
        added_prompts += 1
        
    for h_row in history_to_insert:
        old_prompt_id = h_row.get("prompt_id")
        new_prompt_id = id_map.get(old_prompt_id)
        if not new_prompt_id: continue 
        
        hid = h_row.get("history_id", str(uuid.uuid4()))
        c.execute("SELECT history_id FROM prompt_history WHERE history_id = ?", (hid,))
        if c.fetchone(): continue 
        
        c.execute("""INSERT INTO prompt_history 
                     (history_id, prompt_id, title, prompt, author, tags, image_path, edited_by, edited_at) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (hid, new_prompt_id, h_row.get("title"), h_row.get("prompt"), h_row.get("author"), 
                   h_row.get("tags"), h_row.get("image_path"), h_row.get("edited_by"), h_row.get("edited_at")))
        added_history += 1
        
    conn.commit()
    conn.close()
    return {"status": "success", "added": added_prompts, "added_history": added_history}

@app.get("/share/{token}", response_class=HTMLResponse)
def view_shared_prompt(token: str):
    def _err(msg, code):
        return HTMLResponse(f"""<!DOCTYPE html><html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>NanoBanana Prompts</title><script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-gray-900 text-white min-h-screen flex items-center justify-center">
  <div class="text-center p-8">
    <h1 class="text-3xl font-bold text-yellow-400 mb-4">🍌 NanoBanana Prompts</h1>
    <p class="text-gray-400">{msg}</p>
  </div>
</body></html>""", status_code=code)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM share_links WHERE token = ?", (token,))
    link = c.fetchone()
    if not link:
        conn.close()
        return _err("Link not found.", 404)
        
    try:
        exp_date = datetime.strptime(link['expires_at'], "%Y-%m-%d %H:%M:%S")
    except Exception:
        exp_date = datetime.utcnow() - timedelta(days=1)
        
    if datetime.utcnow() > exp_date:
        conn.close()
        return _err("This share link has expired.", 410)

    c.execute("SELECT * FROM prompts WHERE id = ?", (link['prompt_id'],))
    prompt = c.fetchone()
    conn.close()
    if not prompt:
        return _err("Prompt not found.", 404)

    raw_image_path = prompt['image_path']
    if not raw_image_path:
        images = []
    else:
        try:
            parsed_images = json.loads(str(raw_image_path))
            if isinstance(parsed_images, list):
                images = parsed_images
            else:
                images = [str(parsed_images)]
        except Exception:
            images = [str(raw_image_path)]

    tags_str = str(prompt['tags'] or '')
    tags = [t.strip() for t in tags_str.split(',') if t.strip()]
    expires_label = exp_date.strftime("%Y-%m-%d %H:%M UTC")

    safe_title  = html_mod.escape(str(prompt['title'] or 'Untitled'))
    safe_author = html_mod.escape(str(prompt['author'] or 'Unknown'))
    tags_html   = ''.join(f'<span class="bg-gray-700 text-xs px-2 py-1 rounded">{html_mod.escape(str(t))}</span>' for t in tags)
    imgs_html   = ''.join(f'<img src="/images/{html_mod.escape(str(img))}" class="w-full rounded-lg object-cover aspect-square">' for img in images)
    
    prompt_str  = str(prompt['prompt'] or '')
    prompt_js   = json.dumps(prompt_str).replace("</", "<\\/")

    return HTMLResponse(f"""<!DOCTYPE html><html lang="en">
<head>
  <meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>{safe_title} — NanoBanana Prompts</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-white min-h-screen p-6">
  <div class="max-w-2xl mx-auto">
    <div class="text-center mb-8">
      <h1 class="text-2xl font-bold text-yellow-400">🍌 NanoBanana Prompts</h1>
      <p class="text-xs text-gray-500 mt-1">Shared link · expires {expires_label}</p>
    </div>
    <div class="grid gap-3 mb-6">{imgs_html}</div>
    <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <h2 class="text-2xl font-bold mb-1">{safe_title}</h2>
      <p class="text-sm text-gray-400 mb-3">by {safe_author}</p>
      <div class="flex flex-wrap gap-2 mb-4">{tags_html}</div>
      <div id="fillSection" class="hidden mb-4">
        <p class="text-sm text-blue-400 font-medium mb-3">Fill in the placeholders:</p>
        <div id="fillForm"></div>
      </div>
      <pre id="promptPreview" class="bg-gray-900 rounded p-4 text-sm text-gray-200 whitespace-pre-wrap break-words mb-4"></pre>
      <button id="copyBtn" onclick="copyFilled()"
              class="w-full bg-gray-700 hover:bg-gray-600 py-2 rounded font-bold transition-colors">
        📋 Copy Prompt
      </button>
    </div>
  </div>
  <script>
    const PROMPT_TEXT = {prompt_js};
    const HAS_PLACEHOLDERS = /\\[(.*?)\\]/.test(PROMPT_TEXT);

    function escapeHTML(s) {{
      return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
    }}
    function escapeRegExp(s) {{
      return s.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
    }}
    function updatePreview() {{
      let text = PROMPT_TEXT;
      document.querySelectorAll('.ph-input').forEach(inp => {{
        const ph = inp.dataset.ph;
        const val = inp.value || '[' + ph + ']';
        text = text.replace(new RegExp('\\\\[' + escapeRegExp(ph) + '\\\\]', 'g'), val);
      }});
      document.getElementById('promptPreview').textContent = text;
    }}
    function copyFilled() {{
      let text = PROMPT_TEXT;
      document.querySelectorAll('.ph-input').forEach(inp => {{
        const ph = inp.dataset.ph;
        const val = inp.value || '[' + ph + ']';
        text = text.replace(new RegExp('\\\\[' + escapeRegExp(ph) + '\\\\]', 'g'), val);
      }});
      const btn = document.getElementById('copyBtn');
      navigator.clipboard.writeText(text).then(() => {{
        btn.textContent = '✓ Copied!';
        setTimeout(() => {{ btn.textContent = HAS_PLACEHOLDERS ? '🧩 Fill & Copy' : '📋 Copy Prompt'; }}, 1500);
      }});
    }}
    if (HAS_PLACEHOLDERS) {{
      const phs = [...new Set([...PROMPT_TEXT.matchAll(/\\[(.*?)\\]/g)].map(m => m[1]))];
      const form = document.getElementById('fillForm');
      phs.forEach(ph => {{
        const d = document.createElement('div');
        d.innerHTML = '<label class="block text-xs font-bold text-blue-400 mb-1 tracking-wider uppercase">' + escapeHTML(ph) + '</label>'
          + '<input type="text" data-ph="' + escapeHTML(ph) + '" oninput="updatePreview()" '
          + 'class="ph-input w-full p-2 rounded bg-gray-700 border border-gray-600 text-white mb-3 focus:outline-none focus:border-blue-400" placeholder="Type here...">';
        form.appendChild(d);
      }});
      document.getElementById('fillSection').classList.remove('hidden');
      document.getElementById('copyBtn').textContent = '🧩 Fill & Copy';
    }}
    updatePreview();
  </script>
</body></html>""")

# --- FRONTEND ---
@app.get("/", response_class=HTMLResponse)
def get_html(request: Request):
    user = request.session.get('user')
    if not user:
        return """<!DOCTYPE html><html><head><script src="https://cdn.tailwindcss.com"></script></head>
        <body class="bg-gray-900 text-white h-screen flex items-center justify-center"><div class="text-center">
        <h1 class="text-4xl font-bold mb-6 text-yellow-400">🍌 NanoBanana Prompts</h1>
        <a href="/login" class="bg-yellow-500 hover:bg-yellow-600 text-black font-bold py-3 px-8 rounded">Login via OIDC</a>
        </div></body></html>"""

    admin_badge = '<span class="text-red-400 font-bold text-xs ml-2 border border-red-400 px-1 rounded align-middle" title="Admin Privileges Active">ADMIN</span>' if is_admin(user) else ''

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NanoBanana Prompts</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            #dropZone { transition: all 0.2s ease-in-out; }
            .autocomplete-list::-webkit-scrollbar { width: 6px; }
            .autocomplete-list::-webkit-scrollbar-thumb { background: #4B5563; border-radius: 4px; }
            .autocomplete-list::-webkit-scrollbar-track { background: #1F2937; }
            .flash-success { animation: flashGreen 1.5s ease-out; }
            @keyframes flashGreen {
                0% { border-color: #22c55e; background-color: rgba(34, 197, 94, 0.2); }
                100% { border-color: #4b5563; background-color: #374151; }
            }
            .hide-scrollbar::-webkit-scrollbar { display: none; }
            .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
            
            .bulk-checkbox-container input[type="checkbox"] {
                appearance: none;
                background-color: #374151;
                margin: 0;
                font: inherit;
                display: grid;
                place-content: center;
                transition: 0.2s ease-in-out;
            }
            .bulk-checkbox-container input[type="checkbox"]::before {
                content: "";
                width: 0.65em;
                height: 0.65em;
                transform: scale(0);
                transition: 120ms transform ease-in-out;
                box-shadow: inset 1em 1em white;
                background-color: transform;
                transform-origin: bottom left;
                clip-path: polygon(14% 44%, 0 65%, 50% 100%, 100% 16%, 80% 0%, 43% 62%);
            }
            .bulk-checkbox-container input[type="checkbox"]:checked {
                background-color: #a855f7;
                border-color: #a855f7;
            }
            .bulk-checkbox-container input[type="checkbox"]:checked::before { transform: scale(1); }
        </style>
    </head>
    <body class="bg-gray-900 text-white p-4 md:p-6 lg:p-8 font-sans pb-24">
        <div class="w-full max-w-[2400px] mx-auto">
            
            <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 md:mb-8 gap-4">
                <h1 class="text-3xl md:text-4xl font-bold text-yellow-400 cursor-pointer hover:text-yellow-300 transition-colors select-none" onclick="resetToHome()" title="Back to home">🍌 NanoBanana Prompts</h1>
                
                <div class="flex items-center gap-3 md:gap-4 flex-wrap">
                    <span id="promptCounter" class="bg-gray-800 text-yellow-400 font-bold px-3 py-1 rounded border border-gray-700 text-sm">
                        0 Prompts
                    </span>
                    <span class="text-gray-400 text-sm hidden md:inline">__USER_EMAIL__ __ADMIN_BADGE__</span>
                    <a href="/logout" class="text-red-400 hover:text-red-300 text-sm border border-red-400 hover:border-red-300 rounded px-3 py-1">Logout</a>
                </div>
            </div>
            
            <div class="flex flex-col lg:flex-row gap-4 mb-8">
                <div class="flex-grow relative">
                    <input type="text" id="searchInput" placeholder="Search text, tag:cyberpunk, -tag:3d, author:mike, is:mine..." 
                           class="w-full p-3 rounded bg-gray-800 border border-gray-700 focus:outline-none focus:border-yellow-400 pr-10"
                           onkeyup="triggerRenderReset()">
                    <button onclick="clearSearch()" class="absolute right-3 top-3 text-gray-400 hover:text-white font-bold" title="Clear search">✕</button>
                </div>
                
                <div class="flex gap-2 sm:gap-4 flex-wrap items-center relative z-20">
                    <select id="sortSelect" onchange="applySort()" class="bg-gray-800 border border-gray-700 text-white text-sm md:text-base rounded px-2 py-2 focus:outline-none focus:border-yellow-400 cursor-pointer">
                        <option value="newest">Newest First</option>
                        <option value="most_copied">Most Copied</option>
                        <option value="oldest">Oldest First</option>
                        <option value="least_copied">Least Copied</option>
                    </select>

                    <select id="layoutSelect" onchange="setLayout(parseInt(this.value))" class="bg-gray-800 border border-gray-700 text-white text-sm md:text-base rounded px-2 py-2 focus:outline-none focus:border-yellow-400 cursor-pointer">
                        <option value="1">1 Column</option>
                        <option value="2">2 Columns</option>
                        <option value="3">3 Columns</option>
                        <option value="4">4 Columns</option>
                        <option value="5">5 Columns</option>
                        <option value="6">6 Columns</option>
                        <option value="7">7 Columns</option>
                        <option value="8">8 Columns</option>
                    </select>

                    <button id="filterFavBtn" onclick="toggleFavFilter()" class="bg-gray-800 hover:bg-gray-700 text-gray-300 font-bold py-2 px-3 sm:px-4 rounded border border-gray-700 transition-colors text-sm md:text-base">
                        🤍 Favorites
                    </button>

                    <div class="flex gap-2">
                        <button onclick="openTagsModal()" class="bg-gray-800 hover:bg-gray-700 text-gray-300 font-bold py-2 px-3 sm:px-4 rounded border border-gray-700 transition-colors text-sm md:text-base">🏷️ Tags</button>
                        <button onclick="openAuthorsModal()" class="bg-gray-800 hover:bg-gray-700 text-gray-300 font-bold py-2 px-3 sm:px-4 rounded border border-gray-700 transition-colors text-sm md:text-base">👥 Authors</button>
                        <button onclick="openCollectionsModal()" class="bg-gray-800 hover:bg-gray-700 text-gray-300 font-bold py-2 px-3 sm:px-4 rounded border border-gray-700 transition-colors text-sm md:text-base">📁 Collections</button>
                        <button onclick="document.getElementById('exportModal').classList.remove('hidden')" class="bg-gray-800 hover:bg-gray-700 text-gray-300 font-bold py-2 px-3 sm:px-4 rounded border border-gray-700 transition-colors text-sm md:text-base" title="Manage Data">💾 Data</button>
                    </div>
                    
                    <button id="bulkModeBtn" onclick="toggleBulkMode()" class="bg-purple-700 hover:bg-purple-600 text-white font-bold py-2 px-3 sm:px-4 rounded transition-colors text-sm md:text-base border border-purple-600 shadow-lg flex items-center gap-2">
                        ☑️ Bulk Edit
                    </button>

                    <button onclick="openAddModal()" class="bg-yellow-500 hover:bg-yellow-600 text-black font-bold py-2 px-3 sm:px-6 rounded whitespace-nowrap text-sm md:text-base shadow-lg">
                        + Add Prompt
                    </button>
                </div>
            </div>

            <div id="activeCollectionBanner" class="hidden mb-4 bg-teal-900/30 border border-teal-600 rounded-lg px-4 py-3 flex items-center justify-between">
                <span class="text-teal-300 font-bold">📁 Filtering by collection: <span id="activeCollectionName" class="text-white"></span></span>
                <button onclick="clearCollectionFilter()" class="text-teal-400 hover:text-white transition-colors font-bold text-lg leading-none">✕</button>
            </div>

            <div class="md:hidden text-center mb-4">
                <span id="promptCounterMobile" class="text-yellow-400 font-bold text-sm bg-gray-800 px-3 py-1 rounded border border-gray-700">0 Prompts</span>
            </div>

            <div id="gallery" class="grid grid-cols-1 gap-4 md:gap-6 relative z-10"></div>
            
            <div id="loadingIndicator" class="hidden text-center py-8 text-gray-400">
                <span class="animate-pulse">Loading more prompts...</span>
            </div>
        </div>
        
        <div id="bulkActionBar" class="fixed bottom-0 left-0 right-0 bg-gray-800 border-t-4 border-purple-600 p-4 transform translate-y-full transition-transform z-50 flex justify-center shadow-[0_-10px_25px_-5px_rgba(0,0,0,0.5)]">
            <div class="w-full max-w-5xl flex justify-between items-center gap-4 flex-wrap">
                <div class="font-bold text-white text-lg"><span id="bulkCount" class="text-purple-400 text-2xl">0</span> Prompts Selected</div>
                <div class="flex gap-2 flex-wrap">
                    <button onclick="toggleBulkSelectAll()" id="bulkSelectAllBtn" class="bg-gray-700 hover:bg-gray-600 text-white px-3 py-2 rounded font-bold transition-colors">☑️ Select All</button>
                    <button onclick="bulkTag()" class="bg-blue-600 hover:bg-blue-500 text-white px-3 py-2 rounded font-bold transition-colors">🏷️ Add Tag</button>
                    <button onclick="bulkAutoTag(this)" class="bg-yellow-600 hover:bg-yellow-500 text-white px-3 py-2 rounded font-bold transition-colors">✨ Auto-Tag</button>
                    <button onclick="openBulkCollectionModal()" class="bg-teal-700 hover:bg-teal-600 text-white px-3 py-2 rounded font-bold transition-colors">📁 Collect</button>
                    <button onclick="bulkDelete()" class="bg-red-600 hover:bg-red-500 text-white px-3 py-2 rounded font-bold transition-colors">🗑️ Delete</button>
                    <button onclick="toggleBulkMode()" class="bg-gray-600 hover:bg-gray-500 text-white px-3 py-2 rounded font-bold transition-colors ml-2">Cancel</button>
                </div>
            </div>
        </div>

        <div id="templateModal" class="hidden fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
            <div class="bg-gray-800 p-6 rounded-lg w-full max-w-xl max-h-[90vh] overflow-y-auto flex flex-col">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-2xl font-bold text-yellow-400">🧩 Fill Variables</h2>
                    <button onclick="closeModal('templateModal')" class="text-gray-400 hover:text-white text-xl font-bold">✕</button>
                </div>
                <p class="text-sm text-gray-400 mb-4">Fill in the placeholders to complete your prompt.</p>

                <div id="templateInputsContainer" class="flex flex-col gap-3 mb-6"></div>

                <div class="bg-gray-900 p-4 rounded border border-gray-700 mb-6">
                    <h3 class="text-xs font-bold text-gray-500 mb-2 uppercase tracking-wide">Live Preview</h3>
                    <p id="templatePreview" class="text-sm text-gray-300 font-mono break-words"></p>
                </div>

                <input type="hidden" id="templateOriginalPrompt">
                <input type="hidden" id="templatePromptId">

                <div class="flex justify-between gap-2 mt-auto">
                    <button type="button" onclick="copyRawTemplate(this)" class="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors text-sm font-bold border border-gray-600">📋 Copy Raw</button>
                    <div class="flex gap-2">
                        <button type="button" onclick="closeModal('templateModal')" class="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded transition-colors text-sm">Cancel</button>
                        <button type="button" onclick="executeTemplateCopy(this)" class="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded transition-colors text-sm shadow-lg border border-blue-600">✨ Copy Filled</button>
                    </div>
                </div>
            </div>
        </div>

        <div id="lightboxModal" class="hidden fixed inset-0 z-[100] bg-black/95 flex items-center justify-center p-4" onclick="closeLightbox()">
            <button onclick="closeLightbox()" class="absolute top-4 right-6 text-gray-300 hover:text-yellow-400 text-4xl font-bold z-50 transition-colors focus:outline-none">✕</button>
            <button id="lightboxPrevBtn" onclick="lightboxNavigate(-1, event)" class="absolute left-4 top-1/2 -translate-y-1/2 text-white bg-black/50 hover:bg-black/80 rounded-full w-12 h-12 flex items-center justify-center text-2xl font-bold z-50 transition-colors focus:outline-none hidden">◀</button>
            <img id="lightboxImg" src="" class="max-w-full max-h-full object-contain shadow-2xl rounded select-none" onclick="event.stopPropagation()">
            <button id="lightboxNextBtn" onclick="lightboxNavigate(1, event)" class="absolute right-4 top-1/2 -translate-y-1/2 text-white bg-black/50 hover:bg-black/80 rounded-full w-12 h-12 flex items-center justify-center text-2xl font-bold z-50 transition-colors focus:outline-none hidden">▶</button>
        </div>

        <div id="exportModal" class="hidden fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
            <div class="bg-gray-800 p-6 rounded-lg w-full max-w-md">
                <div class="flex justify-between items-center mb-6">
                    <h2 class="text-2xl font-bold">Data Management</h2>
                    <button onclick="closeModal('exportModal')" class="text-gray-400 hover:text-white text-xl font-bold">✕</button>
                </div>
                
                <div class="mb-6 border-b border-gray-700 pb-6">
                    <h3 class="text-lg font-bold mb-2 text-yellow-400">⬇️ Export Backup</h3>
                    <p class="text-gray-400 text-sm mb-4">Download your prompt library. The ZIP file includes all images AND version history!</p>
                    <div class="flex flex-col gap-3">
                        <a href="/api/export/zip" download class="w-full bg-blue-700 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded text-center border border-blue-600 transition-colors">📦 Full Backup (.zip with images)</a>
                        <div class="flex gap-3">
                            <a href="/api/export/json" download class="flex-1 bg-gray-700 hover:bg-gray-600 text-white font-bold py-2 px-4 rounded text-center border border-gray-600 transition-colors text-sm">📄 Text (.json)</a>
                            <a href="/api/export/csv" download class="flex-1 bg-gray-700 hover:bg-gray-600 text-white font-bold py-2 px-4 rounded text-center border border-gray-600 transition-colors text-sm">📊 Text (.csv)</a>
                        </div>
                    </div>
                </div>

                <div>
                    <h3 class="text-lg font-bold mb-2 text-green-400">⬆️ Import / Restore</h3>
                    <p class="text-gray-400 text-sm mb-4">Upload a backup (.zip, .json, .csv). Restores prompts, images and history.</p>
                    <input type="file" id="importFile" accept=".zip,.json,.csv" class="w-full p-2 mb-3 rounded bg-gray-700 border border-gray-600 text-white text-sm">
                    <button onclick="importData()" id="importBtn" class="w-full bg-green-700 hover:bg-green-600 text-white font-bold py-2 px-4 rounded transition-colors">Upload & Import</button>
                </div>
            </div>
        </div>

        <div id="promptModal" class="hidden fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
            <div class="bg-gray-800 p-6 rounded-lg w-full max-w-lg max-h-[90vh] overflow-y-auto">
                <h2 id="modalTitle" class="text-2xl font-bold mb-4">Add New Prompt</h2>
                <form id="promptForm" onsubmit="submitForm(event)">
                    <input type="hidden" id="editPromptId" name="editPromptId" value="">
                    <input type="hidden" id="formForkedFrom" name="forked_from" value="">
                    
                    <div class="flex justify-between items-end mb-1">
                        <label class="text-sm font-medium text-gray-400">Title</label>
                        <button type="button" onclick="autoGenerateTitle(this)" class="text-xs bg-gray-700 hover:bg-yellow-500 hover:text-black text-gray-300 px-3 py-1 rounded transition-colors focus:outline-none whitespace-nowrap flex-shrink-0" title="Generate title from prompt text via Gemini">
                            ✨ Auto-Title
                        </button>
                    </div>
                    <input type="text" id="formTitle" name="title" placeholder="Title" required class="w-full p-2 mb-3 rounded bg-gray-700 border border-gray-600 text-white transition-colors">
                    
                    <div class="relative w-full mb-3">
                        <input type="text" id="formAuthor" name="author" placeholder="Author/Creator" required autocomplete="off" oninput="showAuthorSuggestions(this.value)" class="w-full p-2 rounded bg-gray-700 border border-gray-600 text-white">
                        <div id="authorSuggestions" class="autocomplete-list absolute z-10 w-full bg-gray-600 border border-gray-500 rounded mt-1 hidden max-h-40 overflow-y-auto shadow-lg text-sm"></div>
                    </div>
                    
                    <div class="flex justify-between items-end mb-1 mt-2">
                        <label class="text-sm font-medium text-gray-400">Tags</label>
                        <div class="flex gap-2">
                            <select id="tagLanguage" onchange="localStorage.setItem('nanobananaTagLanguage', this.value)" class="bg-gray-700 border border-gray-600 text-xs text-white rounded px-2 py-1 focus:outline-none focus:border-yellow-400 cursor-pointer">
                                <option value="English">🇬🇧 English</option>
                                <option value="German">🇩🇪 German</option>
                                <option value="Spanish">🇪🇸 Spanish</option>
                                <option value="French">🇫🇷 French</option>
                            </select>
                            <button type="button" onclick="autoGenerateTags(this)" class="text-xs bg-gray-700 hover:bg-yellow-500 hover:text-black text-gray-300 px-2 py-1 rounded transition-colors focus:outline-none">
                                ✨ Auto-Tag (Gemini)
                            </button>
                        </div>
                    </div>
                    <div class="relative w-full mb-3">
                        <input type="text" id="formTags" name="tags" placeholder="Tags (comma separated)" required autocomplete="off" oninput="showTagSuggestions(this.value)" class="w-full p-2 rounded bg-gray-700 border border-gray-600 text-white transition-colors">
                        <div id="tagSuggestions" class="autocomplete-list absolute z-10 w-full bg-gray-600 border border-gray-500 rounded mt-1 hidden max-h-40 overflow-y-auto shadow-lg text-sm"></div>
                    </div>
                    
                    <div class="flex justify-between items-end mb-1">
                        <label class="text-sm font-medium text-gray-400">Prompt Text</label>
                        <button type="button" onclick="forceExtractPrompt(this)" class="text-xs bg-gray-700 hover:bg-yellow-500 hover:text-black text-gray-300 px-2 py-1 rounded transition-colors focus:outline-none">
                            ✨ Extract from Cover Image
                        </button>
                    </div>
                    <textarea id="formPrompt" name="prompt" placeholder="The actual prompt... Use [PLACEHOLDERS] to create dynamic templates!" required class="w-full p-2 mb-3 rounded bg-gray-700 border border-gray-600 text-white h-32 transition-colors"></textarea>
                    
                    <div id="dropZone" class="w-full p-6 mb-2 rounded bg-gray-700 border-2 border-dashed border-gray-500 text-center cursor-pointer hover:border-yellow-400 hover:bg-gray-600 transition-colors" onclick="document.getElementById('hiddenFileInput').click()">
                        <input type="file" id="hiddenFileInput" multiple accept="image/*" class="hidden" onchange="handleFileSelect(event)">
                        <div class="text-gray-400 pointer-events-none">
                            <span class="text-3xl block mb-2">📸</span>
                            <span class="font-bold text-white">Drag & Drop, Paste multiple images</span><br>or click to browse<br>
                            <span class="text-xs text-yellow-500 mt-2 block">✨ Auto-Extracts metadata from the first image</span>
                        </div>
                    </div>
                    
                    <div id="mediaPreviews" class="flex gap-2 overflow-x-auto hide-scrollbar mb-4 py-2"></div>
                    
                    <div class="flex items-center mb-6">
                        <input type="checkbox" id="is_shared" name="is_shared" value="true" checked class="w-4 h-4 text-yellow-500 bg-gray-700 border-gray-600 rounded">
                        <label for="is_shared" class="ml-2 text-sm font-medium text-gray-300">Share with other users</label>
                    </div>

                    <div class="flex justify-end gap-2">
                        <button type="button" onclick="closeModal('promptModal')" class="px-4 py-2 bg-gray-600 rounded">Cancel</button>
                        <button type="submit" id="saveButton" class="px-4 py-2 bg-yellow-500 text-black font-bold rounded">Save</button>
                    </div>
                </form>
            </div>
        </div>

        <div id="historyModal" class="hidden fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
            <div class="bg-gray-800 p-6 rounded-lg w-full max-w-2xl max-h-[90vh] overflow-y-auto flex flex-col">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-2xl font-bold">Version History</h2>
                    <button onclick="closeModal('historyModal')" class="text-gray-400 hover:text-white text-xl font-bold">✕</button>
                </div>
                <div id="historyContainer" class="flex-grow flex flex-col gap-4"></div>
            </div>
        </div>

        <div id="authorsModal" class="hidden fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
            <div class="bg-gray-800 p-6 rounded-lg w-full max-w-lg max-h-[90vh] overflow-y-auto flex flex-col">
                <div class="flex justify-between items-center mb-6">
                    <h2 class="text-2xl font-bold">Authors Directory</h2>
                    <button onclick="closeModal('authorsModal')" class="text-gray-400 hover:text-white text-xl font-bold">✕</button>
                </div>
                <div id="authorsContainer" class="flex flex-wrap gap-2"></div>
            </div>
        </div>

        <div id="tagsModal" class="hidden fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
            <div class="bg-gray-800 p-6 rounded-lg w-full max-w-lg max-h-[90vh] overflow-y-auto flex flex-col">
                <div class="flex justify-between items-center mb-6">
                    <h2 class="text-2xl font-bold">Tags Manager</h2>
                    <button onclick="closeModal('tagsModal')" class="text-gray-400 hover:text-white text-xl font-bold">✕</button>
                </div>
                <div class="bg-gray-900 p-4 rounded mb-6 border border-gray-700">
                    <h3 class="text-lg font-bold mb-2 text-yellow-400">Merge Tags</h3>
                    <p class="text-xs text-gray-400 mb-3">Combine similar tags (e.g. '3d' and '3D'). Updates all related prompts instantly.</p>
                    <div class="flex flex-col sm:flex-row gap-2 items-start sm:items-center w-full">
                        <div class="relative w-full sm:flex-1">
                            <input type="text" id="mergeOldTag" placeholder="Search old tag..." autocomplete="off" oninput="showMergeOldTagSuggestions(this.value)" class="w-full p-2 rounded bg-gray-700 border border-gray-600 text-white text-sm focus:outline-none focus:border-yellow-400">
                            <div id="mergeOldTagSuggestions" class="autocomplete-list absolute z-10 w-full bg-gray-600 border border-gray-500 rounded mt-1 hidden max-h-40 overflow-y-auto shadow-lg text-sm"></div>
                        </div>
                        <span class="text-gray-400 hidden sm:inline px-2">➔</span>
                        <div class="relative w-full sm:flex-1">
                            <input type="text" id="mergeNewTag" placeholder="Search or type new tag..." autocomplete="off" oninput="showMergeNewTagSuggestions(this.value)" class="w-full p-2 rounded bg-gray-700 border border-gray-600 text-white text-sm focus:outline-none focus:border-yellow-400">
                            <div id="mergeNewTagSuggestions" class="autocomplete-list absolute z-10 w-full bg-gray-600 border border-gray-500 rounded mt-1 hidden max-h-40 overflow-y-auto shadow-lg text-sm"></div>
                        </div>
                        <button onclick="mergeTags()" id="mergeTagBtn" class="w-full sm:w-auto bg-yellow-500 hover:bg-yellow-600 text-black font-bold py-2 px-4 rounded transition-colors text-sm">Merge</button>
                    </div>
                </div>
                <h3 class="text-lg font-bold mb-3 border-b border-gray-700 pb-2">All Tags</h3>
                <div id="tagsContainer" class="flex flex-wrap gap-2"></div>
            </div>
        </div>

        <div id="collectionsModal" class="hidden fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
            <div class="bg-gray-800 p-6 rounded-lg w-full max-w-lg max-h-[90vh] overflow-y-auto flex flex-col">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-2xl font-bold">📁 Collections</h2>
                    <button onclick="closeModal('collectionsModal')" class="text-gray-400 hover:text-white text-xl font-bold">✕</button>
                </div>
                <p class="text-xs text-gray-400 mb-4">Group prompts into folders for easier navigation. Collections are personal — only you can see them.</p>
                <div class="flex gap-2 mb-6">
                    <input type="text" id="newCollectionName" placeholder="New collection name..."
                           class="flex-grow p-2 rounded bg-gray-700 border border-gray-600 text-white focus:outline-none focus:border-teal-400"
                           onkeydown="if(event.key==='Enter') createCollection()">
                    <button onclick="createCollection()" class="bg-teal-600 hover:bg-teal-500 text-white font-bold py-2 px-4 rounded transition-colors">Create</button>
                </div>
                <div id="collectionsListContainer" class="flex flex-col gap-2"></div>
            </div>
        </div>

        <div id="collectionAssignModal" class="hidden fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-[60]">
            <div class="bg-gray-800 p-6 rounded-lg w-full max-w-sm max-h-[80vh] overflow-y-auto flex flex-col">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-xl font-bold">📁 Add to Collection</h2>
                    <button onclick="closeModal('collectionAssignModal')" class="text-gray-400 hover:text-white text-xl font-bold">✕</button>
                </div>
                <input type="hidden" id="assignPromptId">
                <div id="collectionAssignList" class="flex flex-col gap-2 mb-4"></div>
                <p class="text-xs text-gray-500 text-center">Changes save immediately. Create collections via 📁 Collections.</p>
            </div>
        </div>

        <div id="bulkCollectionModal" class="hidden fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-[60]">
            <div class="bg-gray-800 p-6 rounded-lg w-full max-w-sm flex flex-col">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-xl font-bold">📁 Add to Collection</h2>
                    <button onclick="closeModal('bulkCollectionModal')" class="text-gray-400 hover:text-white text-xl font-bold">✕</button>
                </div>
                <p class="text-sm text-gray-400 mb-4">Choose a collection to add <span id="bulkCollectCount" class="text-purple-400 font-bold">0</span> selected prompts to:</p>
                <div id="bulkCollectionList" class="flex flex-col gap-2"></div>
            </div>
        </div>

        <div id="shareLinkModal" class="hidden fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
            <div class="bg-gray-800 p-6 rounded-lg w-full max-w-md flex flex-col gap-4">
                <div class="flex justify-between items-center">
                    <h2 class="text-xl font-bold">🔗 Share Link</h2>
                    <button onclick="closeModal('shareLinkModal')" class="text-gray-400 hover:text-white text-xl font-bold">✕</button>
                </div>
                <p class="text-sm text-gray-400">Create a time-limited link anyone can open without an account.</p>
                <div class="flex gap-2">
                    <select id="shareLinkExpiry" class="flex-grow bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm focus:outline-none">
                        <option value="1">1 hour</option>
                        <option value="6">6 hours</option>
                        <option value="24" selected>24 hours</option>
                        <option value="72">3 days</option>
                        <option value="168">7 days</option>
                        <option value="720">30 days</option>
                    </select>
                    <button onclick="createShareLink()" class="bg-yellow-500 hover:bg-yellow-600 text-black font-bold px-4 py-2 rounded text-sm transition-colors">Generate</button>
                </div>
                <div id="shareLinkResult" class="hidden flex gap-2">
                    <input id="shareLinkUrl" type="text" readonly class="flex-grow bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-gray-200 font-mono focus:outline-none">
                    <button id="shareLinkCopyBtn" onclick="copyShareLink()" class="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded text-sm font-bold transition-colors">📋</button>
                </div>
                <div>
                    <h3 class="text-sm font-bold text-gray-300 mb-2">Active links</h3>
                    <div id="shareLinkList" class="flex flex-col gap-2 max-h-48 overflow-y-auto"></div>
                </div>
            </div>
        </div>
        
        <div id="commentsModal" class="hidden fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
            <div class="bg-gray-800 p-6 rounded-lg w-full max-w-lg flex flex-col gap-4" style="max-height:90vh">
                <div class="flex justify-between items-center flex-shrink-0">
                    <h2 class="text-xl font-bold">💬 Comments</h2>
                    <button onclick="closeModal('commentsModal')" class="text-gray-400 hover:text-white text-xl font-bold">✕</button>
                </div>
                <div id="commentsList" class="flex flex-col gap-3 overflow-y-auto flex-grow min-h-0"></div>
                <div class="border-t border-gray-700 pt-3 flex-shrink-0">
                    <textarea id="newCommentBody" placeholder="Write a comment…" rows="3"
                        class="w-full bg-gray-700 border border-gray-600 rounded p-2 text-sm text-white resize-none focus:outline-none focus:border-yellow-400 mb-2"></textarea>
                    <div class="flex justify-end">
                        <button onclick="submitComment()"
                            class="bg-yellow-500 hover:bg-yellow-600 text-black font-bold px-4 py-2 rounded text-sm transition-colors">
                            Post Comment
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let globalPrompts = [];
            let displayPrompts = [];
            let uniqueAuthors = [];
            let uniqueTags = [];
            
            let renderIndex = 0;
            const BATCH_SIZE = 24; 
            
            let showOnlyFavorites = false;
            let mediaItems = []; 
            
            let isBulkMode = false;
            let selectedPrompts = new Set();

            let collections = [];
            let activeCollectionId = null;
            
            const IS_ADMIN = __IS_ADMIN__;

            function escapeHTML(str) {
                if (!str) return '';
                return str.replace(/[&<>'"]/g, 
                    tag => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'}[tag] || tag)
                );
            }

            function encodeForJS(str) {
                if (!str) return '';
                return encodeURIComponent(str).replace(/'/g, "%27");
            }
            
            function escapeRegExp(string) {
              return string.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\$&');
            }

            // --- BULK MODE LOGIK ---
            function toggleBulkMode() {
                isBulkMode = !isBulkMode;
                selectedPrompts.clear();
                
                const btn = document.getElementById('bulkModeBtn');
                const bar = document.getElementById('bulkActionBar');
                const selectAllBtn = document.getElementById('bulkSelectAllBtn');
                
                if (isBulkMode) {
                    btn.classList.replace('bg-purple-700', 'bg-purple-500');
                    bar.classList.remove('translate-y-full');
                    selectAllBtn.innerText = "☑️ Select All";
                } else {
                    btn.classList.replace('bg-purple-500', 'bg-purple-700');
                    bar.classList.add('translate-y-full');
                }
                
                const currentScroll = window.scrollY;
                document.getElementById('gallery').innerHTML = '';
                renderIndex = 0;
                renderNextBatch();
                
                setTimeout(() => window.scrollTo(0, currentScroll), 10);
                updateBulkCount();
            }

            function toggleBulkSelection(id, isChecked) {
                if (isChecked) selectedPrompts.add(id);
                else selectedPrompts.delete(id);
                updateBulkCount();
                updateSelectAllButtonState();
            }

            function updateBulkCount() {
                document.getElementById('bulkCount').innerText = selectedPrompts.size;
            }

            function toggleBulkSelectAll() {
                const currentlyVisibleIds = displayPrompts.map(p => p.id);
                const allSelected = currentlyVisibleIds.length > 0 && currentlyVisibleIds.every(id => selectedPrompts.has(id));
                
                if (allSelected) {
                    currentlyVisibleIds.forEach(id => selectedPrompts.delete(id));
                } else {
                    currentlyVisibleIds.forEach(id => selectedPrompts.add(id));
                }
                
                const currentScroll = window.scrollY;
                document.getElementById('gallery').innerHTML = '';
                renderIndex = 0;
                renderNextBatch();
                setTimeout(() => window.scrollTo(0, currentScroll), 10);
                
                updateBulkCount();
                updateSelectAllButtonState();
            }
            
            function updateSelectAllButtonState() {
                const btn = document.getElementById('bulkSelectAllBtn');
                if(!btn) return;
                const currentlyVisibleIds = displayPrompts.map(p => p.id);
                if (currentlyVisibleIds.length > 0 && currentlyVisibleIds.every(id => selectedPrompts.has(id))) {
                    btn.innerText = "☐ Deselect All";
                } else {
                    btn.innerText = "☑️ Select All";
                }
            }

            async function bulkDelete() {
                if (selectedPrompts.size === 0) return alert("Please select at least one prompt.");
                if (!confirm(`Are you sure you want to permanently delete ${selectedPrompts.size} prompts?`)) return;
                
                const formData = new FormData();
                formData.append('prompt_ids', Array.from(selectedPrompts).join(','));
                
                try {
                    await fetch('/api/prompts/bulk/delete', { method: 'POST', body: formData });
                    toggleBulkMode();
                    fetchPrompts();
                } catch(e) { alert("Bulk delete failed."); }
            }

            async function bulkTag() {
                if (selectedPrompts.size === 0) return alert("Please select at least one prompt.");
                const newTag = prompt(`Enter a new tag to add to ${selectedPrompts.size} prompts:`);
                if (!newTag || !newTag.trim()) return;
                
                const formData = new FormData();
                formData.append('prompt_ids', Array.from(selectedPrompts).join(','));
                formData.append('new_tag', newTag.trim());
                
                try {
                    const res = await fetch('/api/prompts/bulk/tag', { method: 'POST', body: formData });
                    if(!res.ok) throw new Error();
                    toggleBulkMode();
                    fetchPrompts();
                } catch(e) { alert("Bulk tagging failed."); }
            }

            async function bulkAutoTag(btn) {
                if (selectedPrompts.size === 0) return alert("Please select at least one prompt.");
                if (!confirm(`Are you sure you want Gemini to auto-tag ${selectedPrompts.size} prompts? This might take a moment.`)) return;
                
                const originalText = btn.innerText;
                btn.innerText = `⏳ Tagging ${selectedPrompts.size}...`;
                btn.disabled = true;
                
                const savedLang = localStorage.getItem('nanobananaTagLanguage') || 'English';
                const formData = new FormData();
                formData.append('prompt_ids', Array.from(selectedPrompts).join(','));
                formData.append('language', savedLang);
                
                try {
                    const res = await fetch('/api/prompts/bulk/auto-tag', { method: 'POST', body: formData });
                    if (!res.ok) throw new Error();
                    toggleBulkMode();
                    fetchPrompts();
                } catch(e) {
                    alert("Bulk auto-tagging encountered an error. Some tags might not have been added.");
                }
                
                btn.innerText = originalText;
                btn.disabled = false;
            }

            // --- TEMPLATE SYSTEM (LÜCKENTEXT) ---
            function openTemplateModal(promptId, customText = null) {
                let text = customText;
                if (!text) {
                    const p = globalPrompts.find(x => x.id === promptId);
                    if(!p) return;
                    text = p.prompt;
                }

                const matches = [...text.matchAll(/\\[(.*?)\\]/g)];
                const uniquePlaceholders = [...new Set(matches.map(m => m[1]))];

                const container = document.getElementById('templateInputsContainer');
                container.innerHTML = '';
                
                uniquePlaceholders.forEach(ph => {
                    container.innerHTML += `
                        <div>
                            <label class="block text-xs font-bold text-blue-400 mb-1 tracking-wider uppercase">${escapeHTML(ph)}</label>
                            <input type="text" data-placeholder="${escapeHTML(ph)}" oninput="updateTemplatePreview()" 
                                   class="w-full p-2 rounded bg-gray-700 border border-gray-600 text-white placeholder-input focus:outline-none focus:border-blue-400" 
                                   placeholder="Type here...">
                        </div>
                    `;
                });

                document.getElementById('templateOriginalPrompt').value = text;
                document.getElementById('templatePromptId').value = promptId || '';

                updateTemplatePreview();
                document.getElementById('templateModal').classList.remove('hidden');
                
                const firstInput = container.querySelector('input');
                if(firstInput) setTimeout(() => firstInput.focus(), 100);
            }

            function updateTemplatePreview() {
                const originalText = document.getElementById('templateOriginalPrompt').value;
                let finalText = originalText;
                
                const inputs = document.querySelectorAll('.placeholder-input');
                inputs.forEach(input => {
                    const ph = input.getAttribute('data-placeholder');
                    const val = input.value || '[' + ph + ']';
                    const regex = new RegExp('\\\\[' + escapeRegExp(ph) + '\\\\]', 'g');
                    
                    const displayVal = input.value ? `<span class="text-green-400 font-bold">${escapeHTML(val)}</span>` : `<span class="text-blue-400">[${escapeHTML(ph)}]</span>`;
                    finalText = finalText.replace(regex, displayVal);
                });
                
                document.getElementById('templatePreview').innerHTML = finalText.replace(/\\n/g, '<br>');
            }
            
            function copyRawTemplate(btn) {
                const text = document.getElementById('templateOriginalPrompt').value;
                const pid = document.getElementById('templatePromptId').value;
                executeCopyFinal(btn, text, pid);
            }

            function executeTemplateCopy(btn) {
                const originalText = document.getElementById('templateOriginalPrompt').value;
                const promptId = document.getElementById('templatePromptId').value;
                let finalText = originalText;

                const inputs = document.querySelectorAll('.placeholder-input');
                inputs.forEach(input => {
                    const ph = input.getAttribute('data-placeholder');
                    const val = input.value || '[' + ph + ']';
                    const regex = new RegExp('\\\\[' + escapeRegExp(ph) + '\\\\]', 'g');
                    finalText = finalText.replace(regex, val);
                });

                executeCopyFinal(btn, finalText, promptId);
            }
            
            function executeCopyFinal(btn, textToCopy, promptId) {
                navigator.clipboard.writeText(textToCopy);
                
                const originalText = btn.innerHTML;
                btn.innerHTML = '✅ Copied!';
                btn.classList.add('bg-green-600', 'text-white', 'border-green-600');
                
                if (promptId) {
                    try {
                        fetch(`/api/prompts/${promptId}/copy`, { method: 'POST' });
                        const p = globalPrompts.find(x => x.id === promptId);
                        if (p) {
                            p.copy_count = (p.copy_count || 0) + 1;
                            const countSpan = document.getElementById(`count-${promptId}`);
                            if (countSpan) countSpan.innerText = p.copy_count;
                        }
                    } catch(e) {}
                }

                setTimeout(() => {
                    btn.innerHTML = originalText;
                    btn.classList.remove('bg-green-600', 'text-white', 'border-green-600');
                    closeModal('templateModal');
                }, 1000);
            }

            // --- LIGHTBOX CAROUSEL WITH TOUCH ---
            let currentLightboxImages = [];
            let currentLightboxIndex = 0;
            let touchstartX = 0;
            let touchendX = 0;

            const lightboxModal = document.getElementById('lightboxModal');

            lightboxModal.addEventListener('touchstart', e => {
                touchstartX = e.changedTouches[0].screenX;
            }, {passive: true});

            lightboxModal.addEventListener('touchend', e => {
                touchendX = e.changedTouches[0].screenX;
                handleLightboxSwipe();
            }, {passive: true});

            function handleLightboxSwipe() {
                if (currentLightboxImages.length <= 1) return;
                const threshold = 40;
                if (touchendX < touchstartX - threshold) {
                    lightboxNavigate(1);
                }
                if (touchendX > touchstartX + threshold) {
                    lightboxNavigate(-1);
                }
            }

            function openLightbox(promptId, index = 0) {
                const p = globalPrompts.find(x => x.id === promptId);
                if (!p) return;
                currentLightboxImages = parseImages(p.image_path);
                currentLightboxIndex = index;
                
                updateLightboxImage();
                document.getElementById('lightboxModal').classList.remove('hidden');
                
                document.addEventListener('keydown', lightboxKeyHandler);
            }

            function updateLightboxImage() {
                if (currentLightboxImages.length === 0) return;
                document.getElementById('lightboxImg').src = '/images/' + currentLightboxImages[currentLightboxIndex];
                
                const prevBtn = document.getElementById('lightboxPrevBtn');
                const nextBtn = document.getElementById('lightboxNextBtn');
                
                if (currentLightboxImages.length > 1) {
                    prevBtn.classList.remove('hidden');
                    nextBtn.classList.remove('hidden');
                } else {
                    prevBtn.classList.add('hidden');
                    nextBtn.classList.add('hidden');
                }
            }

            function lightboxNavigate(direction, event) {
                if(event) event.stopPropagation();
                if(currentLightboxImages.length <= 1) return;
                
                currentLightboxIndex += direction;
                if (currentLightboxIndex < 0) currentLightboxIndex = currentLightboxImages.length - 1;
                if (currentLightboxIndex >= currentLightboxImages.length) currentLightboxIndex = 0;
                
                updateLightboxImage();
            }

            function lightboxKeyHandler(e) {
                if (e.key === 'ArrowRight') lightboxNavigate(1);
                else if (e.key === 'ArrowLeft') lightboxNavigate(-1);
                else if (e.key === 'Escape') closeLightbox();
            }

            function closeLightbox() {
                document.getElementById('lightboxModal').classList.add('hidden');
                document.getElementById('lightboxImg').src = '';
                document.removeEventListener('keydown', lightboxKeyHandler);
            }

            function toggleFavFilter() {
                showOnlyFavorites = !showOnlyFavorites;
                const btn = document.getElementById('filterFavBtn');
                if(showOnlyFavorites) {
                    btn.classList.replace('text-gray-300', 'text-red-400');
                    btn.innerHTML = '❤️ Favorites';
                } else {
                    btn.classList.replace('text-red-400', 'text-gray-300');
                    btn.innerHTML = '🤍 Favorites';
                }
                triggerRenderReset();
            }

            async function toggleFavorite(id) {
                const res = await fetch(`/api/prompts/${id}/favorite`, {method: 'POST'});
                if(res.ok) {
                    const data = await res.json();
                    const p = globalPrompts.find(x => x.id === id);
                    if(p) p.is_favorite = data.is_favorite;
                    
                    const btn = document.getElementById('fav-btn-' + id);
                    if(btn) {
                        btn.innerText = data.is_favorite ? '❤️' : '🤍';
                        btn.classList.add('scale-125');
                        setTimeout(() => btn.classList.remove('scale-125'), 200);
                    }
                    if(showOnlyFavorites && !data.is_favorite) triggerRenderReset();
                }
            }

            async function importData() {
                const fileInput = document.getElementById('importFile');
                if (!fileInput.files.length) { alert("Please select a file first."); return; }
                
                const btn = document.getElementById('importBtn');
                btn.innerText = 'Importing...'; btn.disabled = true;
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                try {
                    const res = await fetch('/api/import', { method: 'POST', body: formData });
                    if (res.ok) {
                        const data = await res.json();
                        alert(`Successfully imported ${data.added} prompts and ${data.added_history} history entries!`);
                        closeModal('exportModal');
                        fetchPrompts();
                    } else {
                        const err = await res.json();
                        alert('Error: ' + err.detail);
                    }
                } catch(e) {
                    alert('Import failed.');
                }
                
                btn.innerText = 'Upload & Import'; btn.disabled = false;
                fileInput.value = '';
            }

            function extractAutocompleteData() {
                const authorsSet = new Set();
                const tagsSet = new Set();
                globalPrompts.forEach(p => {
                    if (p.author) authorsSet.add(p.author.trim());
                    if (p.tags) {
                        p.tags.split(',').forEach(t => {
                            const cleanTag = t.trim();
                            if(cleanTag) tagsSet.add(cleanTag);
                        });
                    }
                });
                uniqueAuthors = Array.from(authorsSet).sort((a,b) => a.toLowerCase().localeCompare(b.toLowerCase()));
                uniqueTags = Array.from(tagsSet).sort((a,b) => a.toLowerCase().localeCompare(b.toLowerCase()));
            }

            function showAuthorSuggestions(val) {
                const container = document.getElementById('authorSuggestions');
                if(!val) { container.classList.add('hidden'); return; }
                const matches = uniqueAuthors.filter(a => a.toLowerCase().includes(val.toLowerCase()) && a.toLowerCase() !== val.toLowerCase());
                if(matches.length === 0) { container.classList.add('hidden'); return; }
                container.innerHTML = matches.map(m => `
                    <div class="p-2 cursor-pointer hover:bg-yellow-500 hover:text-black transition-colors" onclick="selectAuthor(decodeURIComponent('${encodeForJS(m)}'))">${escapeHTML(m)}</div>
                `).join('');
                container.classList.remove('hidden');
            }

            function selectAuthor(val) {
                document.getElementById('formAuthor').value = val;
                document.getElementById('authorSuggestions').classList.add('hidden');
            }

            function showTagSuggestions(val) {
                const container = document.getElementById('tagSuggestions');
                if(!val) { container.classList.add('hidden'); return; }
                const parts = val.split(',');
                const currentTag = parts[parts.length - 1].trim(); 
                if(!currentTag) { container.classList.add('hidden'); return; }
                const matches = uniqueTags.filter(t => t.toLowerCase().includes(currentTag.toLowerCase()) && t.toLowerCase() !== currentTag.toLowerCase());
                if(matches.length === 0) { container.classList.add('hidden'); return; }
                container.innerHTML = matches.map(m => `
                    <div class="p-2 cursor-pointer hover:bg-yellow-500 hover:text-black transition-colors" onclick="selectTag(decodeURIComponent('${encodeForJS(m)}'))">${escapeHTML(m)}</div>
                `).join('');
                container.classList.remove('hidden');
            }

            function selectTag(selectedTag) {
                const input = document.getElementById('formTags');
                const parts = input.value.split(',');
                parts.pop(); 
                parts.push(' ' + selectedTag);
                input.value = parts.join(',').trim() + ', '; 
                document.getElementById('tagSuggestions').classList.add('hidden');
                input.focus();
            }
            
            function showMergeOldTagSuggestions(val) {
                const container = document.getElementById('mergeOldTagSuggestions');
                if(!val) { container.classList.add('hidden'); return; }
                const matches = uniqueTags.filter(t => t.toLowerCase().includes(val.toLowerCase()) && t.toLowerCase() !== val.toLowerCase());
                if(matches.length === 0) { container.classList.add('hidden'); return; }
                container.innerHTML = matches.map(m => `
                    <div class="p-2 cursor-pointer hover:bg-yellow-500 hover:text-black transition-colors" onclick="selectMergeOldTag(decodeURIComponent('${encodeForJS(m)}'))">${escapeHTML(m)}</div>
                `).join('');
                container.classList.remove('hidden');
            }

            function selectMergeOldTag(val) {
                document.getElementById('mergeOldTag').value = val;
                document.getElementById('mergeOldTagSuggestions').classList.add('hidden');
            }

            function showMergeNewTagSuggestions(val) {
                const container = document.getElementById('mergeNewTagSuggestions');
                if(!val) { container.classList.add('hidden'); return; }
                const matches = uniqueTags.filter(t => t.toLowerCase().includes(val.toLowerCase()) && t.toLowerCase() !== val.toLowerCase());
                if(matches.length === 0) { container.classList.add('hidden'); return; }
                container.innerHTML = matches.map(m => `
                    <div class="p-2 cursor-pointer hover:bg-yellow-500 hover:text-black transition-colors" onclick="selectMergeNewTag(decodeURIComponent('${encodeForJS(m)}'))">${escapeHTML(m)}</div>
                `).join('');
                container.classList.remove('hidden');
            }

            function selectMergeNewTag(val) {
                document.getElementById('mergeNewTag').value = val;
                document.getElementById('mergeNewTagSuggestions').classList.add('hidden');
            }

            document.addEventListener('click', function(e) {
                if(e.target.id !== 'formAuthor') document.getElementById('authorSuggestions')?.classList.add('hidden');
                if(e.target.id !== 'formTags') document.getElementById('tagSuggestions')?.classList.add('hidden');
                if(e.target.id !== 'mergeOldTag') document.getElementById('mergeOldTagSuggestions')?.classList.add('hidden');
                if(e.target.id !== 'mergeNewTag') document.getElementById('mergeNewTagSuggestions')?.classList.add('hidden');
            });

            const dropZone = document.getElementById('dropZone');

            dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('border-yellow-400', 'bg-gray-600'); });
            dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.classList.remove('border-yellow-400', 'bg-gray-600'); });
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault(); dropZone.classList.remove('border-yellow-400', 'bg-gray-600');
                if (e.dataTransfer.files.length) handleFiles(Array.from(e.dataTransfer.files));
            });

            function handleFileSelect(e) {
                if (e.target.files.length) handleFiles(Array.from(e.target.files));
                e.target.value = ''; 
            }

            document.addEventListener('paste', (e) => {
                const promptModal = document.getElementById('promptModal');
                if (promptModal.classList.contains('hidden')) return;
                if (e.clipboardData && e.clipboardData.files && e.clipboardData.files.length > 0) {
                    e.preventDefault();
                    handleFiles(Array.from(e.clipboardData.files));
                }
            });

            dropZone.addEventListener('contextmenu', async (e) => {
                e.preventDefault(); e.stopPropagation();
                try {
                    const clipboardItems = await navigator.clipboard.read();
                    for (const item of clipboardItems) {
                        const imageTypes = item.types.filter(type => type.startsWith('image/'));
                        if (imageTypes.length > 0) {
                            const blob = await item.getType(imageTypes[0]);
                            const file = new File([blob], "pasted-image.png", { type: blob.type });
                            handleFiles([file]);
                            return; 
                        }
                    }
                    alert("No image found in clipboard.");
                } catch (err) {
                    alert("Clipboard access denied. Please use Ctrl+V / Cmd+V instead.");
                }
            });

            function handleFiles(files) {
                const imgFiles = files.filter(f => f.type.startsWith('image/'));
                if (imgFiles.length === 0) return;
                extractMetadata(imgFiles[0]); 
                imgFiles.forEach(file => {
                    const isFirstEver = mediaItems.length === 0;
                    const url = URL.createObjectURL(file);
                    mediaItems.push({ type: 'new', val: file, url: url, isCover: isFirstEver });
                });
                renderMediaPreviews();
            }

            function renderMediaPreviews() {
                const container = document.getElementById('mediaPreviews');
                if (mediaItems.length === 0) { container.innerHTML = ''; return; }
                container.innerHTML = mediaItems.map((item, index) => `
                    <div class="relative w-24 h-24 flex-shrink-0 rounded overflow-hidden border-2 transition-colors ${item.isCover ? 'border-yellow-400' : 'border-gray-600'}">
                        <img src="${item.url}" class="w-full h-full object-cover">
                        <div class="absolute top-0 right-0 bg-black/80 flex rounded-bl">
                            <button type="button" onclick="event.stopPropagation(); setCover(${index})" class="px-2 py-1 hover:text-yellow-400 text-xs transition-colors ${item.isCover ? 'text-yellow-400' : 'text-gray-400'}">⭐</button>
                            <button type="button" onclick="event.stopPropagation(); removeMedia(${index})" class="px-2 py-1 hover:text-red-500 text-gray-400 text-xs transition-colors">✕</button>
                        </div>
                        ${item.isCover ? '<div class="absolute bottom-0 left-0 right-0 bg-yellow-400 text-black text-[10px] text-center font-bold">COVER</div>' : ''}
                    </div>
                `).join('');
            }

            function setCover(index) {
                mediaItems.forEach(m => m.isCover = false);
                mediaItems[index].isCover = true;
                const coverItem = mediaItems.splice(index, 1)[0];
                mediaItems.unshift(coverItem);
                renderMediaPreviews();
            }

            function removeMedia(index) {
                const removed = mediaItems.splice(index, 1)[0];
                if (removed.type === 'new') URL.revokeObjectURL(removed.url); 
                if (removed.isCover && mediaItems.length > 0) mediaItems[0].isCover = true;
                renderMediaPreviews();
            }

            async function forceExtractPrompt(btn) {
                const promptField = document.getElementById('formPrompt');
                if (mediaItems.length === 0) { alert("Please add an image first."); return; }
                if (promptField.value.trim() !== '') {
                    if (!confirm("This will overwrite your current prompt text. Continue?")) return;
                }

                const targetMedia = mediaItems.find(m => m.isCover) || mediaItems[0];
                const formData = new FormData();
                if (targetMedia.type === 'new') formData.append('image', targetMedia.val);
                else formData.append('existing_image', targetMedia.val);

                const originalText = btn.innerText;
                btn.innerText = '⏳ Extracting...'; btn.disabled = true;

                try {
                    const res = await fetch('/api/extract-metadata', { method: 'POST', body: formData });
                    if (res.ok) {
                        const data = await res.json();
                        if (data.extracted_prompt) {
                            promptField.value = data.extracted_prompt;
                            promptField.classList.remove('flash-success');
                            void promptField.offsetWidth; 
                            promptField.classList.add('flash-success');
                        } else { alert("No metadata/prompt found in this image."); }
                    } else { alert("Failed to extract metadata."); }
                } catch (e) { alert("Error extracting metadata."); }
                
                btn.innerText = originalText; btn.disabled = false;
            }

            async function extractMetadata(file) {
                const promptField = document.getElementById('formPrompt');
                if (promptField.value.trim() !== '') return;
                const formData = new FormData();
                formData.append('image', file);
                try {
                    const res = await fetch('/api/extract-metadata', { method: 'POST', body: formData });
                    if(res.ok) {
                        const data = await res.json();
                        if (data.extracted_prompt && promptField.value.trim() === '') {
                            promptField.value = data.extracted_prompt;
                            promptField.classList.remove('flash-success');
                            void promptField.offsetWidth; 
                            promptField.classList.add('flash-success');
                        }
                    }
                } catch (e) {}
            }

            async function autoGenerateTitle(btn) {
                const promptField = document.getElementById('formPrompt');
                const titleField = document.getElementById('formTitle');
                const langField = document.getElementById('tagLanguage');
                
                if (!promptField.value.trim()) { 
                    alert("Please enter or extract a prompt text first so Gemini knows what to title!"); 
                    return; 
                }
                
                const originalText = btn.innerText;
                btn.innerText = '⏳ Generating...'; 
                btn.disabled = true;
                
                const formData = new FormData();
                formData.append('prompt', promptField.value);
                formData.append('language', langField.value);
                
                try {
                    const res = await fetch('/api/title/auto', { method: 'POST', body: formData });
                    if (res.ok) {
                        const data = await res.json();
                        if (data.title) {
                            titleField.value = data.title;
                            
                            titleField.classList.remove('flash-success');
                            void titleField.offsetWidth; 
                            titleField.classList.add('flash-success');
                        }
                    } else {
                        const err = await res.json();
                        alert("Error generating title: " + err.detail);
                    }
                } catch (e) { 
                    alert("Failed to connect to Auto-Title API."); 
                }
                
                btn.innerText = originalText; 
                btn.disabled = false;
            }

            async function autoGenerateTags(btn) {
                const promptField = document.getElementById('formPrompt');
                const tagsField = document.getElementById('formTags');
                const langField = document.getElementById('tagLanguage');
                
                if (!promptField.value.trim()) { 
                    alert("Please enter or extract a prompt text first so Gemini knows what to tag!"); 
                    return; 
                }
                
                const originalText = btn.innerText;
                btn.innerText = '⏳ Generating...'; 
                btn.disabled = true;
                
                const formData = new FormData();
                formData.append('prompt', promptField.value);
                formData.append('language', langField.value);
                
                try {
                    const res = await fetch('/api/tags/auto', { method: 'POST', body: formData });
                    if (res.ok) {
                        const data = await res.json();
                        if (data.tags) {
                            let currentTags = tagsField.value.split(',').map(t => t.trim()).filter(t => t);
                            let newTags = data.tags.split(',').map(t => t.trim()).filter(t => t);
                            let combined = [...new Set([...currentTags, ...newTags])];
                            
                            tagsField.value = combined.join(', ') + (combined.length > 0 ? ', ' : '');
                            
                            tagsField.classList.remove('flash-success');
                            void tagsField.offsetWidth; 
                            tagsField.classList.add('flash-success');
                        }
                    } else {
                        const err = await res.json();
                        alert("Error generating tags: " + err.detail);
                    }
                } catch (e) { 
                    alert("Failed to connect to Auto-Tag API."); 
                }
                
                btn.innerText = originalText; 
                btn.disabled = false;
            }

            // ── Comments ──────────────────────────────────────────────────────
            let currentCommentPromptId = null;

            async function openCommentsModal(promptId) {
                currentCommentPromptId = promptId;
                document.getElementById('newCommentBody').value = '';
                document.getElementById('commentsModal').classList.remove('hidden');
                await loadComments(promptId);
            }

            async function loadComments(promptId) {
                const container = document.getElementById('commentsList');
                container.innerHTML = '<p class="text-xs text-gray-500 italic text-center py-4">Loading…</p>';
                const res = await fetch(`/api/prompts/${promptId}/comments`);
                if (!res.ok) {
                    container.innerHTML = '<p class="text-xs text-red-400 text-center py-4">Failed to load comments.</p>';
                    return;
                }
                const comments = await res.json();
                if (comments.length === 0) {
                    container.innerHTML = '<p class="text-xs text-gray-500 italic text-center py-4">No comments yet. Be the first!</p>';
                    return;
                }
                container.innerHTML = comments.map(c => renderComment(c, false)).join('');
            }

            function renderComment(c, isReply) {
                const safeBody   = escapeHTML(c.body);
                const safeAuthor = escapeHTML(c.author_name || c.author_email);
                const date       = new Date(c.created_at + 'Z').toLocaleString();
                const dvClass    = c.user_downvoted
                    ? 'text-red-400'
                    : 'text-gray-500 hover:text-red-400';
                const deleteBtn  = (c.is_mine || IS_ADMIN)
                    ? `<button onclick="deleteComment('${c.id}')" class="text-xs text-gray-600 hover:text-red-400 transition-colors" title="Delete">🗑</button>`
                    : '';
                const replyBtn   = isReply ? '' :
                    `<button onclick="toggleReplyForm('${c.id}')" class="text-xs text-gray-500 hover:text-yellow-400 transition-colors">↩ Reply</button>`;
                const replyForm  = isReply ? '' : `
                    <div id="reply-form-${c.id}" class="hidden mt-2">
                        <textarea id="reply-body-${c.id}" placeholder="Write a reply…" rows="2"
                            class="w-full bg-gray-900 border border-gray-600 rounded p-2 text-xs text-white resize-none focus:outline-none focus:border-yellow-400 mb-1"></textarea>
                        <div class="flex gap-2 justify-end">
                            <button onclick="toggleReplyForm('${c.id}')" class="text-xs text-gray-400 hover:text-white px-2 py-1 rounded">Cancel</button>
                            <button onclick="submitReply('${c.id}')" class="text-xs bg-yellow-500 hover:bg-yellow-600 text-black font-bold px-3 py-1 rounded transition-colors">Reply</button>
                        </div>
                    </div>`;
                const repliesHtml = (c.replies && c.replies.length > 0)
                    ? `<div class="ml-4 mt-2 flex flex-col gap-2 border-l-2 border-gray-700 pl-3">
                           ${c.replies.map(r => renderComment(r, true)).join('')}
                       </div>`
                    : '';
                const wrapClass = isReply ? '' : 'bg-gray-900 rounded-lg p-3';
                return `
                <div class="${wrapClass}">
                    <div class="flex items-start justify-between gap-2">
                        <div class="flex-grow min-w-0">
                            <span class="text-xs font-bold text-yellow-400">${safeAuthor}</span>
                            <span class="text-xs text-gray-500 ml-2">${date}</span>
                            <p class="text-sm text-gray-200 mt-1 break-words">${safeBody}</p>
                        </div>
                        <div class="flex items-center gap-1 flex-shrink-0">
                            <button onclick="downvoteComment('${c.id}', this)"
                                class="text-xs ${dvClass} transition-colors flex items-center gap-0.5"
                                title="Downvote" id="dv-btn-${c.id}">
                                👎 <span id="dv-count-${c.id}">${c.downvote_count || 0}</span>
                            </button>
                            ${deleteBtn}
                        </div>
                    </div>
                    <div class="flex gap-3 mt-1">${replyBtn}</div>
                    ${replyForm}
                    ${repliesHtml}
                </div>`;
            }

            function toggleReplyForm(commentId) {
                const form = document.getElementById('reply-form-' + commentId);
                if (form) form.classList.toggle('hidden');
            }

            async function submitComment() {
                const body = document.getElementById('newCommentBody').value.trim();
                if (!body) return;
                const formData = new FormData();
                formData.append('body', body);
                const res = await fetch(`/api/prompts/${currentCommentPromptId}/comments`,
                    { method: 'POST', body: formData });
                if (res.ok) {
                    document.getElementById('newCommentBody').value = '';
                    await loadComments(currentCommentPromptId);
                } else {
                    const err = await res.json().catch(() => ({}));
                    alert('Failed to post comment: ' + (err.detail || res.status));
                }
            }

            async function submitReply(parentId) {
                const body = document.getElementById('reply-body-' + parentId).value.trim();
                if (!body) return;
                const formData = new FormData();
                formData.append('body', body);
                formData.append('parent_id', parentId);
                const res = await fetch(`/api/prompts/${currentCommentPromptId}/comments`,
                    { method: 'POST', body: formData });
                if (res.ok) {
                    await loadComments(currentCommentPromptId);
                } else {
                    const err = await res.json().catch(() => ({}));
                    alert('Failed to post reply: ' + (err.detail || res.status));
                }
            }

            async function downvoteComment(commentId, btn) {
                const res = await fetch(`/api/comments/${commentId}/downvote`, { method: 'POST' });
                if (!res.ok) return;
                const data = await res.json();
                const countEl = document.getElementById('dv-count-' + commentId);
                if (countEl) {
                    const cur = parseInt(countEl.textContent) || 0;
                    countEl.textContent = data.downvoted ? cur + 1 : Math.max(0, cur - 1);
                }
                btn.className = btn.className
                    .replace('text-red-400', '')
                    .replace('text-gray-500 hover:text-red-400', '')
                    .trim()
                    + ' ' + (data.downvoted ? 'text-red-400' : 'text-gray-500 hover:text-red-400');
            }

            async function deleteComment(commentId) {
                if (!confirm('Delete this comment and all its replies?')) return;
                const res = await fetch(`/api/comments/${commentId}`, { method: 'DELETE' });
                if (res.ok) {
                    await loadComments(currentCommentPromptId);
                } else {
                    alert('Failed to delete comment.');
                }
            }

            function updateCarouselCounter(id, total) {
                const container = document.getElementById('carousel-' + id);
                const counter = document.getElementById('carousel-counter-' + id);
                if (container && counter) {
                    const index = Math.round(container.scrollLeft / container.clientWidth) + 1;
                    counter.innerText = `${index} / ${total} 📸`;
                }
            }

            function scrollCarousel(id, direction) {
                const container = document.getElementById('carousel-' + id);
                if (container) {
                    const scrollAmount = container.offsetWidth;
                    container.scrollBy({ left: direction * scrollAmount, behavior: 'smooth' });
                }
            }

            function triggerSearch(term) {
                if(term && (term.startsWith('tag:') || term.startsWith('author:'))) {
                    const current = document.getElementById('searchInput').value.trim();
                    if(!current.includes(term)) {
                        document.getElementById('searchInput').value = current ? current + ' ' + term : term;
                    }
                } else {
                    document.getElementById('searchInput').value = term;
                }
                triggerRenderReset();
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }

            function clearSearch() {
                document.getElementById('searchInput').value = '';
                triggerRenderReset();
            }

            function resetToHome() {
                document.getElementById('searchInput').value = '';
                if (showOnlyFavorites) toggleFavFilter();
                if (activeCollectionId) clearCollectionFilter();
                window.scrollTo({ top: 0, behavior: 'smooth' });
                
                window.history.replaceState({}, document.title, window.location.pathname);
                triggerRenderReset();
            }

            function applySort() {
                localStorage.setItem('nanobananaSort', document.getElementById('sortSelect').value);
                triggerRenderReset();
            }

            function setLayout(cols) {
                localStorage.setItem('nanobananaLayoutCols', cols);
                const gallery = document.getElementById('gallery');
                
                gallery.className = 'grid gap-4 md:gap-6 relative z-10';
                
                if(cols === 1) gallery.classList.add('grid-cols-1');
                if(cols === 2) gallery.classList.add('grid-cols-2');
                if(cols === 3) gallery.classList.add('grid-cols-3');
                if(cols === 4) gallery.classList.add('grid-cols-4');
                if(cols === 5) gallery.classList.add('grid-cols-5');
                if(cols === 6) gallery.classList.add('grid-cols-6');
                if(cols === 7) gallery.classList.add('grid-cols-7');
                if(cols === 8) gallery.classList.add('grid-cols-8');

                const layoutSelect = document.getElementById('layoutSelect');
                if(layoutSelect) layoutSelect.value = cols;
            }

            async function fetchPrompts() {
                const [promptsRes] = await Promise.all([
                    fetch('/api/prompts'),
                    fetchCollections()
                ]);
                if (promptsRes.status === 401) window.location.reload();
                globalPrompts = await promptsRes.json();

                extractAutocompleteData();
                triggerRenderReset();
                
                const urlParams = new URLSearchParams(window.location.search);
                const commentPromptId = urlParams.get('open_comments');
                if (commentPromptId && globalPrompts.find(p => p.id === commentPromptId)) {
                    openCommentsModal(commentPromptId);
                    window.history.replaceState({}, document.title, window.location.pathname);
                }
            }

            function triggerRenderReset() {
                const search = document.getElementById('searchInput').value.trim();
                const sortVal = localStorage.getItem('nanobananaSort') || 'newest';
                document.getElementById('sortSelect').value = sortVal;

                const terms = search.split(/\s+/).filter(t => t.length > 0);

                displayPrompts = globalPrompts.filter(p => {
                    if(showOnlyFavorites && !p.is_favorite) return false;
                    if(activeCollectionId && !(p.collection_ids || []).includes(activeCollectionId)) return false;
                    if(terms.length === 0) return true;

                    let matchesAll = true;
                    for(let term of terms) {
                        const isNegation = term.startsWith('-');
                        const actualTerm = isNegation ? term.substring(1).toLowerCase() : term.toLowerCase();
                        let termMatch = false;

                        if (actualTerm.startsWith('tag:')) {
                            const tagVal = actualTerm.substring(4);
                            const hasTag = p.tags.toLowerCase().includes(tagVal);
                            termMatch = isNegation ? !hasTag : hasTag;
                        } else if (actualTerm.startsWith('author:')) {
                            const authVal = actualTerm.substring(7);
                            const hasAuth = p.author.toLowerCase().includes(authVal);
                            termMatch = isNegation ? !hasAuth : hasAuth;
                        } else if (actualTerm === 'is:favorite') {
                            termMatch = isNegation ? !p.is_favorite : p.is_favorite;
                        } else if (actualTerm === 'is:mine') {
                            termMatch = isNegation ? !p.is_mine : p.is_mine;
                        } else if (actualTerm === 'is:shared') {
                            termMatch = isNegation ? !p.is_shared : p.is_shared;
                        } else {
                            const textToSearch = (p.title + " " + p.prompt + " " + p.tags + " " + p.author).toLowerCase();
                            const hasText = textToSearch.includes(actualTerm) || p.id === actualTerm;
                            termMatch = isNegation ? !hasText : hasText;
                        }

                        if (!termMatch) {
                            matchesAll = false;
                            break;
                        }
                    }
                    return matchesAll;
                });

                if (sortVal === 'oldest') {
                    displayPrompts.reverse(); 
                } else if (sortVal === 'most_copied') {
                    displayPrompts.sort((a, b) => (b.copy_count || 0) - (a.copy_count || 0));
                } else if (sortVal === 'least_copied') {
                    displayPrompts.sort((a, b) => (a.copy_count || 0) - (b.copy_count || 0));
                }

                const activeCol = activeCollectionId ? collections.find(c => c.id === activeCollectionId) : null;
                const counterLabel = activeCol ? ` Prompts in 📁 ${activeCol.name}` : ' Prompts';
                document.getElementById('promptCounter').innerText = displayPrompts.length + counterLabel;

                document.getElementById('gallery').innerHTML = '';
                renderIndex = 0;
                renderNextBatch();
                
                if (isBulkMode) updateSelectAllButtonState();
            }

            function parseImages(pathRaw) {
                try {
                    const arr = JSON.parse(pathRaw);
                    if (Array.isArray(arr)) return arr;
                    return [pathRaw];
                } catch(e) {
                    return [pathRaw];
                }
            }

            function renderNextBatch() {
                const gallery = document.getElementById('gallery');
                const batch = displayPrompts.slice(renderIndex, renderIndex + BATCH_SIZE);
                
                if (batch.length === 0) {
                    document.getElementById('loadingIndicator').classList.add('hidden');
                    return;
                }

                let htmlChunk = '';

                batch.forEach(p => {
                    const safeTitle = escapeHTML(p.title);
                    const safeAuthor = escapeHTML(p.author);
                    const safePrompt = escapeHTML(p.prompt);

                    const tagsHtml = p.tags.split(',').map(t => {
                        const rawTag = t.trim();
                        if(!rawTag) return '';
                        const safeTag = escapeHTML(rawTag);
                        return `<button onclick="triggerSearch(decodeURIComponent('tag:${encodeForJS(rawTag)}'))" class="bg-gray-700 hover:bg-yellow-500 hover:text-black transition-colors text-xs px-2 py-1 rounded truncate max-w-full cursor-pointer focus:outline-none">${safeTag}</button>`;
                    }).join(' ');
                    
                    let badge = p.is_mine
                        ? (p.is_shared ? '<span class="bg-blue-600 text-white text-xs px-2 py-1 rounded shadow">My Shared Prompt</span>'
                                       : '<span class="bg-gray-900 text-white text-xs px-2 py-1 rounded shadow border border-gray-700">Private</span>')
                        : '<span class="bg-green-600 text-white text-xs px-2 py-1 rounded shadow truncate max-w-full inline-block">Shared by ' + escapeHTML(p.user_email) + '</span>';

                    const deleteBtn = (p.is_mine || IS_ADMIN) ? `<button onclick="deletePrompt('${p.id}')" class="flex-1 bg-red-900 hover:bg-red-800 text-red-200 py-1 px-2 rounded text-sm transition-colors border border-red-800">Delete</button>` : '';

                    let actionButtons = `
                        <div class="flex flex-wrap gap-2 mt-2">
                            <button onclick="forkPrompt('${p.id}')" class="flex-1 bg-purple-900 hover:bg-purple-800 text-purple-200 py-1 px-2 rounded text-sm transition-colors border border-purple-800" title="Copy as new prompt">🍴 Fork</button>
                            <button onclick="openEditModal('${p.id}')" class="flex-1 bg-blue-900 hover:bg-blue-800 text-blue-200 py-1 px-2 rounded text-sm transition-colors border border-blue-800">Edit</button>
                            <button onclick="openHistoryModal('${p.id}')" class="flex-1 bg-gray-700 hover:bg-gray-600 text-gray-200 py-1 px-2 rounded text-sm transition-colors border border-gray-600">History</button>
                            <button onclick="openCollectionAssignModal('${p.id}')" class="flex-1 bg-teal-900 hover:bg-teal-800 text-teal-200 py-1 px-2 rounded text-sm transition-colors border border-teal-800" title="Add to Collection">📁 Collect</button>
                            <button onclick="openShareModal('${p.id}')" class="flex-1 bg-yellow-800 hover:bg-yellow-700 text-yellow-200 py-1 px-2 rounded text-sm transition-colors border border-yellow-700" title="Share link">🔗 Share</button>
                            <button onclick="openCommentsModal('${p.id}')" class="flex-1 bg-gray-700 hover:bg-gray-600 text-gray-200 py-1 px-2 rounded text-sm transition-colors border border-gray-600" title="Comments">💬 Comments</button>
                            ${deleteBtn}
                        </div>
                    `;

                    const tooltipPrompt = safePrompt.replace(/"/g, '&quot;');
                    const copyCount = p.copy_count || 0;

                    const images = parseImages(p.image_path);
                    const isMulti = images.length > 1;

                    let carouselHtml = `<div class="flex overflow-x-auto snap-x snap-mandatory hide-scrollbar h-full w-full" id="carousel-${p.id}" onscroll="updateCarouselCounter('${p.id}', ${images.length})">`;
                    images.forEach((img, idx) => {
                        carouselHtml += <img src="/images/${img}" onclick="if(!isBulkMode) openLightbox('${p.id}', ${idx})" loading="lazy" class="snap-center min-w-full h-full w-full object-cover object-center bg-gray-900 cursor-pointer" title="${tooltipPrompt}">;
                    });
                    carouselHtml += `</div>`;

                    if (isMulti) {
                        carouselHtml += `
                            <button onclick="event.stopPropagation(); scrollCarousel('${p.id}', -1)" class="absolute left-2 top-1/2 -translate-y-1/2 bg-black/60 hover:bg-black text-white w-8 h-8 rounded-full flex items-center justify-center opacity-0 group-hover/carousel:opacity-100 transition-opacity z-10 focus:outline-none">◀</button>
                            <button onclick="event.stopPropagation(); scrollCarousel('${p.id}', 1)" class="absolute right-2 top-1/2 -translate-y-1/2 bg-black/60 hover:bg-black text-white w-8 h-8 rounded-full flex items-center justify-center opacity-0 group-hover/carousel:opacity-100 transition-opacity z-10 focus:outline-none">▶</button>
                            <div id="carousel-counter-${p.id}" class="absolute bottom-2 right-2 bg-black/70 text-[10px] px-2 py-0.5 rounded text-white z-10 pointer-events-none transition-opacity">1 / ${images.length} 📸</div>
                        `;
                    }

                    const firstImg = images.length > 0 ? images[0] : '';
                    carouselHtml += `<button onclick="event.stopPropagation(); openLightbox('${p.id}', 0)" class="absolute bottom-2 left-2 bg-black/60 hover:bg-black text-white text-xs px-2 py-1 rounded opacity-0 group-hover/carousel:opacity-100 transition-opacity z-10 focus:outline-none" title="Open fullscreen">⤢</button>`;
                    
                    const favIcon = p.is_favorite ? '❤️' : '🤍';
                    
                    const hasPlaceholders = /\\[(.*?)\\]/.test(p.prompt);
                    let mainCopyBtn = '';
                    if(hasPlaceholders) {
                        mainCopyBtn = `
                            <button onclick="openTemplateModal('${p.id}')" 
                                    title="Fill in variables"
                                    class="w-full bg-blue-700 hover:bg-blue-600 py-2 rounded text-sm sm:text-base font-bold transition-colors cursor-pointer mt-auto border border-blue-600 flex items-center justify-center gap-2">
                                🧩 Fill & Copy
                            </button>
                        `;
                    } else {
                        mainCopyBtn = `
                            <button onclick="copyToClipboard(this, decodeURIComponent('${encodeForJS(p.prompt)}'), '${p.id}')" 
                                    title="${tooltipPrompt}"
                                    class="w-full bg-gray-700 hover:bg-gray-600 py-2 rounded text-sm sm:text-base font-bold transition-colors cursor-pointer mt-auto">
                                📋 Copy Prompt
                            </button>
                        `;
                    }

                    const bulkCheckboxHtml = `
                        <div class="absolute top-2 left-2 z-20 bulk-checkbox-container" style="display: ${isBulkMode ? 'block' : 'none'};">
                            <input type="checkbox" class="w-6 h-6 rounded border-gray-600 cursor-pointer shadow-lg outline-none ring-2 ring-purple-500 ring-offset-2 ring-offset-gray-800" 
                                   onchange="toggleBulkSelection('${p.id}', this.checked)" 
                                   ${selectedPrompts.has(p.id) ? 'checked' : ''}>
                        </div>
                    `;

                    let forkBadge = '';
                    if (p.forked_from) {
                        const parentTitle = p.forked_from_title ? escapeHTML(p.forked_from_title) : 'Deleted Prompt';
                        forkBadge = `
                            <div class="text-xs text-purple-400 mb-2 flex items-center gap-1 font-medium cursor-pointer hover:text-purple-300" onclick="triggerSearch('${p.forked_from}')" title="Show Original Parent">
                                🍴 Forked from: <span class="truncate max-w-[200px] border-b border-purple-800 border-dashed">${parentTitle}</span>
                            </div>`;
                    }

                    htmlChunk += `
                        <div class="bg-gray-800 rounded-lg overflow-hidden border ${selectedPrompts.has(p.id) ? 'border-purple-500' : 'border-gray-700'} shadow-lg flex flex-col relative group transition-colors" id="card-${p.id}">
                            ${bulkCheckboxHtml}

                            <div class="relative w-full aspect-square overflow-hidden group/carousel" onclick="if(isBulkMode){ const cb = this.previousElementSibling.querySelector('input'); cb.checked = !cb.checked; toggleBulkSelection('${p.id}', cb.checked); } else { toggleCardDetails('${p.id}'); }">
                                ${carouselHtml}
                            </div>

                            <div class="p-3 sm:p-4 flex flex-col flex-grow">
                                ${forkBadge}
                                <h3 class="text-lg sm:text-xl font-bold truncate mb-2">${safeTitle}</h3>

                                <div id="card-details-${p.id}" class="hidden">
                                    <div class="flex items-center justify-between mb-2 gap-2">
                                        ${badge}
                                        <div class="flex items-center gap-2 flex-shrink-0 ml-auto">
                                            <button onclick="toggleFavorite('${p.id}')" id="fav-btn-${p.id}" class="text-xl hover:scale-110 transition-transform focus:outline-none" title="Toggle Favorite">${favIcon}</button>
                                            <div class="bg-gray-900 border border-gray-700 rounded px-2 py-0.5 text-xs text-gray-400 font-bold" title="Times copied">
                                                📋 <span id="count-${p.id}">${copyCount}</span>
                                            </div>
                                        </div>
                                    </div>
                                    <p class="text-xs sm:text-sm text-gray-400 mb-2 truncate">
                                        Author: <button onclick="triggerSearch(decodeURIComponent('author:${encodeForJS(p.author)}'))" class="hover:text-yellow-400 underline decoration-gray-600 hover:decoration-yellow-400 transition-colors cursor-pointer focus:outline-none">${safeAuthor}</button>
                                    </p>
                                    <div class="flex flex-wrap gap-1 mb-3 content-start">${tagsHtml}</div>
                                    <div class="collection-badge-wrapper mb-2">${renderCollectionBadgesHtml(p)}</div>
                                    ${actionButtons}
                                </div>

                                ${mainCopyBtn}
                            </div>
                        </div>
                    `;
                });

                gallery.insertAdjacentHTML('beforeend', htmlChunk);
                renderIndex += BATCH_SIZE;

                if (renderIndex < displayPrompts.length) {
                    document.getElementById('loadingIndicator').classList.remove('hidden');
                } else {
                    document.getElementById('loadingIndicator').classList.add('hidden');
                }
            }

            window.addEventListener('scroll', () => {
                if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 500) {
                    if (renderIndex < displayPrompts.length) renderNextBatch();
                }
            });

            function openAddModal() {
                document.getElementById('modalTitle').innerText = 'Add New Prompt';
                document.getElementById('editPromptId').value = '';
                document.getElementById('formForkedFrom').value = ''; 
                document.getElementById('promptForm').reset();
                document.getElementById('is_shared').checked = true;
                
                const savedLang = localStorage.getItem('nanobananaTagLanguage');
                if(savedLang) document.getElementById('tagLanguage').value = savedLang;
                
                mediaItems = [];
                renderMediaPreviews();
                
                document.getElementById('promptModal').classList.remove('hidden');
            }

            function openEditModal(id) {
                const p = globalPrompts.find(x => x.id === id);
                if(!p) return;
                
                document.getElementById('modalTitle').innerText = 'Edit Prompt (Wiki Mode)';
                document.getElementById('editPromptId').value = p.id;
                document.getElementById('formForkedFrom').value = ''; 
                document.getElementById('formTitle').value = p.title;
                document.getElementById('formAuthor').value = p.author;
                document.getElementById('formTags').value = p.tags;
                document.getElementById('formPrompt').value = p.prompt;
                document.getElementById('is_shared').checked = (p.is_shared === 1);
                document.getElementById('is_shared').disabled = !p.is_mine;
                
                const savedLang = localStorage.getItem('nanobananaTagLanguage');
                if(savedLang) document.getElementById('tagLanguage').value = savedLang;
                
                const imgs = parseImages(p.image_path);
                mediaItems = imgs.map((img, idx) => ({
                    type: 'existing', val: img, url: '/images/' + img, isCover: idx === 0 
                }));
                renderMediaPreviews();

                document.getElementById('promptModal').classList.remove('hidden');
            }

            function toggleCardDetails(id) {
                const details = document.getElementById('card-details-' + id);
                if (details) details.classList.toggle('hidden');
            }

            let currentSharePromptId = null;

            async function openShareModal(promptId) {
                currentSharePromptId = promptId;
                document.getElementById('shareLinkResult').classList.add('hidden');
                document.getElementById('shareLinkModal').classList.remove('hidden');
                await loadShareLinks(promptId);
            }

            async function loadShareLinks(promptId) {
                const res = await fetch(`/api/prompts/${promptId}/share-links`);
                const container = document.getElementById('shareLinkList');
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    container.innerHTML = `<p class="text-xs text-red-400 italic">Error loading links: ${err.detail || res.status}</p>`;
                    return;
                }
                const links = await res.json();
                if (links.length === 0) {
                    container.innerHTML = '<p class="text-xs text-gray-500 italic">No active links yet.</p>';
                    return;
                }
                const now = new Date();
                container.innerHTML = links.map(l => {
                    const exp = new Date(l.expires_at + 'Z');
                    const expired = exp < now;
                    const expLabel = expired ? '✗ expired' : 'until ' + exp.toLocaleString();
                    const url = window.location.origin + '/share/' + l.token;
                    return `<div class="flex items-center gap-2 bg-gray-900 rounded px-3 py-2 text-xs">
                        <span class="truncate font-mono ${expired ? 'text-red-400 line-through' : 'text-gray-300'} flex-grow">${url}</span>
                        <span class="text-gray-500 flex-shrink-0 whitespace-nowrap">${expLabel}</span>
                        <button onclick="revokeShareLink('${l.token}')" class="text-red-400 hover:text-red-300 flex-shrink-0 font-bold" title="Revoke">✕</button>
                    </div>`;
                }).join('');
            }

            async function createShareLink() {
                if (!currentSharePromptId) return;
                const hours = parseInt(document.getElementById('shareLinkExpiry').value);
                const res = await fetch(`/api/prompts/${currentSharePromptId}/share-link`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({expires_in_hours: hours})
                });
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    alert('Failed to create link: ' + (err.detail || res.status));
                    return;
                }
                const data = await res.json();
                const url = window.location.origin + '/share/' + data.token;
                document.getElementById('shareLinkUrl').value = url;
                document.getElementById('shareLinkResult').classList.remove('hidden');
                await loadShareLinks(currentSharePromptId);
            }

            async function revokeShareLink(token) {
                await fetch(`/api/share-links/${token}`, { method: 'DELETE' });
                document.getElementById('shareLinkResult').classList.add('hidden');
                await loadShareLinks(currentSharePromptId);
            }

            function copyShareLink() {
                const url = document.getElementById('shareLinkUrl').value;
                const btn = document.getElementById('shareLinkCopyBtn');
                navigator.clipboard.writeText(url).then(() => {
                    btn.textContent = '✓';
                    setTimeout(() => btn.textContent = '📋', 1500);
                });
            }

            function forkPrompt(id) {
                const p = globalPrompts.find(x => x.id === id);
                if(!p) return;
                
                document.getElementById('modalTitle').innerText = 'Fork Prompt';
                document.getElementById('editPromptId').value = ''; 
                document.getElementById('formForkedFrom').value = p.id; 
                
                document.getElementById('formTitle').value = "Fork of " + p.title;
                document.getElementById('formAuthor').value = p.author;
                document.getElementById('formTags').value = p.tags;
                document.getElementById('formPrompt').value = p.prompt;
                document.getElementById('is_shared').checked = false; 
                document.getElementById('is_shared').disabled = false;
                
                const savedLang = localStorage.getItem('nanobananaTagLanguage');
                if(savedLang) document.getElementById('tagLanguage').value = savedLang;
                
                const imgs = parseImages(p.image_path);
                mediaItems = imgs.map((img, idx) => ({
                    type: 'existing', val: img, url: '/images/' + img, isCover: idx === 0 
                }));
                renderMediaPreviews();

                document.getElementById('promptModal').classList.remove('hidden');
            }
            
            async function rollbackToVersion(promptId, historyId) {
                if (!confirm('Rollback to this version? The current version will be saved to history first.')) return;
                const res = await fetch(`/api/prompts/${promptId}/rollback/${historyId}`, { method: 'POST' });
                if (res.ok) {
                    closeModal('historyModal');
                    await fetchPrompts();
                } else {
                    const err = await res.json().catch(() => ({}));
                    alert('Rollback failed: ' + (err.detail || res.status));
                }
            }

            async function openHistoryModal(id) {
                const res = await fetch(`/api/prompts/${id}/history`);
                const history = await res.json();
                const container = document.getElementById('historyContainer');
                container.innerHTML = '';

                if (history.length === 0) {
                    container.innerHTML = '<p class="text-gray-400 italic text-center py-8">No previous versions exist for this prompt yet.</p>';
                } else {
                    history.forEach((h, index) => {
                        const safeTitle = escapeHTML(h.title);
                        const safeAuthor = escapeHTML(h.author);
                        const safeTags = escapeHTML(h.tags);
                        const safePrompt = escapeHTML(h.prompt);
                        const safeEditor = escapeHTML(h.edited_by);
                        const date = new Date(h.edited_at + 'Z').toLocaleString();

                        const imgs = parseImages(h.image_path);
                        const coverImg = imgs.length > 0 ? imgs[0] : '';

                        const hasPlaceholders = /\\[(.*?)\\]/.test(h.prompt);
                        let historyCopyBtn = '';
                        if(hasPlaceholders) {
                            historyCopyBtn = `<button onclick="openTemplateModal('${h.prompt_id}', decodeURIComponent('${encodeForJS(h.prompt)}'))" class="bg-blue-700 hover:bg-blue-600 text-xs px-3 py-1 rounded font-bold transition-colors border border-blue-600">🧩 Fill Old</button>`;
                        } else {
                            historyCopyBtn = `<button onclick="copyToClipboard(this, decodeURIComponent('${encodeForJS(h.prompt)}'), null)" class="bg-gray-700 hover:bg-gray-600 text-xs px-3 py-1 rounded font-bold transition-colors">📋 Copy Old</button>`;
                        }
                        const rollbackBtn = `<button onclick="rollbackToVersion('${h.prompt_id}', '${h.history_id}')" class="bg-orange-700 hover:bg-orange-600 text-xs px-3 py-1 rounded font-bold transition-colors border border-orange-600">↩ Rollback</button>`;

                        container.innerHTML += `
                            <div class="bg-gray-900 rounded p-4 border border-gray-700">
                                <div class="flex justify-between items-center mb-2 border-b border-gray-700 pb-2">
                                    <span class="text-xs font-bold text-yellow-500">Version ${history.length - index}</span>
                                    <span class="text-xs text-gray-400">Changed by: <span class="text-white">${safeEditor}</span> on ${date}</span>
                                </div>
                                <div class="flex gap-4">
                                    <img src="/images/${coverImg}" onclick="openLightbox('/images/${coverImg}')" class="w-24 h-24 object-cover rounded border border-gray-700 flex-shrink-0 bg-gray-800 cursor-zoom-in">
                                    <div class="flex-grow min-w-0">
                                        <h4 class="font-bold text-white mb-1 truncate">${safeTitle}</h4>
                                        <p class="text-xs text-gray-400 mb-2 truncate">Author: ${safeAuthor} | Tags: ${safeTags}</p>
                                        <div class="bg-gray-800 p-2 rounded text-xs text-gray-300 font-mono mb-2 break-words max-h-32 overflow-y-auto">
                                            ${safePrompt}
                                        </div>
                                        <div class="flex gap-2 flex-wrap">
                                            ${historyCopyBtn}
                                            ${rollbackBtn}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        `;
                    });
                }
                document.getElementById('historyModal').classList.remove('hidden');
            }

            function openAuthorsModal() {
                const authorCounts = {};
                globalPrompts.forEach(p => {
                    const a = p.author.trim();
                    if(a) { authorCounts[a] = (authorCounts[a] || 0) + 1; }
                });
                
                const sortedAuthors = Object.keys(authorCounts).sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));
                const container = document.getElementById('authorsContainer');
                
                container.innerHTML = sortedAuthors.map(author => {
                    const safeAuthor = escapeHTML(author);
                    return `
                        <button onclick="triggerSearch(decodeURIComponent('author:${encodeForJS(author)}')); closeModal('authorsModal')" 
                                class="bg-gray-700 hover:bg-yellow-500 hover:text-black text-white px-3 py-2 rounded transition-colors text-sm flex items-center gap-2">
                            ${safeAuthor} 
                            <span class="bg-gray-900 text-gray-400 text-xs px-2 py-0.5 rounded-full">${authorCounts[author]}</span>
                        </button>
                    `;
                }).join('');
                
                if(sortedAuthors.length === 0) container.innerHTML = '<p class="text-gray-400 italic">No authors found.</p>';
                document.getElementById('authorsModal').classList.remove('hidden');
            }

            function openTagsModal() {
                const tagCounts = {};
                globalPrompts.forEach(p => {
                    if (p.tags) {
                        p.tags.split(',').forEach(t => {
                            const clean = t.trim();
                            if(clean) tagCounts[clean] = (tagCounts[clean] || 0) + 1;
                        });
                    }
                });
                
                const sortedTags = Object.keys(tagCounts).sort((a, b) => tagCounts[b] - tagCounts[a]);
                const container = document.getElementById('tagsContainer');
                
                document.getElementById('mergeOldTag').value = '';
                document.getElementById('mergeNewTag').value = '';
                
                container.innerHTML = sortedTags.map(tag => {
                    const safeTag = escapeHTML(tag);
                    return `
                        <button onclick="triggerSearch(decodeURIComponent('tag:${encodeForJS(tag)}')); closeModal('tagsModal')" 
                                class="bg-gray-700 hover:bg-yellow-500 hover:text-black text-white px-3 py-2 rounded transition-colors text-sm flex items-center gap-2">
                            ${safeTag} 
                            <span class="bg-gray-900 text-gray-400 text-xs px-2 py-0.5 rounded-full">${tagCounts[tag]}</span>
                        </button>
                    `;
                }).join('');
                
                if(sortedTags.length === 0) container.innerHTML = '<p class="text-gray-400 italic">No tags found.</p>';
                document.getElementById('tagsModal').classList.remove('hidden');
            }

            async function mergeTags() {
                const oldTag = document.getElementById('mergeOldTag').value.trim();
                const newTag = document.getElementById('mergeNewTag').value.trim();
                
                if(!oldTag || !newTag) { alert("Please provide both an old and a new tag name."); return; }
                if(!confirm(`Are you sure you want to merge '${oldTag}' into '${newTag}'?`)) return;

                const btn = document.getElementById('mergeTagBtn');
                btn.innerText = 'Merging...'; btn.disabled = true;

                const formData = new FormData();
                formData.append('old_tag', oldTag);
                formData.append('new_tag', newTag);

                try {
                    const res = await fetch('/api/tags/merge', { method: 'POST', body: formData });
                    if(res.ok) {
                        document.getElementById('mergeOldTag').value = '';
                        document.getElementById('mergeNewTag').value = '';
                        closeModal('tagsModal');
                        fetchPrompts();
                    } else {
                        const error = await res.json();
                        alert("Error: " + error.detail);
                    }
                } catch(e) { alert("Merge failed."); }
                btn.innerText = 'Merge'; btn.disabled = false;
            }

            function closeModal(modalId) {
                document.getElementById(modalId).classList.add('hidden');
            }

            async function submitForm(e) {
                e.preventDefault();
                if (mediaItems.length === 0) { alert('Please add at least one image.'); return; }
                
                const saveBtn = document.getElementById('saveButton');
                saveBtn.innerText = 'Saving...';
                saveBtn.disabled = true;

                const formData = new FormData(e.target);
                const isSharedCheckbox = document.getElementById('is_shared');
                if(!formData.has('is_shared')) formData.append('is_shared', isSharedCheckbox.checked ? 'true' : 'false');
                
                const mediaOrder = [];
                let newFileCounter = 0;
                
                mediaItems.forEach(item => {
                    if (item.type === 'existing') {
                        mediaOrder.push('existing:' + item.val);
                    } else if (item.type === 'new') {
                        mediaOrder.push('new:' + newFileCounter);
                        formData.append('new_images', item.val);
                        newFileCounter++;
                    }
                });
                formData.append('media_order', JSON.stringify(mediaOrder));

                const editId = document.getElementById('editPromptId').value;
                const url = editId ? `/api/prompts/${editId}` : '/api/prompts';
                const method = editId ? 'PUT' : 'POST';

                try {
                    const res = await fetch(url, { method: method, body: formData });
                    if (!res.ok) {
                        const error = await res.json();
                        alert("Error: " + error.detail);
                    } else {
                        closeModal('promptModal');
                        fetchPrompts();
                    }
                } catch(err) { alert("Upload failed."); }
                
                saveBtn.innerText = 'Save';
                saveBtn.disabled = false;
            }

            async function deletePrompt(id) {
                if(!confirm("Are you sure you want to permanently delete this prompt and all its history?")) return;
                const res = await fetch(`/api/prompts/${id}`, { method: 'DELETE' });
                if(res.ok) fetchPrompts();
            }

            function copyToClipboard(btn, text, promptId) {
                executeCopyFinal(btn, text, promptId);
            }

            // --- COLLECTIONS LOGIC ---
            function renderCollectionBadgesHtml(p) {
                if (!p.collection_ids || p.collection_ids.length === 0) return '';
                const badges = p.collection_ids.map(cid => {
                    const col = collections.find(c => c.id === cid);
                    if (!col) return '';
                    return `<span class="bg-teal-900/60 text-teal-300 text-xs px-2 py-0.5 rounded-full cursor-pointer hover:bg-teal-800 transition-colors border border-teal-700" onclick="setActiveCollection('${cid}')">📁 ${escapeHTML(col.name)}</span>`;
                }).filter(Boolean);
                if (badges.length === 0) return '';
                return `<div class="flex flex-wrap gap-1 mb-2">${badges.join('')}</div>`;
            }

            async function fetchCollections() {
                try {
                    const res = await fetch('/api/collections');
                    if (res.ok) collections = await res.json();
                } catch(e) {}
            }

            function openCollectionsModal() {
                renderCollectionsList();
                document.getElementById('newCollectionName').value = '';
                document.getElementById('collectionsModal').classList.remove('hidden');
            }

            function renderCollectionsList() {
                const container = document.getElementById('collectionsListContainer');
                if (collections.length === 0) {
                    container.innerHTML = '<p class="text-gray-400 italic text-center py-6">No collections yet. Create one above!</p>';
                    return;
                }
                container.innerHTML = collections.map(col => {
                    const safeName = escapeHTML(col.name);
                    const isActive = activeCollectionId === col.id;
                    return `
                        <div class="flex items-center gap-2 p-3 bg-gray-700 rounded border ${isActive ? 'border-teal-500' : 'border-gray-600'} group">
                            <button onclick="setActiveCollection('${col.id}')" class="flex-grow text-left font-medium hover:text-teal-300 transition-colors flex items-center gap-2 min-w-0">
                                <span class="text-lg">📁</span>
                                <span class="truncate">${safeName}</span>
                                <span class="text-xs text-gray-400 ml-2 flex-shrink-0">${col.prompt_count} prompts</span>
                                ${isActive ? '<span class="text-xs bg-teal-600 text-white px-2 py-0.5 rounded-full font-bold ml-1 flex-shrink-0">Active</span>' : ''}
                            </button>
                            <button onclick="startRenameCollection('${col.id}', decodeURIComponent('${encodeForJS(col.name)}'))" class="text-gray-400 hover:text-blue-400 transition-colors text-sm opacity-0 group-hover:opacity-100 flex-shrink-0 px-1" title="Rename">✏️</button>
                            <button onclick="deleteCollection('${col.id}')" class="text-gray-400 hover:text-red-400 transition-colors text-sm opacity-0 group-hover:opacity-100 flex-shrink-0 px-1" title="Delete">🗑️</button>
                        </div>
                    `;
                }).join('');
            }

            async function createCollection() {
                const input = document.getElementById('newCollectionName');
                const name = input.value.trim();
                if (!name) return;
                const formData = new FormData();
                formData.append('name', name);
                try {
                    const res = await fetch('/api/collections', { method: 'POST', body: formData });
                    if (res.ok) {
                        const col = await res.json();
                        collections.push(col);
                        input.value = '';
                        renderCollectionsList();
                    }
                } catch(e) { alert("Failed to create collection."); }
            }

            async function deleteCollection(id) {
                if (!confirm("Delete this collection? Prompts will not be deleted.")) return;
                try {
                    const res = await fetch(`/api/collections/${id}`, { method: 'DELETE' });
                    if (res.ok) {
                        collections = collections.filter(c => c.id !== id);
                        if (activeCollectionId === id) clearCollectionFilter();
                        globalPrompts.forEach(p => {
                            p.collection_ids = (p.collection_ids || []).filter(cid => cid !== id);
                        });
                        renderCollectionsList();
                        triggerRenderReset();
                    }
                } catch(e) { alert("Failed to delete collection."); }
            }

            async function startRenameCollection(id, currentName) {
                const newName = prompt("Rename collection:", currentName);
                if (!newName || newName.trim() === currentName) return;
                const formData = new FormData();
                formData.append('name', newName.trim());
                try {
                    const res = await fetch(`/api/collections/${id}`, { method: 'PUT', body: formData });
                    if (res.ok) {
                        const col = collections.find(c => c.id === id);
                        if (col) col.name = newName.trim();
                        renderCollectionsList();
                        if (activeCollectionId === id) {
                            document.getElementById('activeCollectionName').innerText = newName.trim();
                        }
                        triggerRenderReset();
                    }
                } catch(e) { alert("Failed to rename collection."); }
            }

            function setActiveCollection(id) {
                activeCollectionId = id;
                const col = collections.find(c => c.id === id);
                if (col) {
                    document.getElementById('activeCollectionName').innerText = col.name;
                    document.getElementById('activeCollectionBanner').classList.remove('hidden');
                }
                closeModal('collectionsModal');
                triggerRenderReset();
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }

            function clearCollectionFilter() {
                activeCollectionId = null;
                document.getElementById('activeCollectionBanner').classList.add('hidden');
                triggerRenderReset();
            }

            function openCollectionAssignModal(promptId) {
                document.getElementById('assignPromptId').value = promptId;
                const p = globalPrompts.find(x => x.id === promptId);
                const currentCollectionIds = p ? (p.collection_ids || []) : [];
                const container = document.getElementById('collectionAssignList');
                if (collections.length === 0) {
                    container.innerHTML = '<p class="text-gray-400 italic text-center py-4">No collections yet. Create one via 📁 Collections in the toolbar.</p>';
                } else {
                    container.innerHTML = collections.map(col => {
                        const safeName = escapeHTML(col.name);
                        const isChecked = currentCollectionIds.includes(col.id);
                        return `
                            <label class="flex items-center gap-3 p-3 bg-gray-700 rounded border border-gray-600 cursor-pointer hover:border-teal-500 transition-colors">
                                <input type="checkbox" class="w-4 h-4 rounded accent-teal-500 flex-shrink-0" 
                                       ${isChecked ? 'checked' : ''} 
                                       onchange="togglePromptInCollection('${col.id}', '${promptId}', this.checked)">
                                <span class="font-medium">📁 ${safeName}</span>
                                <span class="text-xs text-gray-400 ml-auto">${col.prompt_count} prompts</span>
                            </label>
                        `;
                    }).join('');
                }
                document.getElementById('collectionAssignModal').classList.remove('hidden');
            }

            async function togglePromptInCollection(collectionId, promptId, shouldAdd) {
                try {
                    let res;
                    if (shouldAdd) {
                        const formData = new FormData();
                        formData.append('prompt_id', promptId);
                        res = await fetch(`/api/collections/${collectionId}/prompts`, { method: 'POST', body: formData });
                    } else {
                        res = await fetch(`/api/collections/${collectionId}/prompts/${promptId}`, { method: 'DELETE' });
                    }
                    if (res.ok) {
                        const p = globalPrompts.find(x => x.id === promptId);
                        if (p) {
                            if (shouldAdd && !p.collection_ids.includes(collectionId)) p.collection_ids.push(collectionId);
                            else if (!shouldAdd) p.collection_ids = p.collection_ids.filter(id => id !== collectionId);
                        }
                        const col = collections.find(c => c.id === collectionId);
                        if (col) col.prompt_count = Math.max(0, col.prompt_count + (shouldAdd ? 1 : -1));
                        if (activeCollectionId === collectionId && !shouldAdd) triggerRenderReset();
                        
                        const cardEl = document.getElementById('card-' + promptId);
                        if (cardEl) {
                            const badgeWrapper = cardEl.querySelector('.collection-badge-wrapper');
                            if (badgeWrapper && p) badgeWrapper.innerHTML = renderCollectionBadgesHtml(p);
                        }
                    }
                } catch(e) { alert("Failed to update collection."); }
            }

            function openBulkCollectionModal() {
                if (selectedPrompts.size === 0) return alert("Please select at least one prompt.");
                if (collections.length === 0) {
                    alert("No collections exist yet. Create one via 📁 Collections in the toolbar.");
                    return;
                }
                document.getElementById('bulkCollectCount').innerText = selectedPrompts.size;
                const container = document.getElementById('bulkCollectionList');
                container.innerHTML = collections.map(col => {
                    const safeName = escapeHTML(col.name);
                    return `<button onclick="bulkAddToCollection('${col.id}')" class="w-full text-left p-3 bg-gray-700 hover:bg-teal-700 rounded border border-gray-600 hover:border-teal-500 transition-colors flex items-center gap-2">📁 ${safeName} <span class="text-xs text-gray-400 ml-auto">${col.prompt_count} prompts</span></button>`;
                }).join('');
                document.getElementById('bulkCollectionModal').classList.remove('hidden');
            }

            async function bulkAddToCollection(collectionId) {
                const ids = Array.from(selectedPrompts);
                let added = 0;
                for (const pid of ids) {
                    const formData = new FormData();
                    formData.append('prompt_id', pid);
                    try {
                        const res = await fetch(`/api/collections/${collectionId}/prompts`, { method: 'POST', body: formData });
                        if (res.ok) added++;
                    } catch(e) {}
                }
                closeModal('bulkCollectionModal');
                const col = collections.find(c => c.id === collectionId);
                if (col) col.prompt_count = Math.max(0, col.prompt_count + added);
                toggleBulkMode();
                fetchPrompts();
            }

            let initialCols = localStorage.getItem('nanobananaLayoutCols');
            if (!initialCols) initialCols = window.innerWidth < 768 ? 1 : 3;
            else initialCols = parseInt(initialCols);
            setLayout(initialCols);
            fetchPrompts();
        </script>
    </body>
    </html>
    """
    
    admin_badge = '<span class="text-red-400 font-bold text-xs ml-1 border border-red-400 px-1 rounded" title="Admin Privileges Active">ADMIN</span>' if is_admin(user) else ''
    is_adm_str = "true" if is_admin(user) else "false"
    
    html = html_template.replace("__USER_EMAIL__", user.get('email', 'User'))
    html = html.replace("__ADMIN_BADGE__", admin_badge)
    html = html.replace("__IS_ADMIN__", is_adm_str)
    
    return html
