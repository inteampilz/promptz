# CLAUDE.md — AI Assistant Guide for promptz

## Project Overview

**NanoBanana Prompts** is a collaborative prompt management application for AI image generation. It allows teams to create, share, fork, and organize prompts with version history, automatic metadata extraction, and intelligent tagging via Google Gemini.

- **Single-file architecture**: The entire backend and frontend live in `app.py` (2,078 lines)
- **Project alias/branding**: "NanoBanana Prompts" (🍌), container name `nanobanana-prompts1`
- **License**: GNU GPL v3

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Web framework | FastAPI (async) |
| ASGI server | Uvicorn |
| Database | SQLite (file-based, `/app/data/prompts.db`) |
| Authentication | OIDC via Authlib |
| AI integration | Google Gemini 2.5 Flash (`google-genai`) |
| Image processing | Pillow (PIL) |
| Frontend | Single-page app embedded in HTML template (Tailwind CSS, vanilla JS) |
| Deployment | Docker Compose (no Dockerfile — deps installed at container start) |

---

## Repository Structure

```
promptz/
├── app.py           # Entire application (backend + frontend HTML template)
├── docker_compose   # Docker Compose config (note: no .yml extension)
├── LICENSE          # GNU GPL v3
└── CLAUDE.md        # This file
```

There is no `requirements.txt`, `Dockerfile`, or separate frontend directory — dependencies are installed inline by the Docker command at startup.

---

## Running the Application

### Via Docker Compose (production/staging)

```bash
# Edit docker_compose to set real values for all environment variables, then:
docker compose -f docker_compose up -d
```

### Local development (without Docker)

```bash
pip install fastapi uvicorn python-multipart authlib httpx itsdangerous pillow google-genai
uvicorn app:app --reload --port 8000
```

The app requires a writable `/app/data` directory (or change `DATA_DIR` in `app.py` for local runs).

---

## Environment Variables

All configuration is passed via environment variables. There are **no .env files** in this repo.

| Variable | Required | Description |
|---|---|---|
| `SESSION_SECRET` | Yes | Secret key for session encryption. Use a long random string in production. |
| `OIDC_CLIENT_ID` | Yes | OIDC/OAuth2 client ID |
| `OIDC_CLIENT_SECRET` | Yes | OIDC/OAuth2 client secret |
| `OIDC_DISCOVERY_URL` | Yes | OIDC discovery endpoint (e.g. `https://auth.example.com/.well-known/openid-configuration`) |
| `GEMINI_API_KEY` | Yes | Google Gemini API key for auto-tag generation |

> The `docker_compose` file contains placeholder values (`secret`). **Never commit real secrets.**

---

## Database Schema

SQLite database at `/app/data/prompts.db`. Schema is auto-created and migrated at startup by `init_db()`.

### `prompts` table
| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | UUID |
| `title` | TEXT | |
| `prompt` | TEXT | The actual prompt text |
| `author` | TEXT | Display name |
| `tags` | TEXT | Comma-separated string |
| `image_path` | TEXT | JSON array of WebP filenames |
| `user_email` | TEXT | Owner's email (from OIDC) |
| `is_shared` | INTEGER | Boolean (0/1) |
| `copy_count` | INTEGER | Usage tracking (migration-added) |
| `forked_from` | TEXT | Parent prompt ID (migration-added) |

### `prompt_history` table
Stores a snapshot every time a prompt is edited.

### `favorites` table
| Column | Notes |
|---|---|
| `user_email` | |
| `prompt_id` | |
Composite unique key `(user_email, prompt_id)`.

### Schema migrations
New columns are added via `ALTER TABLE` in `init_db()` — checked with `PRAGMA table_info`. When adding new columns, follow the same pattern:
```python
if 'new_column' not in columns:
    c.execute("ALTER TABLE prompts ADD COLUMN new_column TYPE DEFAULT value")
```

---

## API Endpoints

All API routes require an authenticated session (OIDC). Unauthenticated requests return HTTP 401.

```
GET    /                              SPA frontend
GET    /images/{filename}             Static image serving

GET    /login                         Initiate OIDC login
GET    /auth                          OIDC callback
GET    /logout                        Clear session

GET    /api/prompts                   List prompts (owned + shared)
POST   /api/prompts                   Create prompt (multipart/form-data)
PUT    /api/prompts/{id}              Update prompt (saves history snapshot)
DELETE /api/prompts/{id}              Delete prompt + orphaned images

GET    /api/prompts/{id}/history      Version history for a prompt
POST   /api/prompts/{id}/copy         Increment copy_count
POST   /api/prompts/{id}/favorite     Toggle favorite status

POST   /api/prompts/bulk/delete       Delete multiple prompts
POST   /api/prompts/bulk/tag          Add tag to multiple prompts

POST   /api/extract-metadata          Extract prompt text from image EXIF/IPTC
POST   /api/tags/auto                 Auto-generate tags via Gemini
POST   /api/tags/merge                Rename/merge tags across all prompts

GET    /api/export/{format}           Export data as json | csv | zip
POST   /api/import                    Import from zip | json | csv
```

---

## Key Code Conventions

### Backend (Python / FastAPI)

- **Async throughout**: All route handlers and I/O functions use `async def`.
- **Raw SQL with parameterized queries**: No ORM. Always use `?` placeholders — never f-strings or `.format()` for SQL values.
- **`sqlite3.connect()`** is called per-request (no connection pooling). Use `conn.row_factory = sqlite3.Row` when dict-like access is needed.
- **Image handling**: Images are always converted to WebP, resized to max 1024×1024 via `optimize_and_save_image()`. UUIDs are used as filenames.
- **Image paths as JSON**: `image_path` column stores a JSON array of filenames (not full paths), e.g. `["abc.webp", "def.webp"]`.
- **Error responses**: Use `raise HTTPException(status_code=..., detail="...")` consistently.
- **Auth check pattern**: Session is read via `request.session.get("user")`. If absent, raise `HTTPException(status_code=401)`.

### Frontend (Embedded HTML/JS)

- The entire frontend is a Jinja2-style f-string HTML template returned by the `GET /` route.
- Tailwind CSS is loaded from CDN.
- All JS is vanilla (no frameworks/bundlers). State lives in module-level variables and `localStorage`.
- **XSS prevention**: Always use the `escapeHTML()` helper when rendering user content into the DOM.
- **Search filter syntax** supported by the client-side filter:
  - `tag:keyword` — filter by tag
  - `author:name` — filter by author
  - `is:mine` / `is:shared` / `is:favorite` — status filters
  - `-tag:keyword` — negation prefix
- **Template variable system**: Prompts can contain `[PLACEHOLDER]` markers that trigger a fill-in dialog on copy.
- **Pagination**: Items are rendered in batches of 24 via infinite scroll.

---

## Image & Media Handling

- Uploaded images are validated with `img.verify()` before processing.
- RGBA/palette images are converted to RGB before WebP encoding.
- EXIF/IPTC metadata is extracted via Pillow to auto-populate prompt text from AI-generated images.
- When a prompt is **forked**, images are shared (referenced by filename) rather than copied. Deletion logic checks for other references before removing image files.

---

## AI Integration (Google Gemini)

- Model used: `gemini-2.5-flash` (via `google-genai` SDK).
- Called in `POST /api/tags/auto` to generate tags from prompt text.
- Requires `GEMINI_API_KEY` environment variable.
- If Gemini is unavailable or returns an error, the endpoint returns HTTP 500 with the error detail.

---

## Deployment Notes

### Docker Compose

The `docker_compose` file (no extension) defines two services:

1. **`prompt-app`**: Main application
   - Image: `python:3.11-slim`
   - Deps installed at container start (no pre-built image)
   - Data persisted in named volume `prompt_data` mounted at `/app/data`
   - Requires an external Docker network named `proxy` (for reverse proxy integration)

2. **`backup-service`**: Alpine container that runs a daily cron job at 02:00 to create a `.tar.gz` backup of `prompt_data` into a bind-mounted backup folder.

### Path configuration in docker_compose

Two paths must be customized before deploying (marked with `# HIER ANPASSEN`):
- The host path to the directory containing `app.py` (bind-mounted to `/app/code`)
- The host path to the backups folder

---

## No Tests

There are currently **no automated tests** in this project. If adding tests, use `pytest` with `httpx.AsyncClient` for FastAPI endpoint testing.

---

## Code Language Note

Some inline comments in `app.py` are written in **German** (e.g., `# HIER ANPASSEN` = "adjust here", `# Konfiguration` = "configuration"). This is intentional — maintain consistency with the existing language in whichever section you're editing.

---

## Common Tasks

### Add a new API endpoint
1. Add the route function in `app.py` near related routes (grouped by feature area).
2. Use `async def`, inject `request: Request` for session access.
3. Check auth at the top: `user = request.session.get("user"); if not user: raise HTTPException(401)`.
4. Use parameterized SQL. Open and close `sqlite3.connect()` within the function.
5. Add the corresponding frontend fetch call in the embedded JS section.

### Add a new database column
1. Add to the `CREATE TABLE IF NOT EXISTS` statement.
2. Add a migration block in `init_db()` following the existing `PRAGMA table_info` pattern.

### Modify frontend behavior
1. Find the relevant JS function in the HTML template section of `app.py` (starts around line 667).
2. Use `escapeHTML()` for any user-supplied values rendered into innerHTML.
3. Keep state in module-level JS variables; persist UI preferences to `localStorage`.

### Update Docker dependencies
Edit the `pip install` command in `docker_compose` and recreate the container.
