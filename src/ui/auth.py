from __future__ import annotations

import hashlib
import os
import re
import secrets
import smtplib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from urllib.parse import urlencode
from typing import Any, Iterator, Optional


PASSWORD_MIN_LENGTH = 8
_PBKDF2_ITERATIONS = 260_000
_USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,63}$")
_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")
RUN_MANIFEST_NAME = "run_manifest.json"
VALID_ROLES = {"admin", "user"}
VALID_ACCOUNT_STATUSES = {"pending", "approved", "rejected"}
BOOTSTRAP_ADMIN_USERNAME = "william"
BOOTSTRAP_ADMIN_EMAIL = "schertzer.william@gmail.com"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"


def normalize_identifier(value: str) -> str:
    return (value or "").strip().lower()


def validate_username(username: str) -> str:
    username = (username or "").strip()
    if not _USERNAME_RE.match(username):
        raise ValueError("Use 3-64 letters, numbers, underscores, dots, or dashes.")
    return username


def validate_email(email: str) -> str:
    email = normalize_identifier(email)
    if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        raise ValueError("Enter a valid email address.")
    if not (email.endswith(".edu") or email.endswith(".com")):
        raise ValueError("Use an email address ending in .edu or .com.")
    return email


def validate_required_profile_field(value: str, label: str) -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError(f"{label} is required.")
    return value


def validate_password(password: str) -> None:
    if len(password or "") < PASSWORD_MIN_LENGTH:
        raise ValueError(f"Password must be at least {PASSWORD_MIN_LENGTH} characters.")


def _profile_slug(username: str) -> str:
    slug = _SLUG_RE.sub("_", username.strip().lower()).strip("._-")
    return slug or secrets.token_hex(4)


def hash_password(password: str) -> str:
    validate_password(password)
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("ascii"),
        _PBKDF2_ITERATIONS,
    ).hex()
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, iterations_raw, salt, expected = stored_hash.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            (password or "").encode("utf-8"),
            salt.encode("ascii"),
            int(iterations_raw),
        ).hex()
        return secrets.compare_digest(digest, expected)
    except Exception:
        return False


def token_digest(token: str) -> str:
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


class AuthStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    first_name TEXT NOT NULL DEFAULT '',
                    last_name TEXT NOT NULL DEFAULT '',
                    institution TEXT NOT NULL DEFAULT '',
                    display_name TEXT NOT NULL DEFAULT '',
                    profile_slug TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    account_status TEXT NOT NULL DEFAULT 'approved',
                    approved_at TEXT,
                    approved_by_user_id INTEGER,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login_at TEXT
                );

                CREATE TABLE IF NOT EXISTS password_reset_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    expires_at TEXT NOT NULL,
                    used_at TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS auth_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    last_seen_at TEXT,
                    revoked_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );
                """
            )
            existing_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(users)").fetchall()
            }
            for column_name in ("first_name", "last_name", "institution"):
                if column_name not in existing_columns:
                    conn.execute(
                        f"ALTER TABLE users ADD COLUMN {column_name} TEXT NOT NULL DEFAULT ''"
                    )
            if "account_status" not in existing_columns:
                conn.execute(
                    "ALTER TABLE users ADD COLUMN account_status "
                    "TEXT NOT NULL DEFAULT 'approved'"
                )
            if "approved_at" not in existing_columns:
                conn.execute("ALTER TABLE users ADD COLUMN approved_at TEXT")
            if "approved_by_user_id" not in existing_columns:
                conn.execute("ALTER TABLE users ADD COLUMN approved_by_user_id INTEGER")

    def user_count(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def admin_count(self) -> int:
        with self.connect() as conn:
            return int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM users
                    WHERE is_active = 1
                      AND account_status = 'approved'
                      AND role = 'admin'
                    """
                ).fetchone()[0]
            )

    def _unique_profile_slug(self, conn: sqlite3.Connection, username: str) -> str:
        base = _profile_slug(username)
        slug = base
        while conn.execute("SELECT 1 FROM users WHERE profile_slug = ?", (slug,)).fetchone():
            slug = f"{base}-{secrets.token_hex(3)}"
        return slug

    def create_user(
        self,
        *,
        username: str,
        email: str,
        password: str,
        first_name: str = "",
        last_name: str = "",
        institution: str = "",
        display_name: str = "",
        account_status: str = "approved",
    ) -> sqlite3.Row:
        username = validate_username(username)
        email = validate_email(email)
        first_name = validate_required_profile_field(first_name, "First name")
        last_name = validate_required_profile_field(last_name, "Last name")
        institution = validate_required_profile_field(institution, "Institution")
        display_name = (display_name or "").strip() or f"{first_name} {last_name}".strip()
        account_status = (account_status or "").strip().lower()
        if account_status not in VALID_ACCOUNT_STATUSES:
            raise ValueError("Invalid account status.")
        password_hash = hash_password(password)
        now = utc_now_iso()
        with self.connect() as conn:
            # Rejected requests from older versions are not accounts and must not
            # permanently reserve their username or email address.
            rejected_user_ids = """
                SELECT id FROM users
                WHERE account_status = 'rejected'
                  AND (lower(username) = ? OR lower(email) = ?)
            """
            conn.execute(
                f"DELETE FROM password_reset_tokens WHERE user_id IN ({rejected_user_ids})",
                (username, email),
            )
            conn.execute(
                f"DELETE FROM auth_sessions WHERE user_id IN ({rejected_user_ids})",
                (username, email),
            )
            conn.execute(
                f"DELETE FROM users WHERE id IN ({rejected_user_ids})",
                (username, email),
            )
            role = (
                "admin"
                if account_status == "approved" and self.user_count() == 0
                else "user"
            )
            slug = self._unique_profile_slug(conn, username)
            try:
                cur = conn.execute(
                    """
                    INSERT INTO users (
                        username, email, first_name, last_name, institution,
                        display_name, profile_slug,
                        password_hash, role, account_status, approved_at,
                        created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        username,
                        email,
                        first_name,
                        last_name,
                        institution,
                        display_name,
                        slug,
                        password_hash,
                        role,
                        account_status,
                        now if account_status == "approved" else None,
                        now,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("Username or email is already registered.") from exc
            return conn.execute(
                "SELECT * FROM users WHERE id = ? AND is_active = 1",
                (cur.lastrowid,),
            ).fetchone()

    def request_account(self, **profile: Any) -> sqlite3.Row:
        if self.admin_count() == 0:
            raise ValueError("Account requests require an existing SALSA administrator.")
        return self.create_user(account_status="pending", **profile)

    def list_users(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(
                conn.execute(
                    """
                    SELECT *
                    FROM users
                    WHERE is_active = 1
                      AND account_status = 'approved'
                    ORDER BY lower(username)
                    """
                ).fetchall()
            )

    def list_pending_users(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(
                conn.execute(
                    """
                    SELECT *
                    FROM users
                    WHERE is_active = 1
                      AND account_status = 'pending'
                    ORDER BY created_at
                    """
                ).fetchall()
            )

    def list_admin_emails(self) -> list[str]:
        with self.connect() as conn:
            return [
                str(row["email"])
                for row in conn.execute(
                    """
                    SELECT email
                    FROM users
                    WHERE is_active = 1
                      AND account_status = 'approved'
                      AND role = 'admin'
                    ORDER BY lower(email)
                    """
                ).fetchall()
            ]

    def get_user(self, user_id: int) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT * FROM users
                WHERE id = ?
                  AND is_active = 1
                  AND account_status = 'approved'
                """,
                (user_id,),
            ).fetchone()

    def find_user(self, identifier: str) -> Optional[sqlite3.Row]:
        ident = normalize_identifier(identifier)
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT * FROM users
                WHERE is_active = 1
                  AND account_status = 'approved'
                  AND (lower(username) = ? OR lower(email) = ?)
                """,
                (ident, ident),
            ).fetchone()

    def authenticate(self, identifier: str, password: str) -> Optional[sqlite3.Row]:
        user = self.find_user(identifier)
        if not user or not verify_password(password, user["password_hash"]):
            return None
        with self.connect() as conn:
            conn.execute(
                "UPDATE users SET last_login_at = ?, updated_at = ? WHERE id = ?",
                (utc_now_iso(), utc_now_iso(), user["id"]),
            )
        return self.get_user(user["id"])

    def create_auth_session_token(self, user_id: int, *, ttl_days: int = 30) -> str:
        token = secrets.token_urlsafe(32)
        now = utc_now_iso()
        expires_at = (
            datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=ttl_days)
        ).isoformat(timespec="seconds") + "Z"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO auth_sessions
                    (user_id, token_hash, created_at, expires_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, token_digest(token), now, expires_at, now),
            )
        return token

    def user_from_auth_session_token(self, token: str) -> Optional[sqlite3.Row]:
        digest = token_digest(token)
        now = utc_now_iso()
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT users.*
                FROM auth_sessions
                JOIN users ON users.id = auth_sessions.user_id
                WHERE auth_sessions.token_hash = ?
                  AND auth_sessions.revoked_at IS NULL
                  AND auth_sessions.expires_at > ?
                  AND users.is_active = 1
                  AND users.account_status = 'approved'
                """,
                (digest, now),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE auth_sessions SET last_seen_at = ? WHERE token_hash = ?",
                (now, digest),
            )
            return row

    def revoke_auth_session_token(self, token: str) -> None:
        if not token:
            return
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE auth_sessions
                SET revoked_at = ?
                WHERE token_hash = ? AND revoked_at IS NULL
                """,
                (utc_now_iso(), token_digest(token)),
            )

    def set_user_role(self, user_id: int, role: str) -> sqlite3.Row:
        role = (role or "").strip().lower()
        if role not in VALID_ROLES:
            raise ValueError("Role must be either admin or user.")

        with self.connect() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE id = ? AND is_active = 1",
                (user_id,),
            ).fetchone()
            if user is None:
                raise ValueError("User not found.")

            if user["role"] == "admin" and role != "admin":
                admin_count = int(
                    conn.execute(
                        """
                        SELECT COUNT(*) FROM users
                        WHERE is_active = 1
                          AND account_status = 'approved'
                          AND role = 'admin'
                        """
                    ).fetchone()[0]
                )
                if admin_count <= 1:
                    raise ValueError("Cannot remove the last admin.")

            conn.execute(
                "UPDATE users SET role = ?, updated_at = ? WHERE id = ?",
                (role, utc_now_iso(), user_id),
            )
            return conn.execute(
                "SELECT * FROM users WHERE id = ? AND is_active = 1",
                (user_id,),
            ).fetchone()

    def delete_user(self, user_id: int, deleted_by_user_id: int) -> None:
        with self.connect() as conn:
            actor = conn.execute(
                """
                SELECT * FROM users
                WHERE id = ?
                  AND is_active = 1
                  AND account_status = 'approved'
                  AND role = 'admin'
                """,
                (deleted_by_user_id,),
            ).fetchone()
            if actor is None:
                raise PermissionError("Only an approved admin can delete accounts.")
            if user_id == deleted_by_user_id:
                raise ValueError("You cannot delete your own account.")

            user = conn.execute(
                "SELECT * FROM users WHERE id = ? AND is_active = 1",
                (user_id,),
            ).fetchone()
            if user is None:
                raise ValueError("User not found.")
            if user["role"] == "admin" and user["account_status"] == "approved":
                admin_count = int(
                    conn.execute(
                        """
                        SELECT COUNT(*) FROM users
                        WHERE is_active = 1
                          AND account_status = 'approved'
                          AND role = 'admin'
                        """
                    ).fetchone()[0]
                )
                if admin_count <= 1:
                    raise ValueError("Cannot delete the last admin.")

            conn.execute("DELETE FROM password_reset_tokens WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM auth_sessions WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))

    def approve_account(self, user_id: int, approved_by_user_id: int) -> sqlite3.Row:
        now = utc_now_iso()
        with self.connect() as conn:
            approver = conn.execute(
                """
                SELECT 1 FROM users
                WHERE id = ?
                  AND is_active = 1
                  AND account_status = 'approved'
                  AND role = 'admin'
                """,
                (approved_by_user_id,),
            ).fetchone()
            if approver is None:
                raise PermissionError("Only an approved admin can approve accounts.")
            user = conn.execute(
                """
                SELECT * FROM users
                WHERE id = ? AND is_active = 1 AND account_status = 'pending'
                """,
                (user_id,),
            ).fetchone()
            if user is None:
                raise ValueError("Pending account request not found.")
            conn.execute(
                """
                UPDATE users
                SET account_status = 'approved',
                    approved_at = ?,
                    approved_by_user_id = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, approved_by_user_id, now, user_id),
            )
            return conn.execute(
                "SELECT * FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()

    def reject_account(self, user_id: int, rejected_by_user_id: int) -> None:
        with self.connect() as conn:
            approver = conn.execute(
                """
                SELECT 1 FROM users
                WHERE id = ?
                  AND is_active = 1
                  AND account_status = 'approved'
                  AND role = 'admin'
                """,
                (rejected_by_user_id,),
            ).fetchone()
            if approver is None:
                raise PermissionError("Only an approved admin can reject accounts.")
            user = conn.execute(
                """
                SELECT 1 FROM users
                WHERE id = ? AND is_active = 1 AND account_status = 'pending'
                """,
                (user_id,),
            ).fetchone()
            if user is None:
                raise ValueError("Pending account request not found.")
            conn.execute("DELETE FROM password_reset_tokens WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM auth_sessions WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))

    def bootstrap_default_admin(self) -> Optional[sqlite3.Row]:
        """
        Ensure the project owner's account is an admin when it exists.

        This is intentionally non-creating: it promotes the matching account but
        never invents a user or password.
        """
        ident_username = normalize_identifier(BOOTSTRAP_ADMIN_USERNAME)
        ident_email = normalize_identifier(BOOTSTRAP_ADMIN_EMAIL)
        with self.connect() as conn:
            user = conn.execute(
                """
                SELECT *
                FROM users
                WHERE is_active = 1
                  AND (lower(username) = ? OR lower(email) = ?)
                ORDER BY id
                LIMIT 1
                """,
                (ident_username, ident_email),
            ).fetchone()
            if user is None:
                return None
            if user["role"] != "admin" or user["account_status"] != "approved":
                conn.execute(
                    """
                    UPDATE users
                    SET role = 'admin',
                        account_status = 'approved',
                        approved_at = COALESCE(approved_at, ?),
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (utc_now_iso(), utc_now_iso(), user["id"]),
                )
            return conn.execute(
                "SELECT * FROM users WHERE id = ? AND is_active = 1",
                (user["id"],),
            ).fetchone()

    def set_password(self, user_id: int, new_password: str) -> None:
        password_hash = hash_password(new_password)
        with self.connect() as conn:
            conn.execute(
                "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                (password_hash, utc_now_iso(), user_id),
            )

    def create_password_reset_token(
        self,
        user_id: int,
        *,
        ttl_minutes: int = 60,
    ) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = (
            datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=ttl_minutes)
        ).isoformat(timespec="seconds") + "Z"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO password_reset_tokens
                    (user_id, token_hash, expires_at, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, token_digest(token), expires_at, utc_now_iso()),
            )
        return token

    def reset_password_with_token(
        self,
        *,
        identifier: str,
        token: str,
        new_password: str,
    ) -> bool:
        user = self.find_user(identifier)
        if not user:
            return False
        validate_password(new_password)
        now = utc_now_iso()
        digest = token_digest(token)
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM password_reset_tokens
                WHERE user_id = ?
                  AND token_hash = ?
                  AND used_at IS NULL
                  AND expires_at > ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (user["id"], digest, now),
            ).fetchone()
            if not row:
                return False
            conn.execute(
                "UPDATE password_reset_tokens SET used_at = ? WHERE id = ?",
                (now, row["id"]),
            )
            conn.execute(
                "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
                (hash_password(new_password), now, user["id"]),
            )
        return True


def data_root_from_env(default_root: Path) -> Path:
    root = Path(os.environ.get("SALSA_DATA_ROOT", str(default_root))).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def user_workspaces_root(data_root: Path, user: sqlite3.Row) -> Path:
    root = Path(data_root) / "users" / str(user["profile_slug"]) / "workspaces"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def user_profile_root(data_root: Path, user: sqlite3.Row) -> Path:
    root = Path(data_root) / "users" / str(user["profile_slug"])
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def is_admin_user(user: Optional[sqlite3.Row]) -> bool:
    return bool(user is not None and user["role"] == "admin")


def safe_child_path(root: Path, *parts: str) -> Path:
    root = Path(root).resolve()
    path = root.joinpath(*parts).resolve()
    if path != root and root not in path.parents:
        raise PermissionError(f"Path escapes root: {path}")
    return path


def write_run_manifest(
    run_dir: Path,
    *,
    user_id: Any,
    username: str,
    workspace: str,
    run_id: str,
) -> Path:
    import json

    manifest = {
        "owner_user_id": str(user_id),
        "owner_username": username,
        "workspace": workspace,
        "run_id": run_id,
        "created_at": utc_now_iso(),
    }
    path = Path(run_dir) / RUN_MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def run_belongs_to_user(run_dir: Path, workspaces_root: Path, user: sqlite3.Row) -> bool:
    try:
        root = Path(workspaces_root).resolve()
        path = Path(run_dir).resolve()
        path.relative_to(root)
    except Exception:
        return False

    manifest_path = Path(run_dir) / RUN_MANIFEST_NAME
    if not manifest_path.exists():
        return True

    try:
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(manifest.get("owner_user_id")) == str(user["id"])


def build_reset_url(public_url: str, token: str, identifier: str = "") -> str:
    public_url = (public_url or "").strip().rstrip("/")
    if not public_url:
        return token
    params = {"reset_token": token}
    identifier = (identifier or "").strip()
    if identifier:
        params["reset_identifier"] = identifier
    return f"{public_url}/?{urlencode(params)}"


def _send_email(message: EmailMessage) -> bool:
    host = os.environ.get("SALSA_SMTP_HOST")
    if not host or not message.get("From"):
        return False

    port = int(os.environ.get("SALSA_SMTP_PORT", "587"))
    local_hostname = os.environ.get("SALSA_SMTP_LOCAL_HOSTNAME")
    username = os.environ.get("SALSA_SMTP_USERNAME")
    password = os.environ.get("SALSA_SMTP_PASSWORD")
    use_tls = os.environ.get("SALSA_SMTP_TLS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }

    with smtplib.SMTP(
        host,
        port,
        local_hostname=local_hostname,
        timeout=10,
    ) as smtp:
        if use_tls:
            smtp.starttls()
        if username and password:
            smtp.login(username, password)
        smtp.send_message(message)
    return True


def _new_email(*, subject: str, recipients: list[str]) -> Optional[EmailMessage]:
    sender = os.environ.get("SALSA_SMTP_FROM")
    recipients = [email.strip() for email in recipients if email and email.strip()]
    if not sender or not recipients:
        return None
    local_hostname = os.environ.get("SALSA_SMTP_LOCAL_HOSTNAME")
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message["Date"] = formatdate(localtime=False)
    message["Message-ID"] = make_msgid(domain=local_hostname)
    return message


def send_password_reset_email(email: str, token: str) -> bool:
    message = _new_email(
        subject="Reset your SALSA password",
        recipients=[email],
    )
    if message is None:
        return False
    message.set_content(
        "Use this one-time token to reset your SALSA password. "
        "It expires soon and can only be used once.\n\n"
        f"Reset token: {token}\n"
    )
    return _send_email(message)


def send_account_request_email(
    admin_emails: list[str],
    requested_user: sqlite3.Row,
) -> bool:
    message = _new_email(
        subject=f"SALSA account request: {requested_user['username']}",
        recipients=admin_emails,
    )
    if message is None:
        return False
    message.set_content(
        "A new user is requesting access to SALSA.\n\n"
        f"Name: {requested_user['first_name']} {requested_user['last_name']}\n"
        f"Username: {requested_user['username']}\n"
        f"Email: {requested_user['email']}\n"
        f"Institution/organization: {requested_user['institution']}\n"
        f"Requested at: {requested_user['created_at']}\n\n"
        "An administrator must approve this request from the SALSA Admin page "
        "before the user can sign in.\n"
    )
    return _send_email(message)


def send_account_approved_email(approved_user: sqlite3.Row) -> bool:
    public_url = os.environ.get("SALSA_PUBLIC_URL", "").strip().rstrip("/")
    if not public_url:
        return False
    message = _new_email(
        subject="Your SALSA account has been approved",
        recipients=[approved_user["email"]],
    )
    if message is None:
        return False
    message.set_content(
        f"Hello {approved_user['first_name']},\n\n"
        "Your SALSA account has been approved. You can now sign in using "
        "the username and password you chose when requesting access.\n\n"
        f"Open SALSA: {public_url}\n"
        f"Username: {approved_user['username']}\n"
    )
    return _send_email(message)


def send_run_notification_email(
    email: str,
    *,
    display_name: str = "",
    run_id: str,
    workspace: str,
    state: str,
    processed_count: int | None = None,
    failure_count: int | None = None,
    error: str = "",
    run_dir: str = "",
) -> bool:
    state_label = (state or "finished").strip().lower()
    subject_state = {
        "done": "completed",
        "error": "failed",
        "cancelled": "was cancelled",
    }.get(state_label, state_label or "finished")
    message = _new_email(
        subject=f"SALSA run {subject_state}: {run_id}",
        recipients=[email],
    )
    if message is None:
        return False

    public_url = os.environ.get("SALSA_PUBLIC_URL", "").strip().rstrip("/")
    greeting = f"Hello {display_name}," if display_name else "Hello,"
    lines = [
        greeting,
        "",
        f"Your SALSA run has {subject_state}.",
        "",
        f"Run ID: {run_id}",
        f"Workspace: {workspace}",
        f"Status: {subject_state}",
    ]
    if processed_count is not None:
        lines.append(f"Processed DOIs/items: {processed_count}")
    if failure_count is not None:
        lines.append(f"Failed DOIs/items: {failure_count}")
    if error:
        lines.extend(["", "Error:", error])
    if public_url:
        lines.extend(["", f"Open SALSA: {public_url}"])
    if run_dir:
        lines.append(f"Run directory: {run_dir}")
    message.set_content("\n".join(lines) + "\n")
    return _send_email(message)
