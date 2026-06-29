import sys
import smtplib
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ui"))

from auth import (  # noqa: E402
    AuthStore,
    build_reset_url,
    run_belongs_to_user,
    safe_child_path,
    send_account_approved_email,
    send_account_request_email,
    send_password_reset_email,
    user_workspaces_root,
    validate_email,
    write_run_manifest,
)


class TestUiAuth(unittest.TestCase):
    def test_existing_users_migrate_as_approved(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "auth.sqlite3"
            with sqlite3.connect(db_path) as conn:
                conn.executescript(
                    """
                    CREATE TABLE users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL UNIQUE,
                        email TEXT NOT NULL UNIQUE,
                        display_name TEXT NOT NULL DEFAULT '',
                        profile_slug TEXT NOT NULL UNIQUE,
                        password_hash TEXT NOT NULL,
                        role TEXT NOT NULL DEFAULT 'user',
                        is_active INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        last_login_at TEXT
                    );
                    INSERT INTO users (
                        username, email, display_name, profile_slug,
                        password_hash, role, created_at, updated_at
                    )
                    VALUES (
                        'legacy', 'legacy@example.edu', 'Legacy User', 'legacy',
                        'not-used', 'user', '2026-01-01T00:00:00Z',
                        '2026-01-01T00:00:00Z'
                    );
                    """
                )

            store = AuthStore(db_path)
            users = store.list_users()
            self.assertEqual(len(users), 1)
            self.assertEqual(users[0]["account_status"], "approved")

    def test_login_and_single_use_password_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AuthStore(Path(tmp) / "auth.sqlite3")
            user = store.create_user(
                username="william",
                email="william@example.com",
                first_name="William",
                last_name="Schertzer",
                institution="Georgia Tech",
                password="old-password",
            )

            self.assertIsNotNone(store.authenticate("william", "old-password"))
            self.assertIsNone(store.authenticate("william", "wrong-password"))

            token = store.create_password_reset_token(user["id"])
            self.assertTrue(
                store.reset_password_with_token(
                    identifier="william@example.com",
                    token=token,
                    new_password="new-password",
                )
            )
            self.assertFalse(
                store.reset_password_with_token(
                    identifier="william@example.com",
                    token=token,
                    new_password="another-password",
                )
            )
            self.assertIsNotNone(store.authenticate("william", "new-password"))
            self.assertIsNone(store.authenticate("william", "old-password"))

    def test_user_workspace_path_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            store = AuthStore(data_root / "auth.sqlite3")
            user = store.create_user(
                username="alice",
                email="alice@example.edu",
                first_name="Alice",
                last_name="Example",
                institution="Example University",
                password="correct-password",
            )
            workspaces_root = user_workspaces_root(data_root, user)
            run_dir = workspaces_root / "default" / "run_1"
            run_dir.mkdir(parents=True)
            write_run_manifest(
                run_dir,
                user_id=user["id"],
                username=user["username"],
                workspace="default",
                run_id="run_1",
            )

            self.assertEqual(safe_child_path(workspaces_root, "default"), workspaces_root / "default")
            with self.assertRaises(PermissionError):
                safe_child_path(workspaces_root, "..", "other-user")
            self.assertTrue(run_belongs_to_user(run_dir, workspaces_root, user))

    def test_bootstrap_william_admin_and_role_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AuthStore(Path(tmp) / "auth.sqlite3")
            first = store.create_user(
                username="alice",
                email="alice@example.edu",
                first_name="Alice",
                last_name="Example",
                institution="Example University",
                password="correct-password",
            )
            william = store.create_user(
                username="william",
                email="schertzer.william@gmail.com",
                first_name="William",
                last_name="Schertzer",
                institution="Georgia Tech",
                password="correct-password",
            )

            self.assertEqual(first["role"], "admin")
            self.assertEqual(william["role"], "user")

            promoted = store.bootstrap_default_admin()
            self.assertIsNotNone(promoted)
            self.assertEqual(promoted["username"], "william")
            self.assertEqual(promoted["role"], "admin")

            store.set_user_role(first["id"], "user")
            self.assertEqual(store.get_user(first["id"])["role"], "user")
            self.assertEqual(store.get_user(william["id"])["role"], "admin")

            with self.assertRaises(ValueError):
                store.set_user_role(william["id"], "user")

    def test_auth_session_token_can_restore_and_revoke_login(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AuthStore(Path(tmp) / "auth.sqlite3")
            user = store.create_user(
                username="alice",
                email="alice@example.edu",
                first_name="Alice",
                last_name="Example",
                institution="Example University",
                password="correct-password",
            )

            token = store.create_auth_session_token(user["id"])
            restored = store.user_from_auth_session_token(token)
            self.assertIsNotNone(restored)
            self.assertEqual(restored["id"], user["id"])

            store.revoke_auth_session_token(token)
            self.assertIsNone(store.user_from_auth_session_token(token))

    def test_admin_can_delete_user_without_deleting_workspace_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            store = AuthStore(data_root / "auth.sqlite3")
            admin = store.create_user(
                username="admin",
                email="admin@example.edu",
                first_name="Ada",
                last_name="Admin",
                institution="Example University",
                password="correct-password",
            )
            user = store.create_user(
                username="alice",
                email="alice@example.edu",
                first_name="Alice",
                last_name="Example",
                institution="Example University",
                password="correct-password",
            )
            session_token = store.create_auth_session_token(user["id"])
            workspace_root = user_workspaces_root(data_root, user)
            marker = workspace_root / "research-data.txt"
            marker.write_text("preserve me", encoding="utf-8")

            with self.assertRaises(PermissionError):
                store.delete_user(admin["id"], user["id"])
            with self.assertRaises(ValueError):
                store.delete_user(admin["id"], admin["id"])

            store.delete_user(user["id"], admin["id"])

            self.assertIsNone(store.find_user("alice"))
            self.assertIsNone(store.user_from_auth_session_token(session_token))
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve me")

            replacement = store.create_user(
                username="alice-replacement",
                email="alice@example.edu",
                first_name="Alice",
                last_name="Replacement",
                institution="Example University",
                password="replacement-password",
            )
            self.assertNotEqual(replacement["id"], user["id"])

    def test_rejected_request_releases_email_and_legacy_rejections_are_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AuthStore(Path(tmp) / "auth.sqlite3")
            admin = store.create_user(
                username="admin",
                email="admin@example.edu",
                first_name="Ada",
                last_name="Admin",
                institution="Example University",
                password="correct-password",
            )
            requested = store.request_account(
                username="alice",
                email="alice@example.edu",
                first_name="Alice",
                last_name="Applicant",
                institution="Example University",
                password="correct-password",
            )

            store.reject_account(requested["id"], admin["id"])
            replacement = store.request_account(
                username="alice-replacement",
                email="alice@example.edu",
                first_name="Alice",
                last_name="Replacement",
                institution="Example University",
                password="replacement-password",
            )
            self.assertNotEqual(replacement["id"], requested["id"])

            with store.connect() as conn:
                conn.execute(
                    "UPDATE users SET account_status = 'rejected' WHERE id = ?",
                    (replacement["id"],),
                )
            legacy_replacement = store.request_account(
                username="alice-legacy-replacement",
                email="alice@example.edu",
                first_name="Alice",
                last_name="Legacy Replacement",
                institution="Example University",
                password="replacement-password",
            )
            self.assertNotEqual(legacy_replacement["id"], replacement["id"])

    def test_signup_requires_profile_fields_and_edu_or_com_email(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AuthStore(Path(tmp) / "auth.sqlite3")

            with self.assertRaises(ValueError):
                validate_email("alice@example.org")

            with self.assertRaises(ValueError):
                store.create_user(
                    username="alice",
                    email="alice@example.edu",
                    first_name="Alice",
                    last_name="",
                    institution="Example University",
                    password="correct-password",
                )

            user = store.create_user(
                username="alice",
                email="alice@example.edu",
                first_name="Alice",
                last_name="Example",
                institution="Example University",
                password="correct-password",
            )
            self.assertEqual(user["display_name"], "Alice Example")
            self.assertEqual(user["institution"], "Example University")

    def test_reset_url_prefills_token_and_identifier(self):
        url = build_reset_url("https://salsa.example.com", "token value", "Alice@Example.edu")
        self.assertIn("reset_token=token+value", url)
        self.assertIn("reset_identifier=Alice%40Example.edu", url)

    def test_password_reset_email_uses_configured_relay_hostname(self):
        env = {
            "SALSA_SMTP_HOST": "outbound.mail.gatech.edu",
            "SALSA_SMTP_PORT": "25",
            "SALSA_SMTP_LOCAL_HOSTNAME": "polymergenome.storm.gatech.edu",
            "SALSA_SMTP_FROM": "salsa@gatech.edu",
            "SALSA_SMTP_TLS": "0",
        }
        with patch.dict("os.environ", env, clear=True), patch("auth.smtplib.SMTP") as smtp_cls:
            sent = send_password_reset_email(
                "alice@example.edu",
                "secret-token",
            )

        self.assertTrue(sent)
        smtp_cls.assert_called_once_with(
            "outbound.mail.gatech.edu",
            25,
            local_hostname="polymergenome.storm.gatech.edu",
            timeout=10,
        )
        smtp = smtp_cls.return_value.__enter__.return_value
        smtp.starttls.assert_not_called()
        smtp.login.assert_not_called()
        message = smtp.send_message.call_args.args[0]
        self.assertEqual(message["From"], "salsa@gatech.edu")
        self.assertEqual(message["To"], "alice@example.edu")
        self.assertIsNotNone(message["Date"])
        self.assertTrue(message["Message-ID"].endswith("@polymergenome.storm.gatech.edu>"))
        self.assertIn("Reset token: secret-token", message.get_content())
        self.assertNotIn("http", message.get_content())

    def test_account_request_requires_admin_approval_and_notifies_admins(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AuthStore(Path(tmp) / "auth.sqlite3")
            admin = store.create_user(
                username="admin",
                email="admin@example.edu",
                first_name="Ada",
                last_name="Admin",
                institution="Example University",
                password="correct-password",
            )
            requested = store.request_account(
                username="alice",
                email="alice@example.edu",
                first_name="Alice",
                last_name="Applicant",
                institution="Applicant University",
                password="correct-password",
            )

            self.assertEqual(requested["account_status"], "pending")
            self.assertIsNone(store.authenticate("alice", "correct-password"))
            self.assertEqual(
                [user["username"] for user in store.list_pending_users()],
                ["alice"],
            )

            with self.assertRaises(PermissionError):
                store.approve_account(requested["id"], requested["id"])

            approved = store.approve_account(requested["id"], admin["id"])
            self.assertEqual(approved["account_status"], "approved")
            self.assertIsNotNone(store.authenticate("alice", "correct-password"))
            self.assertEqual(store.list_pending_users(), [])

        env = {
            "SALSA_SMTP_HOST": "outbound.mail.gatech.edu",
            "SALSA_SMTP_PORT": "25",
            "SALSA_SMTP_LOCAL_HOSTNAME": "polymergenome.storm.gatech.edu",
            "SALSA_SMTP_FROM": "salsa@gatech.edu",
            "SALSA_SMTP_TLS": "0",
        }
        with patch.dict("os.environ", env, clear=True), patch("auth.smtplib.SMTP") as smtp_cls:
            sent = send_account_request_email(["admin@example.edu"], requested)

        self.assertTrue(sent)
        message = smtp_cls.return_value.__enter__.return_value.send_message.call_args.args[0]
        body = message.get_content()
        self.assertEqual(message["To"], "admin@example.edu")
        self.assertIn("Alice Applicant", body)
        self.assertIn("alice@example.edu", body)
        self.assertIn("Applicant University", body)
        self.assertNotIn("correct-password", body)

    def test_account_request_email_surfaces_relay_data_rejection(self):
        requested = {
            "username": "alice",
            "first_name": "Alice",
            "last_name": "Applicant",
            "email": "alice@example.edu",
            "institution": "Applicant University",
            "created_at": "2026-06-09T17:00:00Z",
        }
        env = {
            "SALSA_SMTP_HOST": "outbound.mail.gatech.edu",
            "SALSA_SMTP_PORT": "25",
            "SALSA_SMTP_LOCAL_HOSTNAME": "polymergenome.storm.gatech.edu",
            "SALSA_SMTP_FROM": "salsa@gatech.edu",
            "SALSA_SMTP_TLS": "0",
        }
        with patch.dict("os.environ", env, clear=True), patch(
            "auth.smtplib.SMTP"
        ) as smtp_cls:
            smtp = smtp_cls.return_value.__enter__.return_value
            smtp.send_message.side_effect = smtplib.SMTPDataError(
                550,
                b"#5.7.1 Your access to submit messages has been rejected.",
            )
            with self.assertRaises(smtplib.SMTPDataError):
                send_account_request_email(["admin@example.edu"], requested)

    def test_account_approval_email_links_to_salsa(self):
        approved_user = {
            "username": "alice",
            "first_name": "Alice",
            "email": "alice@example.edu",
        }
        env = {
            "SALSA_PUBLIC_URL": "https://salsa.storm.gatech.edu/",
            "SALSA_SMTP_HOST": "outbound.mail.gatech.edu",
            "SALSA_SMTP_PORT": "25",
            "SALSA_SMTP_LOCAL_HOSTNAME": "polymergenome.storm.gatech.edu",
            "SALSA_SMTP_FROM": "salsa@gatech.edu",
            "SALSA_SMTP_TLS": "0",
        }
        with patch.dict("os.environ", env, clear=True), patch(
            "auth.smtplib.SMTP"
        ) as smtp_cls:
            sent = send_account_approved_email(approved_user)

        self.assertTrue(sent)
        message = smtp_cls.return_value.__enter__.return_value.send_message.call_args.args[0]
        body = message.get_content()
        self.assertEqual(message["Subject"], "Your SALSA account has been approved")
        self.assertEqual(message["To"], "alice@example.edu")
        self.assertIn("https://salsa.storm.gatech.edu", body)
        self.assertIn("Username: alice", body)
        self.assertNotIn("password:", body.lower())


if __name__ == "__main__":
    unittest.main()
