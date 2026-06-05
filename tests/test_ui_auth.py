import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "ui"))

from auth import (  # noqa: E402
    AuthStore,
    build_reset_url,
    run_belongs_to_user,
    safe_child_path,
    user_workspaces_root,
    validate_email,
    write_run_manifest,
)


class TestUiAuth(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
