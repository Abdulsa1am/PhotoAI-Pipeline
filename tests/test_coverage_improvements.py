import importlib.util
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from unittest import mock


# Repository root is used to dynamically import numbered script files for testing.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_script_module(module_name, script_filename, stubs=None):
    script_path = os.path.join(REPO_ROOT, script_filename)
    with mock.patch.dict(sys.modules, stubs or {}, clear=False):
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class TestBuildArchiveUtilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numpy_stub = types.ModuleType("numpy")
        cls.mod = load_script_module(
            "build_archive_module",
            "4_build_archive.py",
            stubs={"numpy": numpy_stub},
        )

    def setUp(self):
        self._original_placed = set(self.mod.placed_source_paths)

    def tearDown(self):
        self.mod.placed_source_paths.clear()
        self.mod.placed_source_paths.update(self._original_placed)

    def test_get_next_archive_dir_increments_highest_existing_number(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(os.path.join(temp_dir, "Archive_1"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "Archive_7"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "OtherFolder"), exist_ok=True)

            old_base = self.mod.BASE_PHOTOS_DIR
            self.mod.BASE_PHOTOS_DIR = temp_dir
            try:
                nxt = self.mod.get_next_archive_dir(0.4, 0.15)
            finally:
                self.mod.BASE_PHOTOS_DIR = old_base

            self.assertTrue(nxt.endswith("Archive_8_merge0.4_conf0.15"))

    def test_place_file_copy_mode_places_and_tracks_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            src = os.path.join(temp_dir, "src.txt")
            dst = os.path.join(temp_dir, "nested", "dst.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("hello")

            self.mod.placed_source_paths.clear()
            ok = self.mod.place_file(src, dst, mode="copy")

            self.assertTrue(ok)
            self.assertTrue(os.path.exists(dst))
            self.assertIn(
                os.path.normcase(os.path.abspath(src)),
                self.mod.placed_source_paths,
            )

    def test_place_file_auto_falls_back_to_copy_when_link_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            src = os.path.join(temp_dir, "src.txt")
            dst = os.path.join(temp_dir, "dst.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("data")

            with (
                mock.patch.object(self.mod.os, "link", side_effect=OSError),
                mock.patch.object(self.mod.shutil, "copy2", wraps=self.mod.shutil.copy2) as copy2_mock,
            ):
                ok = self.mod.place_file(src, dst, mode="auto")

            self.assertTrue(ok)
            self.assertTrue(os.path.exists(dst))
            self.assertGreaterEqual(copy2_mock.call_count, 1)

    def test_load_cluster_labels_maps_unknown_custom_and_generated_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("CREATE TABLE images (id INTEGER PRIMARY KEY, file_path TEXT)")
            c.execute(
                "CREATE TABLE people (id INTEGER PRIMARY KEY, custom_name TEXT, centroid BLOB)"
            )
            c.execute(
                "CREATE TABLE faces (id INTEGER PRIMARY KEY, image_id INTEGER, person_id INTEGER)"
            )

            c.executemany(
                "INSERT INTO images (id, file_path) VALUES (?, ?)",
                [
                    (1, "/photos/a.jpg"),
                    (2, "/photos/b.jpg"),
                    (3, "/photos/c.jpg"),
                ],
            )
            c.executemany(
                "INSERT INTO people (id, custom_name, centroid) VALUES (?, ?, ?)",
                [
                    (1, "Alice", b"x"),
                    (2, "", b"y"),
                ],
            )
            c.executemany(
                "INSERT INTO faces (id, image_id, person_id) VALUES (?, ?, ?)",
                [
                    (1, 1, 1),
                    (2, 2, 2),
                    (3, 1, 2),
                    (4, 3, -1),
                ],
            )
            conn.commit()
            conn.close()

            image_persons, image_paths = self.mod.load_cluster_labels(db_path)

            self.assertIn("Alice", image_persons[1])
            self.assertIn("Person_002_2faces", image_persons[1])
            self.assertIn("Person_002_2faces", image_persons[2])
            self.assertIn("Unknown_Faces", image_persons[3])
            self.assertEqual(image_paths[2], "/photos/b.jpg")


class TestCompressUtilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pil_stub = types.ModuleType("PIL")
        pil_stub.Image = object()
        cls.mod = load_script_module(
            "compress_module",
            "5_compress_images.py",
            stubs={"PIL": pil_stub},
        )

    def test_find_images_recursively_filters_supported_extensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(os.path.join(temp_dir, "sub"), exist_ok=True)
            keep1 = os.path.join(temp_dir, "a.jpg")
            keep2 = os.path.join(temp_dir, "sub", "b.PNG")
            skip = os.path.join(temp_dir, "sub", "note.txt")

            for path in (keep1, keep2, skip):
                with open(path, "wb") as f:
                    f.write(b"x")

            found = self.mod.find_images(temp_dir)
            self.assertEqual(found, sorted([keep1, keep2]))

    def test_get_exif_data_returns_exif_or_none(self):
        class WithExif:
            info = {"exif": b"abc"}

        class BadInfo:
            @property
            def info(self):
                raise RuntimeError("boom")

        self.assertEqual(self.mod.get_exif_data(WithExif()), b"abc")
        self.assertIsNone(self.mod.get_exif_data(BadInfo()))


if __name__ == "__main__":
    unittest.main()
