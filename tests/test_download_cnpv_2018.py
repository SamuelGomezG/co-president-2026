"""Tests for the CNPV 2018 microdata downloader script.

Verifies manifest parsing, URL construction, and downloader behavior
against a captured snapshot of the DANE catalog page. Network behavior
is exercised with mocked HTTP responses — no real DANE calls during tests.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest
import requests

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "download_cnpv_2018.py"
_spec = importlib.util.spec_from_file_location("cnpv_downloader_under_test", _SCRIPT_PATH)
assert _spec is not None
download_cnpv = importlib.util.module_from_spec(_spec)
sys.modules["cnpv_downloader_under_test"] = download_cnpv
_spec.loader.exec_module(download_cnpv)  # type: ignore[union-attr]

CnpvFile = download_cnpv.CnpvFile
MANIFEST: tuple[CnpvFile, ...] = download_cnpv.MANIFEST
CATALOG_URL = download_cnpv.CATALOG_URL
DDI_XML_URL = download_cnpv.DDI_XML_URL
PDF_DOC_URL = download_cnpv.PDF_DOC_URL
DOC_FILES = download_cnpv.DOC_FILES
parse_manifest = download_cnpv.parse_manifest
build_url = download_cnpv.build_url
download_file = download_cnpv.download_file
DownloadError = download_cnpv.DownloadError


@pytest.fixture(scope="module")
def catalog_html() -> str:
    """Load the captured CNPV catalog HTML used by parse_manifest tests."""
    fixture = Path(__file__).resolve().parent / "fixtures" / "cnpv_catalog.html"
    return fixture.read_text(encoding="utf-8")


@pytest.fixture
def mock_session() -> MagicMock:
    """Provide a mocked ``requests.Session`` for HTTP-isolated tests."""
    return MagicMock(spec=requests.Session)


class TestCnpvFile:
    """Tests for the CnpvFile frozen dataclass."""

    def test_instantiation(self) -> None:
        """Verify CnpvFile creates with all fields and derives url."""
        entry = CnpvFile(file_id=12143, name="05_Antioquia.zip")
        assert entry.file_id == 12143
        assert entry.name == "05_Antioquia.zip"
        assert entry.url == "https://microdatos.dane.gov.co/index.php/catalog/643/download/12143"

    def test_immutability(self) -> None:
        """Verify the dataclass rejects attribute assignment."""
        entry = CnpvFile(file_id=12143, name="05_Antioquia.zip")
        with pytest.raises(AttributeError):
            entry.file_id = 99999  # type: ignore[misc]

    def test_url_uses_build_url(self) -> None:
        """Verify the url field always matches build_url(file_id)."""
        entry = CnpvFile(file_id=12175, name="99_Vichada.zip")
        assert entry.url == build_url(12175)


class TestBuildUrl:
    """Tests for build_url()."""

    def test_minimum_id(self) -> None:
        """Verify the smallest catalog ID maps to the expected URL."""
        assert (
            build_url(12143)
            == "https://microdatos.dane.gov.co/index.php/catalog/643/download/12143"
        )

    def test_maximum_id(self) -> None:
        """Verify the largest catalog ID maps to the expected URL."""
        assert (
            build_url(12175)
            == "https://microdatos.dane.gov.co/index.php/catalog/643/download/12175"
        )


class TestParseManifest:
    """Tests for parse_manifest() against the captured catalog page."""

    EXPECTED_COUNT: ClassVar[int] = 33
    FIRST_NAME: ClassVar[str] = "05_Antioquia.zip"
    LAST_NAME: ClassVar[str] = "99_Vichada.zip"

    def test_returns_correct_count(self, catalog_html: str) -> None:
        """Verify the parser extracts all 33 departmental file entries."""
        result = parse_manifest(catalog_html)
        assert len(result) == self.EXPECTED_COUNT

    def test_first_entry_is_antioquia(self, catalog_html: str) -> None:
        """Verify the first entry is Antioquia with id 12143."""
        result = parse_manifest(catalog_html)
        assert result[0].name == self.FIRST_NAME
        assert result[0].file_id == 12143

    def test_last_entry_is_vichada(self, catalog_html: str) -> None:
        """Verify the last entry is Vichada with id 12175."""
        result = parse_manifest(catalog_html)
        assert result[-1].name == self.LAST_NAME
        assert result[-1].file_id == 12175

    def test_all_entries_have_valid_urls(self, catalog_html: str) -> None:
        """Verify every parsed entry has a download URL that matches its id."""
        for entry in parse_manifest(catalog_html):
            assert entry.url == build_url(entry.file_id)
            assert entry.name.endswith(".zip")

    def test_ids_are_unique(self, catalog_html: str) -> None:
        """Verify no two entries share the same file_id."""
        ids = [e.file_id for e in parse_manifest(catalog_html)]
        assert len(set(ids)) == len(ids)

    def test_names_are_unique(self, catalog_html: str) -> None:
        """Verify no two entries share the same filename."""
        names = [e.name for e in parse_manifest(catalog_html)]
        assert len(set(names)) == len(names)

    def test_ids_are_sequential(self, catalog_html: str) -> None:
        """Verify the 33 ids form a contiguous integer range starting at 12143."""
        ids = sorted(e.file_id for e in parse_manifest(catalog_html))
        assert ids == list(range(12143, 12143 + 33))

    def test_empty_html_returns_empty_list(self) -> None:
        """Verify an empty HTML string returns no entries."""
        assert parse_manifest("") == []

    def test_html_without_download_links_returns_empty_list(self) -> None:
        """Verify HTML with no download links returns no entries."""
        html = "<html><body><p>no files here</p></body></html>"
        assert parse_manifest(html) == []


class TestManifestConstant:
    """Tests for the hardcoded MANIFEST constant."""

    def test_has_33_entries(self) -> None:
        """Verify the hardcoded manifest contains all 33 entries."""
        assert len(MANIFEST) == 33

    def test_matches_parsed(self, catalog_html: str) -> None:
        """Verify the hardcoded MANIFEST exactly matches parse_manifest() output."""
        assert list(MANIFEST) == parse_manifest(catalog_html)

    def test_all_names_cover_departments(self) -> None:
        """Verify the manifest covers each of the 33 Colombian departments."""
        names = {e.name for e in MANIFEST}
        assert "05_Antioquia.zip" in names
        assert "11_Bogota.zip" in names
        assert "25_Cundinamarca.zip" in names
        assert "76_ValleDelCauca.zip" in names
        assert "99_Vichada.zip" in names


class TestDocFiles:
    """Tests for the DOC_FILES (data dictionary + PDF) constants."""

    def test_ddi_xml_url(self) -> None:
        """Verify the DDI/XML metadata URL is well-formed."""
        assert "metadata/export" in DDI_XML_URL
        assert DDI_XML_URL.endswith("/ddi")

    def test_pdf_doc_url(self) -> None:
        """Verify the PDF documentation URL is well-formed."""
        assert "pdf-documentation" in PDF_DOC_URL
        assert "catalog/643" in PDF_DOC_URL

    def test_doc_files_iterable(self) -> None:
        """Verify DOC_FILES is a non-empty sequence of (url, filename) pairs."""
        assert len(DOC_FILES) >= 2
        for url, filename in DOC_FILES:
            assert url.startswith("https://")
            assert filename.endswith((".pdf", ".xml", ".ddi"))


class TestDownloadFile:
    """Tests for download_file() with mocked HTTP."""

    def _make_response(
        self, payload: bytes, *, status_code: int = 200, headers: dict[str, str] | None = None
    ) -> MagicMock:
        """Build a mock requests.Response that yields payload chunks on iter_content."""
        response = MagicMock()
        response.status_code = status_code
        response.headers = headers or {}
        response.url = "https://example.com/test"
        response.iter_content.return_value = [payload]
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        return response

    def test_downloads_payload_to_disk(self, tmp_path: Path, mock_session: MagicMock) -> None:
        """Verify a successful download writes payload bytes to the destination path."""
        entry = MANIFEST[0]
        payload = b"PK\x03\x04fake-zip-content"
        head_response = MagicMock()
        head_response.headers = {"Content-Length": str(len(payload))}
        mock_session.head.return_value = head_response
        mock_session.get.return_value = self._make_response(payload)

        result = download_file(mock_session, entry, tmp_path)

        assert result == tmp_path / entry.name
        assert result.read_bytes() == payload
        mock_session.get.assert_called_once()

    def test_skips_existing_file_with_correct_size(
        self, tmp_path: Path, mock_session: MagicMock
    ) -> None:
        """Verify a file already at the expected size is not re-downloaded."""
        entry = MANIFEST[0]
        target = tmp_path / entry.name
        payload = b"complete-content"
        target.write_bytes(payload)

        head_response = MagicMock()
        head_response.headers = {"Content-Length": str(len(payload))}
        mock_session.head.return_value = head_response

        result = download_file(mock_session, entry, tmp_path, skip_existing=True)

        assert result == target
        assert result.read_bytes() == payload
        mock_session.get.assert_not_called()
        mock_session.head.assert_called_once()

    def test_redownloads_when_size_mismatches(
        self, tmp_path: Path, mock_session: MagicMock
    ) -> None:
        """Verify a partial/incorrectly-sized file is overwritten."""
        entry = MANIFEST[0]
        target = tmp_path / entry.name
        target.write_bytes(b"partial")
        payload = b"PK\x03\x04new-full-content"
        head_response = MagicMock()
        head_response.headers = {"Content-Length": str(len(payload))}
        mock_session.head.return_value = head_response
        mock_session.get.return_value = self._make_response(payload)

        result = download_file(mock_session, entry, tmp_path, skip_existing=True)

        assert result.read_bytes() == payload
        mock_session.get.assert_called_once()

    def test_raises_on_http_error(self, tmp_path: Path, mock_session: MagicMock) -> None:
        """Verify a 4xx/5xx response raises DownloadError."""
        entry = MANIFEST[0]
        bad = self._make_response(b"", status_code=404)
        mock_session.get.return_value = bad

        with pytest.raises(DownloadError, match="404"):
            download_file(mock_session, entry, tmp_path)

    def test_creates_dest_dir(self, tmp_path: Path, mock_session: MagicMock) -> None:
        """Verify a non-existent destination directory is created."""
        entry = MANIFEST[0]
        payload = b"x"
        head_response = MagicMock()
        head_response.headers = {"Content-Length": str(len(payload))}
        mock_session.head.return_value = head_response
        mock_session.get.return_value = self._make_response(payload)

        nested = tmp_path / "deep" / "nested" / "dir"
        download_file(mock_session, entry, nested)

        assert (nested / entry.name).exists()

    def test_retries_on_transient_error(self, tmp_path: Path, mock_session: MagicMock) -> None:
        """Verify a transient ConnectionError is retried until success."""
        entry = MANIFEST[0]
        payload = b"recovered"
        head_response = MagicMock()
        head_response.headers = {"Content-Length": str(len(payload))}
        mock_session.head.return_value = head_response

        call_count = {"n": 0}

        def fake_get(*_args: object, **_kwargs: object) -> MagicMock:
            call_count["n"] += 1
            if call_count["n"] < 3:
                msg = "connection reset"
                raise requests.exceptions.ConnectionError(msg)
            return self._make_response(payload)

        mock_session.get.side_effect = fake_get

        with patch("time.sleep"):
            result = download_file(mock_session, entry, tmp_path)

        assert call_count["n"] == 3
        assert result.read_bytes() == payload

    def test_returns_dest_path(self, tmp_path: Path, mock_session: MagicMock) -> None:
        """Verify the returned path equals the destination of the written file."""
        entry = MANIFEST[0]
        payload = b"abc"
        head_response = MagicMock()
        head_response.headers = {"Content-Length": str(len(payload))}
        mock_session.head.return_value = head_response
        mock_session.get.return_value = self._make_response(payload)

        with patch("time.sleep"):
            result = download_file(mock_session, entry, tmp_path)

        assert isinstance(result, Path)
        assert result.name == entry.name
