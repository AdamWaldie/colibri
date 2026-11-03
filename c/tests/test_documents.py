"""Documents in API requests (documents.py): every format read to text, every API
shape rewritten, the bounds that keep a hostile file from costing the server, and
the gateway end to end (a PDF sent through /v1/chat/completions and /v1/messages
reaches the model as its text).

The files are built here, byte for byte: a PDF with compressed content, a
ToUnicode map and its page objects inside an object stream; Office and
OpenDocument archives with the XML their applications write."""
import base64
import io
import json
import sys
import unittest
import zipfile
import zlib
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import documents  # noqa: E402


# ---------------------------------------------------------------- building files


def pdf_bytes(pages=("Hello", "World"), text=True):
    """A PDF whose first page uses a Type0 font with a ToUnicode map (2-byte codes)
    and whose second uses a simple WinAnsi font with escapes; page dicts live in an
    object stream, content streams are FlateDecoded."""
    objs = {}
    cmap = (b"/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
            b"1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
            b"3 beginbfchar\n<0001> <0048>\n<0002> <00E9>\n<0003> <0065>\nendbfchar\n"
            b"1 beginbfrange\n<0010> <0013> <006C>\nendbfrange\nendcmap\n")
    word = pages[0]
    codes = {"H": 1, "é": 2, "e": 3, "l": 0x10, "m": 0x11, "n": 0x12, "o": 0x13}
    hexed = "".join(f"{codes.get(ch, 0x10):04X}" for ch in word)
    content1 = f"BT /F1 12 Tf 72 720 Td <{hexed}> Tj ET\nBT /F1 12 Tf 72 700 Td [<0001> -300 <0002>] TJ ET".encode()
    content2 = (b"BT /F2 12 Tf 72 720 Td (" + pages[1].encode("cp1252") + rb" \(x\) caf\351) Tj ET")
    if text is False:          # a scan: pictures, no text operators at all
        content1 = content2 = b"q 600 0 0 800 0 0 cm /Im1 Do Q"
    objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objs[5] = b"<< /Type /Font /Subtype /Type0 /BaseFont /X /Encoding /Identity-H /ToUnicode 6 0 R >>"
    objs[7] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    streams = {6: zlib.compress(cmap), 8: zlib.compress(content1), 9: zlib.compress(content2)}
    in_objstm = {2: b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 /Resources << /Font << /F1 5 0 R /F2 7 0 R >> >> >>",
                 3: b"<< /Type /Page /Parent 2 0 R /Contents 8 0 R >>",
                 4: b"<< /Type /Page /Parent 2 0 R /Contents [9 0 R] >>"}
    header, body = b"", b""
    for num, obj in in_objstm.items():
        header += f"{num} {len(body)} ".encode()
        body += obj + b"\n"
    objstm = header + body
    out = io.BytesIO()
    out.write(b"%PDF-1.7\n")
    for num, obj in objs.items():
        out.write(f"{num} 0 obj\n".encode() + obj + b"\nendobj\n")
    for num, data in streams.items():
        out.write(f"{num} 0 obj\n<< /Length {len(data)} /Filter /FlateDecode >>\nstream\n".encode()
                  + data + b"\nendstream\nendobj\n")
    packed = zlib.compress(objstm)
    out.write(f"10 0 obj\n<< /Type /ObjStm /N 3 /First {len(header)} /Length {len(packed)} "
              f"/Filter /FlateDecode >>\nstream\n".encode() + packed + b"\nendstream\nendobj\n")
    out.write(b"trailer\n<< /Root 1 0 R >>\n%%EOF\n")
    return out.getvalue()


def zip_bytes(members):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return out.getvalue()


W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'


def docx():
    xml = (f'<w:document {W}><w:body><w:p><w:r><w:t>Title</w:t></w:r></w:p>'
           '<w:p><w:r><w:t>Hello</w:t><w:tab/><w:t>world</w:t></w:r></w:p>'
           '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>a</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>b</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
           '</w:body></w:document>')
    return zip_bytes({"[Content_Types].xml": "<Types/>", "word/document.xml": xml})


def pptx():
    slide = lambda text: f'<p:sld xmlns:p="p" {A}><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:sld>'  # noqa: E731
    return zip_bytes({"ppt/presentation.xml": "<p/>", "ppt/slides/slide2.xml": slide("Second"),
                      "ppt/slides/slide1.xml": slide("First"), "ppt/slides/slide10.xml": slide("Tenth")})


def xlsx():
    return zip_bytes({
        "xl/workbook.xml": f'<workbook {S}><sheets><sheet name="Sales" sheetId="1"/></sheets></workbook>',
        "xl/sharedStrings.xml": f'<sst {S}><si><t>Region</t></si><si><t>North, East</t></si></sst>',
        "xl/worksheets/sheet1.xml": (f'<worksheet {S}><sheetData><row><c t="s"><v>0</v></c><c><v>42</v></c></row>'
                                     '<row><c t="s"><v>1</v></c><c t="inlineStr"><is><t>ok</t></is></c></row>'
                                     '</sheetData></worksheet>')})


def odt():
    ns = 'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"'
    return zip_bytes({"mimetype": "application/vnd.oasis.opendocument.text",
                      "content.xml": f'<office:document-content {ns}><office:body><office:text>'
                                     '<text:h>Heading</text:h><text:p>One<text:s text:c="2"/>two</text:p>'
                                     '</office:text></office:body></office:document-content>'})


def epub():
    return zip_bytes({
        "META-INF/container.xml": '<container><rootfiles><rootfile full-path="OEBPS/c.opf"/></rootfiles></container>',
        "OEBPS/c.opf": '<package><manifest><item id="a" href="b.xhtml"/><item id="b" href="a.xhtml"/></manifest>'
                       '<spine><itemref idref="b"/><itemref idref="a"/></spine></package>',
        "OEBPS/a.xhtml": "<html><body><p>Chapter one</p></body></html>",
        "OEBPS/b.xhtml": "<html><body><p>Chapter two</p></body></html>"})


def data_url(mime, data):
    return f"data:{mime};base64," + base64.b64encode(data).decode()


# ---------------------------------------------------------------- formats


class Formats(unittest.TestCase):
    def text(self, name, data, mime=None):
        return documents.extract(name, mime, data)[0]

    def test_pdf_with_the_built_in_reader(self):
        # MiniPDF itself: pypdf, when installed, would answer extract() first
        pages = documents.MiniPDF(pdf_bytes(("Hello", "World"))).pages_text()
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0], "Hello\nH é")          # ToUnicode map; TJ's wide gap is a space
        self.assertEqual(pages[1], "World (x) café")      # escapes and WinAnsi octal

    def test_pdf_through_extract(self):
        text, attrs = documents.extract("a.pdf", "application/pdf", pdf_bytes(("Hello", "World")))
        self.assertEqual(attrs, {"pages": "2"})
        self.assertIn("Hello", text)
        self.assertIn("World", text)

    def test_scanned_pdf_says_so(self):
        with mock.patch.dict(sys.modules, {"pypdf": None}):           # the built-in reader
            text = self.text("scan.pdf", pdf_bytes(text=False))
        self.assertIn("scanned", text)

    def test_encrypted_pdf_refused(self):
        with self.assertRaisesRegex(documents.DocumentError, "encrypted"):
            documents.extract("e.pdf", None, b"%PDF-1.4\n1 0 obj << /Encrypt 2 0 R >> endobj")

    def test_office_and_open_formats(self):
        self.assertEqual(self.text("a.docx", docx()), "Title\nHello\tworld\na | b")
        slides, attrs = documents.extract("a.pptx", None, pptx())
        self.assertEqual(attrs, {"slides": "3"})
        self.assertLess(slides.index("First"), slides.index("Second"))
        self.assertLess(slides.index("Second"), slides.index("Tenth"))     # numeric, not 1,10,2
        self.assertEqual(self.text("a.xlsx", xlsx()), '[sheet Sales]\nRegion,42\n"North, East",ok')
        self.assertEqual(self.text("a.odt", odt()), "Heading\nOne  two")
        self.assertEqual(self.text("a.epub", epub()), "Chapter one\n\nChapter two")    # spine order

    def test_web_rtf_and_text(self):
        page = b"<html><head><style>x{}</style><script>bad()</script></head><body><h1>T</h1><p>a &amp; b</p></body></html>"
        self.assertEqual(self.text("p.html", page), "T\n\na & b")
        rtf = (rb"{\rtf1\ansi{\fonttbl{\f0 Arial;}}\f0 Hello\par caf\'e9 " + b"\\u8364?}")
        self.assertEqual(self.text("d.rtf", rtf), "Hello\ncafé €?")
        self.assertEqual(self.text("n.md", "# héllo".encode()), "# héllo")
        self.assertEqual(self.text("w.txt", "x".encode("utf-16")), "x")
        self.assertEqual(self.text("legacy.csv", "naïve".encode("cp1252")), "naïve")

    def test_refusals(self):
        for name, data, why in (("x.bin", b"\0\1\2" * 100, "binary"),
                                ("old.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 64, ".docx"),
                                ("a.zip", zip_bytes({"x.txt": "y"}), "zip")):
            with self.subTest(name=name), self.assertRaisesRegex(documents.DocumentError, why):
                documents.extract(name, None, data)


class Bounds(unittest.TestCase):
    def test_a_pdf_stream_bomb_is_refused(self):
        bomb = zlib.compress(b"\0" * (documents.MAX_INFLATED + (1 << 20)), 9)
        pdf = (b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
               b"2 0 obj << /Type /Pages /Kids [3 0 R] >> endobj\n3 0 obj << /Type /Page /Contents 4 0 R >> endobj\n"
               + f"4 0 obj << /Length {len(bomb)} /Filter /FlateDecode >>\nstream\n".encode() + bomb
               + b"\nendstream endobj\n")
        with self.assertRaisesRegex(documents.DocumentError, "expands to more than"):
            documents.extract("bomb.pdf", None, pdf)

    def test_an_archive_bomb_is_refused(self):
        big = zip_bytes({"word/document.xml": "<a>" + "x" * (documents.MAX_INFLATED + 10) + "</a>"})
        with self.assertRaisesRegex(documents.DocumentError, "too large"):
            documents.extract("b.docx", None, big)

    def test_long_text_is_cut_and_control_characters_go(self):
        part = documents._as_text_part("t.txt", "text/plain", ("a\x07b\r\n" + "z" * (documents.MAX_CHARS + 5)).encode())
        self.assertIn("ab\nzz", part["text"])
        self.assertIn("was left out", part["text"])
        self.assertNotIn("\x07", part["text"])

    def test_a_parser_failure_is_a_document_error(self):
        broken = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R"   # cut mid-object
        try:
            documents.expand_part({"type": "file", "file": {"filename": "b.pdf",
                                                            "file_data": data_url("application/pdf", broken)}})
        except documents.DocumentError:
            pass                                    # a clean refusal is fine too


# ---------------------------------------------------------------- the API shapes


class ApiShapes(unittest.TestCase):
    def test_openai_file_part(self):
        messages = [{"role": "user", "content": [
            {"type": "file", "file": {"filename": "r.docx", "file_data": data_url("application/x", docx())}},
            {"type": "text", "text": "Summarise"}]}]
        self.assertEqual(documents.expand_messages(messages), 1)
        content = messages[0]["content"]
        self.assertIsInstance(content, str)              # text only now: one string, as before
        self.assertTrue(content.startswith('<document name="r.docx">\nTitle'))
        self.assertTrue(content.endswith("</document>\n\nSummarise"))

    def test_images_keep_the_array_form(self):
        messages = [{"role": "user", "content": [
            {"type": "file", "file": {"filename": "n.txt", "file_data": data_url("text/plain", b"note")}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]
        documents.expand_messages(messages)
        self.assertEqual([p["type"] for p in messages[0]["content"]], ["text", "image_url"])

    def test_responses_input_file(self):
        part = documents.expand_part({"type": "input_file", "filename": "a.md",
                                      "file_data": data_url("text/markdown", b"# A")})
        self.assertEqual(part["text"], '<document name="a.md">\n# A\n</document>')

    def test_anthropic_document_blocks(self):
        pdf = documents.expand_part({"type": "document", "title": "Q3 report", "context": "draft",
                                     "source": {"type": "base64", "media_type": "application/pdf",
                                                "data": base64.b64encode(pdf_bytes()).decode()}})
        self.assertTrue(pdf["text"].startswith('<document name="Q3 report" pages="2">\ndraft\n[page 1]'))
        plain = documents.expand_part({"type": "document", "source": {"type": "text", "media_type": "text/plain",
                                                                      "data": "plain words"}})
        self.assertIn("plain words", plain["text"])
        blocks = documents.expand_part({"type": "document", "source": {"type": "content", "content": [
            {"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}})
        self.assertIn("one\n\ntwo", blocks["text"])

    def test_what_cannot_be_read_is_refused(self):
        for part in ({"type": "file", "file": {"file_id": "file-abc"}},
                     {"type": "document", "source": {"type": "url", "url": "https://x/y.pdf"}},
                     {"type": "file", "file": {"filename": "x", "file_data": ""}}):
            with self.subTest(part=part), self.assertRaises(documents.DocumentError):
                documents.expand_part(part)

    def test_other_parts_untouched(self):
        self.assertIsNone(documents.expand_part({"type": "text", "text": "x"}))
        self.assertEqual(documents.expand_body({"messages": [{"role": "user", "content": "hi"}]}), 0)
        self.assertEqual(documents.expand_body("not a dict"), 0)


class ResponsesShim(unittest.TestCase):
    def test_codex_input_file_reaches_the_gateway_as_a_file_part(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "ui"))
        import responses_shim
        content = responses_shim._user_content([
            {"type": "input_text", "text": "read this"},
            {"type": "input_file", "filename": "spec.md", "file_data": data_url("text/markdown", b"# Spec")}])
        self.assertEqual(content[1], {"type": "file", "file": {"filename": "spec.md", "file_id": None,
                                                                "file_data": data_url("text/markdown", b"# Spec")}})
        documents.expand_messages([{"role": "user", "content": content}])
        self.assertIn("# Spec", content[1]["text"])


# ---------------------------------------------------------------- the gateway


class Gateway(unittest.TestCase):
    def setUp(self):
        from test_openai_server import FakeEngine, _post_json, _spawn_test_server, _error_body
        self.engine = FakeEngine()
        self.base = _spawn_test_server(self, self.engine)
        self.post = lambda path, body: _post_json(self.base, path, body)
        self.error_body = _error_body

    def test_chat_completions_reads_the_pdf(self):
        body = {"model": "test-model", "max_tokens": 4, "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What does it say?"},
            {"type": "file", "file": {"filename": "a.pdf", "file_data": data_url("application/pdf", pdf_bytes())}}]}]}
        with self.post("/v1/chat/completions", body) as response:
            self.assertEqual(response.status, 200)
            json.load(response)
        prompt = self.engine.calls[0][0]
        prompt = prompt if isinstance(prompt, str) else json.dumps(prompt)
        self.assertIn("World (x) café", prompt)
        self.assertIn('<document name="a.pdf" pages="2">', prompt)

    def test_messages_reads_a_document_block(self):
        body = {"model": "test-model", "max_tokens": 4, "messages": [{"role": "user", "content": [
            {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "the secret is 42"}},
            {"type": "text", "text": "What is the secret?"}]}]}
        with self.post("/v1/messages", body) as response:
            self.assertEqual(response.status, 200)
        prompt = self.engine.calls[0][0]
        self.assertIn("the secret is 42", prompt if isinstance(prompt, str) else json.dumps(prompt))

    def test_an_unreadable_document_is_a_400(self):
        body = {"model": "test-model", "messages": [{"role": "user", "content": [
            {"type": "file", "file": {"filename": "old.doc",
                                      "file_data": data_url("application/msword", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 64)}}]}]}
        status, error = self.error_body(self, lambda: self.post("/v1/chat/completions", body))
        self.assertEqual(status, 400)
        self.assertIn(".docx", error["message"])
        self.assertEqual(self.engine.calls, [])


if __name__ == "__main__":
    unittest.main()
