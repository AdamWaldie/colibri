#!/usr/bin/env python3
"""Documents in a conversation, as text the model reads: the files Claude and
ChatGPT accept, sent through any of the APIs colibri serves.

    Anthropic /v1/messages   {"type": "document", "source": {"type": "base64",
                             "media_type": "application/pdf", "data": ...}}, or a
                             "text" source, or a "content" source of text blocks;
                             "title" and "context" kept
    OpenAI chat              {"type": "file", "file": {"filename", "file_data":
                             "data:<mime>;base64,..."}}
    OpenAI Responses         {"type": "input_file", "filename", "file_data"}
                             (Codex, through ui/responses_shim.py)

expand_body() rewrites each such part into a text part,
<document name="report.pdf" pages="12"> ... </document>, before the request
reaches the model: every model reads documents this way, vision or not.

Formats: PDF (pypdf when it is installed, else the reader below: text operators,
FlateDecode, object streams, ToUnicode maps), Word .docx, PowerPoint .pptx,
Excel .xlsx, OpenDocument .odt .ods .odp, EPUB, RTF, HTML, and anything that is
text (plain, Markdown, CSV, JSON, XML, YAML, source code, logs). A scanned PDF
has no text to extract; that is said in the document's place, not guessed at.

Standard library only (pypdf optional).
"""
import base64
import html
import html.parser
import io
import os
import re
import zipfile
import zlib
from xml.etree import ElementTree

MAX_CHARS = int(os.environ.get("COLI_DOC_MAX_CHARS") or 400_000)   # about 100k tokens
MAX_BYTES = 32 << 20
# Documents arrive from the network: what a compressed stream or an archive member
# may expand to is bounded, so a small "zip bomb" cannot take the server's memory.
MAX_INFLATED = 64 << 20          # one PDF stream or one archive member, decompressed
MAX_ARCHIVE_TOTAL = 256 << 20    # all the members read from one .docx/.xlsx/.epub
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class DocumentError(ValueError):
    """A document that cannot be read: the reason, for a 400 answer."""


# ---------------------------------------------------------------- the API side


def _data_url(value):
    """(mime, bytes) from a data: URL, or (None, bytes) from bare base64."""
    if not isinstance(value, str) or not value:
        raise DocumentError("a document part has no data")
    mime = None
    if value.startswith("data:"):
        head, _, value = value.partition(",")
        mime = head[5:].split(";")[0] or None
        if ";base64" not in head:
            return mime, value.encode("utf-8")
    try:
        data = base64.b64decode(value, validate=False)
    except (ValueError, TypeError) as error:
        raise DocumentError("a document's data is not valid base64") from error
    if len(data) > MAX_BYTES:
        raise DocumentError(f"a document is larger than {MAX_BYTES >> 20} MB")
    return mime, data


def _as_text_part(name, mime, data, title=None, context=None):
    try:
        text, attrs = extract(name, mime, data)
    except DocumentError:
        raise
    except Exception as error:  # noqa: BLE001 - a malformed file is the client's 400, not a 500
        raise DocumentError(f"{name}: could not be read ({type(error).__name__})") from error
    text = CONTROL.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    label = title or name or "document"
    head = f'<document name="{html.escape(label, quote=True)}"'
    for key, value in attrs.items():
        head += f' {key}="{value}"'
    body = text
    if len(body) > MAX_CHARS:
        body = body[:MAX_CHARS] + f"\n[... the rest of this document ({len(text) - MAX_CHARS} characters) was left out]"
    note = f"\n{context}" if context else ""
    return {"type": "text", "text": f"{head}>{note}\n{body}\n</document>"}


def expand_part(part):
    """The text part a document part becomes, or None for any other part."""
    if not isinstance(part, dict):
        return None
    kind = part.get("type")
    if kind == "file":
        file = part.get("file") or {}
        if file.get("file_id") and not file.get("file_data"):
            raise DocumentError("file ids need an uploads API colibri does not have: send file_data instead")
        mime, data = _data_url(file.get("file_data"))
        return _as_text_part(file.get("filename") or "document", mime, data)
    if kind == "input_file":
        if part.get("file_id") and not part.get("file_data"):
            raise DocumentError("file ids need an uploads API colibri does not have: send file_data instead")
        mime, data = _data_url(part.get("file_data"))
        return _as_text_part(part.get("filename") or "document", mime, data)
    if kind == "document":
        source = part.get("source") or {}
        title, context = part.get("title"), part.get("context")
        stype = source.get("type")
        if stype == "base64":
            mime = source.get("media_type")
            _, data = _data_url(source.get("data"))
            return _as_text_part(title or ("document.pdf" if mime == "application/pdf" else "document"),
                                 mime, data, title, context)
        if stype == "text":
            return _as_text_part(title or "document.txt", "text/plain",
                                 str(source.get("data") or "").encode("utf-8"), title, context)
        if stype == "content":
            blocks = source.get("content")
            text = blocks if isinstance(blocks, str) else "\n\n".join(
                b.get("text", "") for b in blocks or [] if isinstance(b, dict) and b.get("type") == "text")
            return _as_text_part(title or "document.txt", "text/plain", text.encode("utf-8"), title, context)
        if stype in ("url", "file"):
            raise DocumentError(f"a document by {stype} needs to be fetched or uploaded first: send it as base64")
        raise DocumentError(f"unknown document source {stype!r}")
    return None


def expand_messages(messages):
    """Rewrite every document part of every message, in place; the count."""
    count = 0
    for message in messages if isinstance(messages, list) else []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            replacement = expand_part(part)
            if replacement is not None:
                content[index] = replacement
                count += 1
        # A message left with text parts only reads as one string, as before.
        if count and all(isinstance(p, dict) and p.get("type") == "text" for p in content) \
                and message.get("role") != "assistant" and len(content) > 0:
            message["content"] = "\n\n".join(p.get("text", "") for p in content)
    return count


def expand_body(body):
    """Documents in a chat or messages request body become text; the count."""
    return expand_messages(body.get("messages")) if isinstance(body, dict) else 0


# ---------------------------------------------------------------- formats


ZIP_KINDS = {"docx": "word/document.xml", "pptx": "ppt/presentation.xml", "xlsx": "xl/workbook.xml",
             "odt": "content.xml", "ods": "content.xml", "odp": "content.xml", "epub": "META-INF/container.xml"}
MIME_EXT = {"application/pdf": "pdf", "text/html": "html", "application/rtf": "rtf", "text/rtf": "rtf",
            "application/epub+zip": "epub",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
            "application/vnd.oasis.opendocument.text": "odt",
            "application/vnd.oasis.opendocument.spreadsheet": "ods",
            "application/vnd.oasis.opendocument.presentation": "odp"}
LEGACY_OFFICE = {"doc", "xls", "ppt"}


def kind_of(name, mime, data):
    ext = os.path.splitext(name or "")[1].lower().lstrip(".")
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        try:
            with _Bounded(data) as archive:
                names = set(archive.namelist())
        except zipfile.BadZipFile:
            names = set()
        if ext in ZIP_KINDS and ZIP_KINDS[ext] in names:
            return ext
        for kind, marker in ZIP_KINDS.items():
            if marker in names and (kind not in ("odt", "ods", "odp") or ext == kind):
                return kind
        if "content.xml" in names:
            return "odt"
        return "zip"
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "legacy-office"
    if data[:5] == b"{\\rtf":
        return "rtf"
    if (mime or "") in MIME_EXT:
        return MIME_EXT[mime]
    if ext in ("html", "htm", "xhtml") or re.match(rb"\s*(<!doctype html|<html)", data[:512], re.I):
        return "html"
    if ext in LEGACY_OFFICE:
        return "legacy-office"
    return "text"


def extract(name, mime, data):
    """(text, attributes for the <document> tag)."""
    kind = kind_of(name, mime, data)
    if kind == "pdf":
        pages = pdf_text(data)
        text = "\n\n".join(f"[page {i}]\n{page}".rstrip() for i, page in enumerate(pages, 1) if page.strip())
        if not text.strip():
            text = ("[no text could be read from this PDF: it is probably scanned pages (pictures of text). "
                    "Send the pages as images to a model with vision to read them.]")
        return text, {"pages": str(len(pages))}
    if kind == "docx":
        return docx_text(data), {}
    if kind == "pptx":
        slides = pptx_text(data)
        return "\n\n".join(f"[slide {i}]\n{s}" for i, s in enumerate(slides, 1)), {"slides": str(len(slides))}
    if kind == "xlsx":
        return xlsx_text(data), {}
    if kind in ("odt", "ods", "odp"):
        return odf_text(data), {}
    if kind == "epub":
        return epub_text(data), {}
    if kind == "html":
        return html_text(decode_text(data)), {}
    if kind == "rtf":
        return rtf_text(decode_text(data)), {}
    if kind == "legacy-office":
        raise DocumentError(f"{name}: the old binary Office format (.doc/.xls/.ppt) cannot be read; "
                            "save it as .docx/.xlsx/.pptx or PDF")
    if kind == "zip":
        raise DocumentError(f"{name}: a zip archive, not a document")
    text = decode_text(data)
    if text is None:
        raise DocumentError(f"{name}: not a document colibri can read (binary data)")
    return text, {}


def decode_text(data):
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", "replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", "replace")
    sample = data[:8192]
    if sample and sample.count(b"\0") > len(sample) // 100:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", "replace")


# ---------------------------------------------------------------- Office Open XML, OpenDocument, EPUB


def _local(tag):
    return tag.rsplit("}", 1)[-1]


class _Bounded:
    """A zip archive whose members are read within MAX_INFLATED each and
    MAX_ARCHIVE_TOTAL in all."""

    def __init__(self, data):
        self.zip = zipfile.ZipFile(io.BytesIO(data))
        self.total = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.zip.close()

    def namelist(self):
        return self.zip.namelist()

    def read(self, name):
        info = self.zip.getinfo(name)
        if info.file_size > MAX_INFLATED or self.total + info.file_size > MAX_ARCHIVE_TOTAL:
            raise DocumentError(f"{name} inside the document is too large to read")
        with self.zip.open(info) as member:
            data = member.read(MAX_INFLATED + 1)
        if len(data) > MAX_INFLATED:
            raise DocumentError(f"{name} inside the document is too large to read")
        self.total += len(data)
        return data


def _xml(archive, name):
    try:
        return ElementTree.fromstring(archive.read(name))
    except (KeyError, ElementTree.ParseError):
        return None


def _numbered(names, prefix):
    found = [n for n in names if re.fullmatch(re.escape(prefix) + r"(\d+)\.xml", n)]
    return sorted(found, key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)))


def _paragraphs(root, para="p", text="t", tab="tab", brk="br", cell="tc", row="tr"):
    """Text of a WordprocessingML/DrawingML tree: a line per paragraph, table rows
    as cells joined with ' | '."""
    lines = []

    def run(node):
        out = []
        for child in node.iter():
            tag = _local(child.tag)
            if tag == text and child.text:
                out.append(child.text)
            elif tag == tab:
                out.append("\t")
            elif tag == brk:
                out.append("\n")
        return "".join(out)

    def walk(node):
        for child in node:
            tag = _local(child.tag)
            if tag == row:
                cells = [" ".join(run(p) for p in c.iter() if _local(p.tag) == para).strip()
                         for c in child if _local(c.tag) == cell]
                lines.append(" | ".join(cells))
            elif tag == para:
                lines.append(run(child))
            else:
                walk(child)

    walk(root)
    return "\n".join(lines).strip()


def docx_text(data):
    with _Bounded(data) as archive:
        root = _xml(archive, "word/document.xml")
        return _paragraphs(root) if root is not None else ""


def pptx_text(data):
    slides = []
    with _Bounded(data) as archive:
        for name in _numbered(archive.namelist(), "ppt/slides/slide"):
            root = _xml(archive, name)
            slides.append(_paragraphs(root) if root is not None else "")
    return slides


def xlsx_text(data, max_rows=5000):
    with _Bounded(data) as archive:
        shared = []
        root = _xml(archive, "xl/sharedStrings.xml")
        if root is not None:
            for si in root:
                shared.append("".join(t.text or "" for t in si.iter() if _local(t.tag) == "t"))
        names = []
        book = _xml(archive, "xl/workbook.xml")
        if book is not None:
            names = [s.get("name") for s in book.iter() if _local(s.tag) == "sheet"]
        out = []
        for index, name in enumerate(_numbered(archive.namelist(), "xl/worksheets/sheet")):
            sheet = _xml(archive, name)
            if sheet is None:
                continue
            rows = []
            for row in sheet.iter():
                if _local(row.tag) != "row":
                    continue
                cells = []
                for c in row:
                    if _local(c.tag) != "c":
                        continue
                    kind, value = c.get("t"), ""
                    v = next((x for x in c if _local(x.tag) == "v"), None)
                    if kind == "s" and v is not None and (v.text or "").isdigit() and int(v.text) < len(shared):
                        value = shared[int(v.text)]
                    elif kind == "inlineStr":
                        value = "".join(t.text or "" for t in c.iter() if _local(t.tag) == "t")
                    elif v is not None:
                        value = v.text or ""
                    cells.append(_csv(value))
                rows.append(",".join(cells))
                if len(rows) >= max_rows:
                    rows.append(f"[... more rows left out after {max_rows}]")
                    break
            title = names[index] if index < len(names) and names[index] else f"Sheet{index + 1}"
            out.append(f"[sheet {title}]\n" + "\n".join(rows))
        return "\n\n".join(out)


def _csv(value):
    return f'"{value.replace(chr(34), chr(34) * 2)}"' if any(c in value for c in ',"\n') else value


def odf_text(data):
    with _Bounded(data) as archive:
        root = _xml(archive, "content.xml")
    if root is None:
        return ""
    lines = []

    def text_of(node):
        parts = [node.text or ""]
        for child in node:
            tag = _local(child.tag)
            if tag == "s":
                parts.append(" " * int(child.get("{urn:oasis:names:tc:opendocument:xmlns:text:1.0}c", "1")))
            elif tag == "tab":
                parts.append("\t")
            elif tag == "line-break":
                parts.append("\n")
            else:
                parts.append(text_of(child))
            parts.append(child.tail or "")
        return "".join(parts)

    def walk(node):
        for child in node:
            tag = _local(child.tag)
            if tag == "table-row":
                cells = [" ".join(text_of(p) for p in c if _local(p.tag) in ("p", "h"))
                         for c in child if _local(c.tag) == "table-cell"]
                if any(cells):
                    lines.append(" | ".join(cells).rstrip(" |"))
            elif tag in ("p", "h"):
                lines.append(text_of(child))
            else:
                walk(child)

    walk(root)
    return "\n".join(lines).strip()


def epub_text(data):
    with _Bounded(data) as archive:
        names = archive.namelist()
        order = []
        container = _xml(archive, "META-INF/container.xml")
        opf_path = next((r.get("full-path") for r in container.iter() if _local(r.tag) == "rootfile"), None) \
            if container is not None else None
        opf = _xml(archive, opf_path) if opf_path else None
        if opf is not None:
            base = os.path.dirname(opf_path)
            items = {i.get("id"): i.get("href") for i in opf.iter() if _local(i.tag) == "item"}
            for ref in opf.iter():
                if _local(ref.tag) == "itemref" and items.get(ref.get("idref")):
                    order.append(os.path.normpath(os.path.join(base, items[ref.get("idref")])).replace(os.sep, "/"))
        if not order:
            order = sorted(n for n in names if n.lower().endswith((".xhtml", ".html", ".htm")))
        chapters = []
        for name in order:
            try:
                chapters.append(html_text(decode_text(archive.read(name)) or ""))
            except KeyError:
                continue
        return "\n\n".join(c for c in chapters if c.strip())


# ---------------------------------------------------------------- HTML and RTF


class _HTMLText(html.parser.HTMLParser):
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
             "header", "footer", "pre", "blockquote", "table", "ul", "ol"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "template"):
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")
        elif tag in ("td", "th"):
            self.out.append(" | ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "template"):
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_text(text):
    parser = _HTMLText()
    parser.feed(text or "")
    parser.close()
    joined = re.sub(r"[ \t\r\f\v]+", " ", "".join(parser.out))
    return re.sub(r"\n\s*\n\s*(\n\s*)+", "\n\n", joined).strip()


def rtf_text(text):
    """Plain text of an RTF file: groups that hold no text (fonts, colours,
    pictures) dropped, control words removed, \\par as a new line."""
    out, depth, skip_at = [], 0, None
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "{":
            depth += 1
            if skip_at is None and re.match(r"\{\\\*|\{\\(fonttbl|colortbl|stylesheet|info|pict|header|footer)\b",
                                            text[i:i + 12]):
                skip_at = depth
            i += 1
        elif ch == "}":
            if skip_at == depth:
                skip_at = None
            depth -= 1
            i += 1
        elif ch == "\\":
            m = re.match(r"\\([a-z]+)(-?\d+)? ?|\\'([0-9a-fA-F]{2})|\\(.)", text[i:])
            if not m:
                i += 1
                continue
            if skip_at is None:
                word, hexed, sym = m.group(1), m.group(3), m.group(4)
                if word in ("par", "line", "row"):
                    out.append("\n")
                elif word == "tab" or word == "cell":
                    out.append("\t")
                elif word == "u" and m.group(2):
                    out.append(chr(int(m.group(2)) % 65536))
                elif hexed:
                    out.append(bytes([int(hexed, 16)]).decode("cp1252", "replace"))
                elif sym in ("\\", "{", "}"):
                    out.append(sym)
            i += m.end()
        else:
            if skip_at is None and ch not in "\r\n":
                out.append(ch)
            i += 1
    return re.sub(r"\n{3,}", "\n\n", "".join(out)).strip()


# ---------------------------------------------------------------- PDF


def pdf_text(data):
    """Text of each page: pypdf when it is installed, else the reader below."""
    try:
        import pypdf  # noqa: PLC0415 - optional, better on unusual files
    except ImportError:
        pypdf = None
    if pypdf is not None:
        import logging  # noqa: PLC0415
        logging.getLogger("pypdf").setLevel(logging.ERROR)    # its warnings are not the client's business
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                reader.decrypt("")
            return [page.extract_text() or "" for page in reader.pages]
        except Exception:  # noqa: BLE001 - fall back to the built-in reader
            pass
    return MiniPDF(data).pages_text()


_TOKEN = re.compile(rb"\s*(?:(%[^\r\n]*)|(<<|>>|\[|\]|\{|\})|(/[^\s/<>\[\]()%{}]*)|(\()|(<[0-9A-Fa-f\s]*>)"
                    rb"|([+-]?(?:\d+\.?\d*|\.\d+))|([A-Za-z_'\"*][A-Za-z0-9_'\"*.]*))")


class Ref(tuple):
    pass


class MiniPDF:
    """Enough of PDF to read its text: indirect objects (also inside object
    streams), FlateDecode, the page tree with inherited resources, fonts'
    ToUnicode maps or a single-byte encoding, and the text operators."""

    def __init__(self, data):
        self.data = data
        self.objects = {}
        self.streams = {}
        if re.search(rb"/Encrypt\s", data):
            raise DocumentError("this PDF is encrypted (password protected): save an unprotected copy")
        self._scan()

    # ---- objects
    def _scan(self):
        for match in re.finditer(rb"(\d+)\s+(\d+)\s+obj\b", self.data):
            num = int(match.group(1))
            try:
                value, end = self._parse(self.data, match.end())
            except (ValueError, IndexError):
                continue
            self.objects[num] = value
            stream = re.match(rb"\s*stream\r?\n", self.data[end:end + 20])
            if stream and isinstance(value, dict):
                start = end + stream.end()
                length = value.get("Length")
                if isinstance(length, Ref):
                    length = None
                if isinstance(length, int) and self.data[start + length:start + length + 20].lstrip().startswith(b"endstream"):
                    stop = start + length
                else:
                    stop = self.data.find(b"endstream", start)
                    stop = stop if stop >= 0 else len(self.data)
                self.streams[num] = self.data[start:stop]
        for num, value in list(self.objects.items()):
            if isinstance(value, dict) and value.get("Type") == "ObjStm":
                self._object_stream(num, value)

    def _object_stream(self, num, head):
        try:
            raw = self.stream(num)
            count, first = int(head.get("N", 0)), int(head.get("First", 0))
            numbers = [int(x) for x in raw[:first].split()]
            for k in range(min(count, len(numbers) // 2)):
                obj, offset = numbers[2 * k], numbers[2 * k + 1]
                if obj not in self.objects:
                    self.objects[obj], _ = self._parse(raw, first + offset)
        except (ValueError, IndexError, zlib.error):
            pass

    def get(self, value):
        seen = 0
        while isinstance(value, Ref) and seen < 32:
            value = self.objects.get(value[0])
            seen += 1
        return value

    def stream(self, num):
        raw = self.streams.get(num, b"")
        head = self.objects.get(num) or {}
        filters = self.get(head.get("Filter")) if isinstance(head, dict) else None
        filters = filters if isinstance(filters, list) else [filters] if filters else []
        for f in filters:
            f = self.get(f)
            if f == "FlateDecode":
                inflater = zlib.decompressobj()
                try:
                    raw = inflater.decompress(raw, MAX_INFLATED)
                except zlib.error:
                    return b""
                if inflater.unconsumed_tail:
                    raise DocumentError("a stream in this PDF expands to more than "
                                        f"{MAX_INFLATED >> 20} MB: not read")
            elif f == "ASCIIHexDecode":
                raw = bytes.fromhex(re.sub(rb"[^0-9A-Fa-f]", b"", raw.split(b">")[0]).decode())
            elif f == "ASCII85Decode":
                raw = base64.a85decode(raw.strip().removesuffix(b"~>"), adobe=False)
            else:
                return b""          # images and other filters hold no text
        return raw

    def stream_of(self, value):
        return self.stream(value[0]) if isinstance(value, Ref) else b""

    # ---- the object syntax
    def _parse(self, buf, pos):
        m = _TOKEN.match(buf, pos)
        while m and m.group(1):
            pos = m.end()
            m = _TOKEN.match(buf, pos)
        if not m:
            raise ValueError("bad token")
        pos = m.end()
        if m.group(2) == b"<<":
            out = {}
            while True:
                m2 = _TOKEN.match(buf, pos)
                if m2 and m2.group(2) == b">>":
                    return out, m2.end()
                key, pos = self._parse(buf, pos)
                value, pos = self._parse(buf, pos)
                if isinstance(key, str):
                    out[key] = value
        if m.group(2) == b"[":
            out = []
            while True:
                m2 = _TOKEN.match(buf, pos)
                if m2 and m2.group(2) == b"]":
                    return out, m2.end()
                item, pos = self._parse(buf, pos)
                out.append(item)
        if m.group(3):
            return m.group(3)[1:].decode("latin-1"), pos
        if m.group(4):
            return self._literal(buf, pos)
        if m.group(5):
            hexed = re.sub(rb"\s", b"", m.group(5)[1:-1])
            return bytes.fromhex((hexed + b"0" * (len(hexed) % 2)).decode()), pos
        if m.group(6):
            text = m.group(6)
            number = float(text) if b"." in text else int(text)
            m2 = re.match(rb"\s+(\d+)\s+R\b", buf[pos:pos + 24])
            if isinstance(number, int) and m2:
                return Ref((number, int(m2.group(1)))), pos + m2.end()
            return number, pos
        word = m.group(7)
        if word == b"true":
            return True, pos
        if word == b"false":
            return False, pos
        if word == b"null":
            return None, pos
        raise ValueError(f"unexpected {word!r}")

    @staticmethod
    def _literal(buf, pos):
        out, depth = bytearray(), 1
        escapes = {ord("n"): 10, ord("r"): 13, ord("t"): 9, ord("b"): 8, ord("f"): 12}
        while pos < len(buf):
            c = buf[pos]
            if c == 0x5C:                         # backslash
                pos += 1
                e = buf[pos] if pos < len(buf) else 0
                if e in escapes:
                    out.append(escapes[e])
                elif 0x30 <= e <= 0x37:
                    digits = re.match(rb"[0-7]{1,3}", buf[pos:pos + 3]).group(0)
                    out.append(int(digits, 8) & 0xFF)
                    pos += len(digits) - 1
                elif e in (0x0A, 0x0D):
                    if e == 0x0D and buf[pos + 1:pos + 2] == b"\n":
                        pos += 1
                else:
                    out.append(e)
            elif c == 0x28:
                depth += 1
                out.append(c)
            elif c == 0x29:
                depth -= 1
                if depth == 0:
                    return bytes(out), pos + 1
                out.append(c)
            else:
                out.append(c)
            pos += 1
        return bytes(out), pos

    # ---- pages and fonts
    def pages(self):
        root = None
        for value in self.objects.values():
            if isinstance(value, dict) and value.get("Type") == "Catalog":
                root = value
                break
        found = []

        def walk(node, inherited, depth=0):
            node = self.get(node)
            if not isinstance(node, dict) or depth > 64:
                return
            resources = self.get(node.get("Resources")) or inherited
            if node.get("Type") == "Pages" or "Kids" in node:
                for kid in self.get(node.get("Kids")) or []:
                    walk(kid, resources, depth + 1)
            else:
                found.append((node, resources))

        if root is not None:
            walk(root.get("Pages"), {})
        if not found:   # no usable page tree: every page object, in file order
            found = [(v, self.get(v.get("Resources")) or {}) for v in self.objects.values()
                     if isinstance(v, dict) and v.get("Type") == "Page"]
        return found

    def font_decoder(self, font):
        font = self.get(font)
        if not isinstance(font, dict):
            return lambda raw: raw.decode("latin-1"), 1
        cmap = self.get(font.get("ToUnicode"))
        if isinstance(font.get("ToUnicode"), Ref):
            mapping, width = self._cmap(self.stream_of(font["ToUnicode"]))
            if mapping:
                def decode(raw, mapping=mapping, width=width):
                    out = []
                    for i in range(0, len(raw) - width + 1, width):
                        code = int.from_bytes(raw[i:i + width], "big")
                        out.append(mapping.get(code, "" if width > 1 else chr(code)))
                    return "".join(out)
                return decode, width
        del cmap
        encoding = self.get(font.get("Encoding"))
        if font.get("Subtype") == "Type0" or encoding in ("Identity-H", "Identity-V"):
            return lambda raw: raw.decode("utf-16-be", "ignore"), 2   # often wrong; nothing better without a map
        codec = "mac_roman" if encoding == "MacRomanEncoding" else "cp1252"
        return lambda raw, codec=codec: raw.decode(codec, "replace"), 1

    @staticmethod
    def _cmap(data):
        mapping, width = {}, 1
        span = re.search(rb"begincodespacerange\s*<([0-9A-Fa-f]+)>", data)
        if span:
            width = max(1, len(span.group(1)) // 2)

        def uni(hexed):
            raw = bytes.fromhex(hexed.decode())
            return raw.decode("utf-16-be", "ignore")

        for block in re.findall(rb"beginbfchar(.*?)endbfchar", data, re.S):
            for src, dst in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]*)>", block):
                mapping[int(src, 16)] = uni(dst)
        for block in re.findall(rb"beginbfrange(.*?)endbfrange", data, re.S):
            for lo, hi, rest in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(<[0-9A-Fa-f]*>|\[[^\]]*\])", block):
                lo, hi = int(lo, 16), int(hi, 16)
                if hi - lo > 65535:
                    continue
                if rest.startswith(b"["):
                    for k, dst in enumerate(re.findall(rb"<([0-9A-Fa-f]*)>", rest)):
                        mapping[lo + k] = uni(dst)
                else:
                    base = int(rest[1:-1] or b"0", 16)
                    for k in range(hi - lo + 1):
                        try:
                            mapping[lo + k] = chr(base + k)
                        except (ValueError, OverflowError):
                            break
        return mapping, width

    def pages_text(self):
        out = []
        for page, resources in self.pages():
            fonts = self.get((resources or {}).get("Font")) if isinstance(resources, dict) else {}
            fonts = fonts if isinstance(fonts, dict) else {}
            contents = self.get(page.get("Contents"))
            refs = contents if isinstance(contents, list) else [page.get("Contents")]
            data = b"\n".join(self.stream_of(r) for r in refs if isinstance(r, Ref))
            out.append(self._text(data, fonts))
        return out

    def _text(self, data, fonts):
        decoders, decode = {}, (lambda raw: raw.decode("latin-1"))
        stack, lines, line = [], [], []
        pos, n = 0, len(data)

        def flush():
            if line:
                lines.append("".join(line).rstrip())
                line.clear()

        while pos < n:
            m = _TOKEN.match(data, pos)
            if not m:
                pos += 1
                continue
            if m.group(4) or m.group(5) or m.group(3) or m.group(6):
                try:
                    value, pos = self._parse(data, pos)
                except (ValueError, IndexError):
                    pos = m.end()
                    continue
                stack.append(value)
                continue
            if m.group(2) == b"[":
                try:
                    value, pos = self._parse(data, pos)
                except (ValueError, IndexError):
                    pos = m.end()
                    continue
                stack.append(value)
                continue
            pos = m.end()
            op = m.group(7)
            if op is None:
                stack.clear()
                continue
            if op == b"Tf" and len(stack) >= 2 and isinstance(stack[-2], str):
                name = stack[-2]
                if name not in decoders:
                    decoders[name] = self.font_decoder(fonts.get(name))[0]
                decode = decoders[name]
            elif op in (b"Tj", b"'", b'"') and stack and isinstance(stack[-1], bytes):
                if op != b"Tj":
                    flush()
                line.append(decode(stack[-1]))
            elif op == b"TJ" and stack and isinstance(stack[-1], list):
                for item in stack[-1]:
                    if isinstance(item, bytes):
                        line.append(decode(item))
                    elif isinstance(item, (int, float)) and item < -180:
                        line.append(" ")
            elif op in (b"Td", b"TD") and len(stack) >= 2 and isinstance(stack[-1], (int, float)):
                if stack[-1] != 0:
                    flush()
                elif line and not line[-1].endswith(" "):
                    line.append(" ")
            elif op in (b"T*", b"ET"):
                flush()
            elif op == b"Tm":
                flush()
            elif op == b"BI":                     # an inline image: skip to its end
                end = data.find(b"EI", pos)
                pos = end + 2 if end >= 0 else n
            stack.clear()
        flush()
        return "\n".join(line for line in lines if line.strip())
