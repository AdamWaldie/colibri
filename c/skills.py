#!/usr/bin/env python3
"""Agent Skills for colibri's own chat: the skill folders Claude Code, Codex and other
agents already use (a folder per skill holding SKILL.md: YAML front matter with
`name` and `description`, then the instructions), offered to any chat client.

    GET /v1/skills          {"data": [{"name", "description", "source"}]}
    a chat request's        "skills": ["name", ...] puts those skills' instructions in
                            the system prompt of that request, before the engine reads it

Claude Code and Codex load a skill themselves, through a tool call, when it fits;
a plain chat has no tools, so here the user turns a skill on (the web chat's Skills
menu, or `/name` at the start of a message) and its instructions ride along with
every turn while it is on. Only SKILL.md is read: a skill's scripts and other files
need an agent that can run them.

Folders searched, first wins on a name: COLI_SKILLS_DIRS (os.pathsep-separated),
then ~/.claude/skills, ~/.codex/skills, ~/.agents/skills. Standard library only.
"""
import os
import re

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MAX_SKILL_BYTES = 64 * 1024          # one SKILL.md
MAX_TOTAL_BYTES = 192 * 1024         # every skill a request turns on, together
MAX_SKILLS = 200                     # listed


class SkillError(ValueError):
    pass


def skill_dirs(env=None, home=None):
    """The folders that hold skills, existing or not, in search order."""
    env = os.environ if env is None else env
    home = home or os.path.expanduser("~")
    extra = [p for p in (env.get("COLI_SKILLS_DIRS") or "").split(os.pathsep) if p.strip()]
    return extra + [os.path.join(home, ".claude", "skills"), os.path.join(home, ".codex", "skills"),
                    os.path.join(home, ".agents", "skills")]


def parse(text):
    """(front matter as a dict of plain strings, body) of a SKILL.md. Only the simple
    `key: value` lines skills use (a quoted or a folded `>`/`|` value included)."""
    meta, body = {}, text
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is not None:
            body = "\n".join(lines[end + 1:])
            key = None
            for line in lines[1:end]:
                found = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$", line)
                if found and not line.startswith((" ", "\t")):
                    key, value = found.group(1), found.group(2).strip()
                    if value in (">", "|", ">-", "|-"):
                        value = ""
                    elif len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                        value = value[1:-1]
                    meta[key] = value
                elif key and line.strip():
                    meta[key] = (meta[key] + " " + line.strip()).strip()
    return meta, body.strip()


def _read(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read(MAX_SKILL_BYTES + 1)


def discover(dirs=None):
    """{name: {"name", "description", "source", "path"}} for every readable skill."""
    found = {}
    for root in dirs if dirs is not None else skill_dirs():
        try:
            entries = sorted(os.listdir(root))
        except OSError:
            continue
        for entry in entries:
            path = os.path.join(root, entry, "SKILL.md")
            if not os.path.isfile(path):
                continue
            try:
                meta, _ = parse(_read(path))
            except OSError:
                continue
            name = meta.get("name") or entry
            if not NAME_RE.match(name) or name in found:
                continue
            found[name] = {"name": name, "description": meta.get("description", "")[:1024],
                           "source": root, "path": path}
            if len(found) >= MAX_SKILLS:
                return found
    return found


def listing(dirs=None):
    """GET /v1/skills: what a client shows (the folder each one is in, not its file)."""
    return {"object": "list", "data": [{"name": s["name"], "description": s["description"],
                                        "source": s["source"]} for s in discover(dirs).values()]}


def instructions(names, dirs=None):
    """The system-prompt text for the skills `names` turns on."""
    known = discover(dirs)
    parts, total = [], 0
    for name in names:
        skill = known.get(name)
        if skill is None:
            raise SkillError(f"There is no skill called {name!r} on this server "
                             f"(GET /v1/skills lists them).")
        text = _read(skill["path"])
        if len(text) > MAX_SKILL_BYTES:
            raise SkillError(f"The skill {name!r} is larger than {MAX_SKILL_BYTES // 1024} KB.")
        _, body = parse(text)
        total += len(body)
        if total > MAX_TOTAL_BYTES:
            raise SkillError(f"The skills turned on are larger than {MAX_TOTAL_BYTES // 1024} KB "
                             "together: turn some off.")
        parts.append(f'<skill name="{name}">\n{body}\n</skill>')
    return ("The user turned on these skills. Follow a skill's instructions whenever the "
            "request is one it covers; it may mention scripts or files you cannot run or "
            "open here, so do what it describes yourself or say what is missing.\n\n"
            + "\n\n".join(parts))


def expand_body(body, dirs=None):
    """A chat request with "skills": the field removed and those skills' instructions
    added to its system prompt (the first system message, or a new one first)."""
    if not isinstance(body, dict) or "skills" not in body:
        return body
    names = body.pop("skills")
    if names in (None, []):
        return body
    if not isinstance(names, list) or not all(isinstance(n, str) and NAME_RE.match(n) for n in names):
        raise SkillError("`skills` must be a list of skill names.")
    names = list(dict.fromkeys(names))
    text = instructions(names, dirs)
    messages = body.get("messages")
    if isinstance(messages, list):
        first = messages[0] if messages else None
        if isinstance(first, dict) and first.get("role") in ("system", "developer") \
                and isinstance(first.get("content"), str):
            messages[0] = dict(first, content=first["content"].rstrip() + "\n\n" + text)
        else:
            messages.insert(0, {"role": "system", "content": text})
    elif isinstance(body.get("system"), str):                     # Anthropic shape
        body["system"] = body["system"].rstrip() + "\n\n" + text
    elif isinstance(body.get("system"), list):
        body["system"] = body["system"] + [{"type": "text", "text": text}]
    else:
        body["system"] = text
    return body
