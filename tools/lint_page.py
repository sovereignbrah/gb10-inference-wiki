#!/usr/bin/env python3
"""Lint a GB10 Inference Wiki page: skeleton, frontmatter, pathful wikilinks, placeholders, export safety.

Usage:
  lint_page.py PAGE.md [PAGE2.md ...]        full page lint (structure + scrub)
  lint_page.py --scrub-only FILE.md [...]    export-safety scrub only (research notes, public blocks)
  lint_page.py --allow-private FILE.md       structure checks only (private notes)
Exit 0 = CLEAN, 1 = BLOCKED. A personal denylist at ~/.config/public-lab/denylist.txt is read if present (never printed); it is required only when PUBLIC_LAB_STRICT=1 (the maintainer's publish path).
"""
import os, re, sys

HEADINGS = [
    "## 1. What it is",
    "## 2. How each engine does it",
    "## 3. What is different on GB10 / SM121",
    "## 4. What has been measured",
    "## 5. Levers, ranked",
    "## 6. Protocol",
    "## 7. Anti-patterns and known-bad",
    "## 8. Open questions",
    "## 9. Sources",
]
FM_KEYS = ["title", "slug", "topic_number", "created", "last_updated", "status", "type"]
GRADES = ["[Measured]", "[Community-measured]", "[Code-verified]", "[Historical diagnostic]", "[Interpretation]", "[Proposed]"]
PLACEHOLDERS = ["{NN}", "{slug}", "{Topic title}", "{YYYY-MM-DD}", "TBD", "TODO", "{1|2}", "{grade}"]
SCRUB = [
    ("home-path",   r"/(Users|home)/[a-z0-9_]+"),
    ("tailnet-ip",  r"(?<![\d.])100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]+\.[0-9]+(?![\d.])"),
    ("private-ip",  r"(?<![\d.])(192\.168|10\.[0-9]+|172\.(1[6-9]|2[0-9]|3[01]))\.[0-9]+\.[0-9]+(?![\d.])"),
    ("link-local",  r"fe80::"),
    ("email",       r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"),
    ("ssh-key",     r"ssh-(ed25519|rsa) AAAA"),
    ("token",       r"(ghp_[A-Za-z0-9]{20,}|github_pat_|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,})"),
]
DENYLIST = os.path.expanduser("~/.config/public-lab/denylist.txt")


def load_denylist():
    if not os.path.exists(DENYLIST):
        return None
    with open(DENYLIST) as f:
        return [t.strip() for t in f if t.strip()]


def scrub(text, issues, denylist):
    for label, pat in SCRUB:
        for m in re.finditer(pat, text, flags=re.I):
            line = text.count("\n", 0, m.start()) + 1
            issues.append(f"scrub[{label}] line {line}")
    if denylist is None:
        if os.environ.get("PUBLIC_LAB_STRICT") == "1":
            issues.append("scrub[denylist] personal denylist missing; refusing to pass")
        return
    for term in denylist:
        pat = r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])" if re.fullmatch(r"[A-Za-z0-9 _.-]+", term) else re.escape(term)
        for m in re.finditer(pat, text, flags=re.I):
            line = text.count("\n", 0, m.start()) + 1
            issues.append(f"scrub[denylist] line {line}")  # term deliberately not printed


def structure(text, issues):
    # frontmatter
    if not text.startswith("---\n"):
        issues.append("frontmatter: missing")
        fm = ""
    else:
        end = text.find("\n---", 4)
        fm = text[4:end] if end > 0 else ""
        for k in FM_KEYS:
            if not re.search(rf"^{k}:\s*\S", fm, flags=re.M):
                issues.append(f"frontmatter: missing key '{k}'")
    # headings in order
    pos = -1
    for h in HEADINGS:
        i = text.find("\n" + h + "\n")
        if i < 0:
            issues.append(f"skeleton: missing heading '{h}'")
        elif i < pos:
            issues.append(f"skeleton: heading out of order '{h}'")
        else:
            pos = i
    # placeholders
    for p in PLACEHOLDERS:
        if p in text:
            issues.append(f"placeholder: '{p}' present")
    # section 4 rows carry a grade
    s4 = text.find("\n## 4. ")
    s5 = text.find("\n## ", s4 + 1) if s4 > 0 else -1
    if s5 < 0:
        s5 = len(text)
    if s4 > 0:
        for n, line in enumerate(text[s4:s5].splitlines()):
            if line.startswith("|") and not re.match(r"^\|\s*-", line) and not line.lower().startswith("| change"):
                if not any(g in line for g in GRADES):
                    issues.append(f"section4: table row without an evidence grade: {line[:60]}")


def wikilinks(text, issues):
    for m in re.finditer(r"\[\[([^\]|]+)", text):
        target = m.group(1).strip()
        if not (target.startswith("wiki/") or target.startswith("raw/")):
            line = text.count("\n", 0, m.start()) + 1
            issues.append(f"wikilink: bare or non-pathful link '[[{target}' line {line}")


def main(argv):
    mode = "page"
    files = []
    for a in argv:
        if a == "--scrub-only":
            mode = "scrub"
        elif a == "--allow-private":
            mode = "private"
        else:
            files.append(a)
    if not files:
        print(__doc__)
        return 2
    denylist = load_denylist()
    blocked = 0
    for f in files:
        issues = []
        try:
            text = open(f, encoding="utf-8").read()
        except Exception as e:
            print(f"{f}: BLOCKED (cannot read: {e})")
            blocked += 1
            continue
        if mode in ("page", "private"):
            structure(text, issues)
            wikilinks(text, issues)
        if mode in ("page", "scrub"):
            scrub(text, issues, denylist)
        if issues:
            blocked += 1
            print(f"{f}: BLOCKED ({len(issues)} issues)")
            for i in issues[:60]:
                print("  - " + i)
            if len(issues) > 60:
                print(f"  ... {len(issues) - 60} more")
        else:
            print(f"{f}: CLEAN")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
