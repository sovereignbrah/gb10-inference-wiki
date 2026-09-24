#!/bin/bash
# Publish gate: blocks anything private from entering the public repo. Contributors can run it as is;
# the maintainer's publish.sh runs it with PUBLIC_LAB_STRICT=1, which also requires a personal denylist.
# Generic patterns live here (safe to publish); the personal denylist lives
# OUTSIDE the repo at ~/.config/public-lab/denylist.txt and is never committed.
set -u
FAIL=0
FILES=$(git ls-files 2>/dev/null; git diff --cached --name-only 2>/dev/null)
FILES=$(echo "$FILES" | sort -u | grep -vE "^tools/(scrub.sh|lint_page.py)$" || true)
check() { # $1=pattern label, $2=grep args...
  local label="$1"; shift
  local hits
  hits=$(echo "$FILES" | xargs -I{} grep -lIiE "$@" {} 2>/dev/null || true)
  if [ -n "$hits" ]; then echo "BLOCKED [$label]:"; echo "$hits" | sed "s/^/  /"; FAIL=1; fi
}
# Generic classes: home paths, private/tailnet IPs, link-local, emails, keys, tokens
check "home-path"      '/(Users|home)/[a-z0-9_]+'
check "tailnet-ip"     '100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]+\.[0-9]+'
check "private-ip"     '(192\.168|10\.[0-9]+|172\.(1[6-9]|2[0-9]|3[01]))\.[0-9]+\.[0-9]+'
check "link-local"     'fe80::'
check "email"          '[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}'
check "ssh-key"        'ssh-(ed25519|rsa) AAAA'
check "token"          '(ghp_[A-Za-z0-9]{20,}|github_pat_|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,})'
# Personal denylist (kept outside the repo)
DL="$HOME/.config/public-lab/denylist.txt"
if [ -f "$DL" ]; then
  while IFS= read -r term; do
    [ -z "$term" ] && continue
    check "denylist" -w "$term"   # whole-word: a term must not match inside another word
  done < "$DL"
else
  if [ "${PUBLIC_LAB_STRICT:-0}" = "1" ]; then echo "personal denylist missing at $DL — refusing to pass (strict mode)"; FAIL=1; else echo "note: no personal denylist at $DL; generic checks only"; fi
fi
[ $FAIL -eq 0 ] && echo "scrub: CLEAN" || echo "scrub: BLOCKED — fix or remove the files above; nothing was pushed"
exit $FAIL
