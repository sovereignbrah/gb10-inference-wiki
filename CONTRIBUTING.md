# How pages in this wiki are written

Every topic page has the same nine sections, in this order, with these exact headings:
1. What it is · 2. How each engine does it · 3. What is different on GB10 / SM121 · 4. What has been measured ·
5. Levers, ranked · 6. Protocol · 7. Anti-patterns and known-bad · 8. Open questions · 9. Sources.

Evidence vocabulary, in square brackets on every claim:
[Measured] · [Community-measured] · [Code-verified] · [Historical diagnostic] · [Interpretation] · [Proposed].

Rules:
- No number without baseline, workload, and scope. A nominal spec is never presented as achieved.
- Every source pinned: commit SHA or tag for code, dated fetch for pages, arXiv id for papers.
- Negative findings stay. A new experiment appends a dated finding and says what it supersedes.
- Pages are export-safe: nodes are node-1, node-2, node-3; hardware is generic; no paths, hostnames, IPs, or names.
  Own measurements are cited by packet date and title. Lane-specific detail lives in the private overlay, not here.
- Links in this repo are relative markdown links (`[Speculative decoding](02-speculative-decoding.md)`), converted on export from the wikilinks in the maintainer's working notes. Pull requests use the relative form.
- Frontmatter keys every topic page needs: `title`, `slug`, `topic_number`, `created`, `last_updated`, `status`, `type` (checked by `tools/lint_page.py`).
- Community work is credited by handle and link.
