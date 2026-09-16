from pathlib import Path
from datetime import datetime, timezone

README_PATH = Path("README.md")

if not README_PATH.exists():
    raise FileNotFoundError("README.md not found. Run this script from the repository root.")

text = README_PATH.read_text(encoding="utf-8")
lines = text.splitlines(keepends=True)

if not lines:
    # Empty file: just add one line
    new_first_line = "# NMTF Co-Clustering\n"
    lines = [new_first_line]
else:
    # Replace only the first line
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    new_first_line = f"# NMTF Co-Clustering (updated {date_str})\n"
    lines[0] = new_first_line

new_text = "".join(lines)

if text == new_text:
    print("README.md unchanged; no commit created.")
else:
    README_PATH.write_text(new_text, encoding="utf-8")
    print("README.md updated: first line changed.")