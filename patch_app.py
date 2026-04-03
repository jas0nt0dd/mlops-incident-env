import re

path = r"C:\projects\mlops-incident-env\mlops-incident-env\server\app.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Remove old if __name__ block
content = re.sub(
    r'\n# ── Entry point.*',
    '',
    content,
    flags=re.DOTALL
)

# Append new main() function
new_block = """
# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    import os, uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1, log_level="info")

if __name__ == "__main__":
    main()
"""
content = content.rstrip() + "\n" + new_block

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("server/app.py patched successfully")