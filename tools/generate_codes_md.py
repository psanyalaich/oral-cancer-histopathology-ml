from pathlib import Path

OUTPUT_FILE = "codes.md"

INCLUDE_EXTENSIONS = {
    ".py",
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".toml"
}

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "feature_cache",
    "results",
    "data",
    "venv",
    ".venv",
    "node_modules",
    ".idea",
    ".vscode",
    "wandb",
    "checkpoints",
    "artifacts"
}

EXCLUDE_FILES = {OUTPUT_FILE}

def should_skip(path: Path):
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True

    if path.name in EXCLUDE_FILES:
        return True

    return False

def get_language(ext):
    mapping = {
        ".py": "python",
        ".md": "markdown",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".toml": "toml"
    }

    return mapping.get(ext, "")

repo_root = Path(".")
files = []

for path in repo_root.rglob("*"):
    if path.is_file():

        if should_skip(path):
            continue

        if path.suffix not in INCLUDE_EXTENSIONS:
            continue

        files.append(path)

files = sorted(files)

output_lines = []

current_parent = None

for file_path in files:

    parent = str(file_path.parent)

    if parent != current_parent:
        output_lines.append(f"\n---\n")
        output_lines.append(f"# {parent}\n")
        current_parent = parent

    output_lines.append(f"## {file_path.name}\n")

    lang = get_language(file_path.suffix)

    output_lines.append(f"```{lang}")

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        content = f"ERROR READING FILE: {e}"

    output_lines.append(content)
    output_lines.append("```\n")

Path(OUTPUT_FILE).write_text(
    "\n".join(output_lines),
    encoding="utf-8"
)

print(f"Generated {OUTPUT_FILE}")
