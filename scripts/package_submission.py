"""Build a compact assignment ZIP without datasets, weights or training archives.

Run from any directory with Python 3.10+: python scripts/package_submission.py
Only package copies of the Markdown links are rewritten; project files stay intact.
"""

import argparse
import io
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
from tempfile import NamedTemporaryFile
from urllib.parse import unquote, urlsplit
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "BDDL_HAR_Project"
REPOSITORY = "https://github.com/song6767/bddl-har-comparison"
LINK = re.compile(r"(?P<open>!?\[[^\]\n]*\]\()(?P<target>[^)\n]+)(?P<close>\))")
FIXED_FILES = (
    "README.md", "requirements.txt", "docs/EXPERIMENT_PROTOCOL.md",
    "reports/project_report.docx", "reports/project_report.md",
)
RESULT_FILES = {
    "reports/formal/per_run.csv": "reports/results/formal_per_run.csv",
    "reports/formal/summary.json": "reports/results/formal_summary.json",
    "reports/formal/tuning.csv": "reports/results/formal_tuning.csv",
    "reports/noise/per_run.csv": "reports/results/noise_per_run.csv",
    "reports/noise/summary.json": "reports/results/noise_summary.json",
}


def source_file(root, relative):
    path = root / relative
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Not a regular project file: {relative}")
    return path


def collect_files(root):
    """Use an explicit directory and extension allowlist, not a Git dependency."""
    mapping = {name: name for name in FIXED_FILES}
    for directory, suffixes in (("har_compare", {".py"}), ("scripts", {".py", ".ps1"}), ("tests", {".py"})):
        for path in sorted((root / directory).rglob("*")):
            relative = path.relative_to(root)
            if (path.is_file() and path.suffix in suffixes
                    and not any(part.startswith(".") or part == "__pycache__" for part in relative.parts)):
                mapping[relative.as_posix()] = relative.as_posix()
    if (root / "reports/formal").is_dir():
        mapping.update(RESULT_FILES)
        for directory in ("formal", "noise"):
            for path in sorted((root / "reports" / directory).glob("*.png")):
                mapping[path.relative_to(root).as_posix()] = f"reports/figures/{path.name}"
        for path in sorted((root / "reports/formal").glob("subject_*")):
            if path.is_file() and path.suffix in {".csv", ".json"}:
                mapping[path.relative_to(root).as_posix()] = f"reports/results/{path.name}"
    else:
        # The script can also repackage an already-extracted compact submission.
        for directory, suffixes in (("figures", {".png"}), ("results", {".json", ".csv"})):
            for path in sorted((root / "reports" / directory).glob("*")):
                if path.is_file() and path.suffix in suffixes:
                    relative = path.relative_to(root).as_posix()
                    mapping[relative] = relative
        for destination in RESULT_FILES.values():
            if destination not in mapping:
                raise FileNotFoundError(f"Missing submission result: {destination}")
    destinations = list(mapping.values())
    if len(destinations) != len(set(destinations)):
        raise ValueError("Two project files map to the same archive destination")
    for name in mapping:
        source_file(root, name)
    return mapping


def local_target(document, target):
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    path = unquote(parsed.path).replace("\\", "/")
    if path.startswith("/"):
        raise ValueError(f"Absolute file link in {document}: {target}")
    combined = posixpath.normpath(posixpath.join(posixpath.dirname(document), path))
    if combined == ".." or combined.startswith("../"):
        raise ValueError(f"File link escapes the project in {document}: {target}")
    return combined


def rewrite_markdown(text, source, destination, mapping):
    def rewrite(match):
        target = match["target"]
        relative = local_target(source, target)
        if relative is None:
            return match[0]
        if relative not in mapping:
            raise ValueError(f"Linked file is not included: {source} -> {target}")
        parsed = urlsplit(target)
        new_target = posixpath.relpath(mapping[relative], posixpath.dirname(destination) or ".")
        if parsed.query:
            new_target += "?" + parsed.query
        if parsed.fragment:
            new_target += "#" + parsed.fragment
        return match["open"] + new_target + match["close"]

    text = LINK.sub(rewrite, text)
    if source == "README.md":
        text = text.replace(
            "仓库附带的原实验结果位于 `reports/smoke/`、`reports/formal/` 和 `reports/noise/`。",
            "提交包的主要结果位于 `reports/results/`，图表位于 `reports/figures/`。",
        )
        text = text.replace(
            "仓库保存代码、报告和轻量实验结果。数据与模型权重由上述命令在本地生成。",
            "提交包保存代码、报告和主要实验结果。数据与模型权重由上述命令在本地生成。",
        )
        note = (
            "本提交包保留项目报告、代码、唯一 README、主要结果及图表。"
            "逐轮训练日志、试跑结果和逐检查点原始记录保存在"
            f"[GitHub 仓库]({REPOSITORY})。"
            "报告正文位于 `reports/`，其下仅有 `figures/` 和 `results/` 两个资料目录。"
        )
        if note not in text:
            first, rest = text.split("\n", 1)
            text = first + "\n\n" + note + "\n" + rest
    return text


def prepare_payloads(root):
    mapping = collect_files(root)
    payloads = {}
    for source, destination in sorted(mapping.items(), key=lambda item: item[1]):
        data = source_file(root, source).read_bytes()
        if source.endswith(".md"):
            text = rewrite_markdown(data.decode("utf-8-sig"), source, destination, mapping)
            data = text.replace("\r\n", "\n").encode("utf-8")
        if source.endswith(".py"):
            compile(data.decode("utf-8-sig"), source, "exec")
        payloads[destination] = data
    readmes = [name for name in payloads if PurePosixPath(name).name.lower() == "readme.md"]
    if readmes != ["README.md"]:
        raise ValueError(f"Expected exactly one root README, found {readmes}")
    for name, data in payloads.items():
        if name.endswith(".md"):
            for match in LINK.finditer(data.decode("utf-8")):
                relative = local_target(name, match["target"])
                if relative is not None and relative not in payloads:
                    raise ValueError(f"Broken archive link: {name} -> {match['target']}")
    with ZipFile(io.BytesIO(payloads["reports/project_report.docx"])) as document:
        if document.testzip() is not None or "word/document.xml" not in document.namelist():
            raise ValueError("Project report is not a valid Word document")
    return payloads


def build_submission(root, output):
    payloads = prepare_payloads(root)
    if output.suffix.lower() != ".zip":
        raise ValueError("Submission output must have a .zip extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(prefix=".bddl-submission-", suffix=".zip", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
        with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=9) as archive:
            for name, data in payloads.items():
                archive.writestr(f"{ARCHIVE_ROOT}/{name}", data)
        with ZipFile(temporary) as archive:
            if archive.testzip() is not None or len(archive.namelist()) != len(payloads):
                raise ValueError("Submission archive integrity check failed")
            for name, data in payloads.items():
                if archive.read(f"{ARCHIVE_ROOT}/{name}") != data:
                    raise ValueError(f"Archive content mismatch: {name}")
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"archive": str(output), "files": len(payloads), "bytes": output.stat().st_size,
            "report_subdirectories": ["figures", "results"], "readme_count": 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("submission/BDDL_HAR_Project_2026.zip"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    print(json.dumps(build_submission(PROJECT_ROOT, output), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
