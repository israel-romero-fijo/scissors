import re
from pathlib import Path

import yaml


def test_distributable_skill_metadata_and_references():
    root = Path(__file__).resolve().parents[1]
    package = root / "skill" / "scissors"
    source = (package / "SKILL.md").read_text(encoding="utf-8")
    _, frontmatter, body = source.split("---", 2)
    metadata = yaml.safe_load(frontmatter)
    assert metadata["name"] == "scissors"
    assert isinstance(metadata["description"], str)
    assert metadata["description"].strip()
    references = re.findall(r"\]\((references/[^)]+)\)", body)
    assert references
    for reference in references:
        resource = package / reference
        assert resource.is_file(), f"Missing skill reference: {reference}"
        assert resource.read_text(encoding="utf-8").strip()
