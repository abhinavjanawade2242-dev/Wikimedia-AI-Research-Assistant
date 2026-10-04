import json


def parse_json_field(value):
    """Convert a JSON string into a Python object."""

    if not value:
        return None

    if isinstance(value, (dict, list)):
        return value

    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def extract_section_text(sections):
    """Extract readable text from article sections."""

    if not sections:
        return ""

    text_parts = []

    for section in sections:
        section_name = section.get("name", "")

        if section_name:
            text_parts.append(section_name)

        for part in section.get("has_parts", []):
            if part.get("value"):
                text_parts.append(part["value"])

    return "\n".join(text_parts)


def create_article_record(row):
    """Create a clean article record from a dataframe row."""

    sections = parse_json_field(row["sections"])
    infoboxes = parse_json_field(row["infoboxes"])

    article = {
        "id": row["identifier"],
        "title": row["name"],
        "description": row["description"],
        "abstract": row["abstract"],
        "sections": sections,
        "infoboxes": infoboxes,
        "url": row["url"],
        "language": row["in_language"],
        "main_entity": row["main_entity"],
        "additional_entities": row["additional_entities"],
        "references": row["references"],
    }

    article["full_text"] = extract_section_text(sections)

    return article