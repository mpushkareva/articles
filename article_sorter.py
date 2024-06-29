import re

def extract_subsections(markdown_text):
    subsection_pattern = re.compile(r'(### \d+\. .+?)(?=### \d+\. |\Z)', re.DOTALL)
    subsections = subsection_pattern.findall(markdown_text)
    return subsections

def extract_year(subsection):
    year_pattern = re.compile(r'\*\*Year\*\*: (\d{4})')
    match = year_pattern.search(subsection)
    if match:
        return int(match.group(1))
    return None

def sort_subsections_by_year(subsections):
    return sorted(subsections, key=extract_year)

def main():
    with open('/Users/mihailratko/articles/structured_pruning/structured_pruning.md', 'r') as f:
        markdown_text = f.read()

    subsections = extract_subsections(markdown_text)
    sorted_subsections = sort_subsections_by_year(subsections)

    sorted_markdown = "# Structured pruning\n\n" + "\n\n".join(sorted_subsections)

    with open('/Users/mihailratko/articles/structured_pruning/structured_pruning_sorted.md', 'w') as f:
        f.write(sorted_markdown)

if __name__ == '__main__':
    main()