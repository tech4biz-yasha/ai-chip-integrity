"""
The submission route has three parts that must agree: the GitHub issue form, the
Submit a row page, and the places that link to them. These tests fail when one drifts.
Run from the repo root:  python -m pytest -q
"""

import pathlib
import re

import pytest

yaml = pytest.importorskip("yaml")

ROOT = pathlib.Path(__file__).resolve().parents[1]
FORM_PATH = ROOT / ".github" / "ISSUE_TEMPLATE" / "submit-a-row.yml"
FORM = yaml.safe_load(FORM_PATH.read_text())
CONFIG = yaml.safe_load((ROOT / ".github" / "ISSUE_TEMPLATE" / "config.yml").read_text())
PAGE = (ROOT / "site" / "pages" / "submit.html").read_text()
README = (ROOT / "README.md").read_text()
FORM_URL = "https://github.com/tech4biz-yasha/ai-chip-integrity/issues/new?template=submit-a-row.yml"
# The ten files a full run writes, as the CLI names them.
TEN_FILES = ["clean", "inject", "kernels", "kernels_inject", "abft", "abft_inject",
             "patterns", "patterns_inject", "memcheck", "memcheck_inject"]


def field(field_id):
    return next(b for b in FORM["body"] if b.get("id") == field_id)


def test_the_form_is_a_valid_issue_form():
    assert FORM["name"] and FORM["description"]
    for block in FORM["body"]:
        assert block["type"] in {"markdown", "input", "textarea", "dropdown", "checkboxes"}
        assert "attributes" in block
        if block["type"] != "markdown":
            assert block.get("id"), f"{block['attributes'].get('label')}: every field needs an id"
            assert block["attributes"].get("label")


def test_the_form_asks_for_everything_a_row_needs():
    for field_id in ("chip", "files", "how", "suite-version", "software", "host"):
        assert field(field_id)["validations"]["required"] is True, f"{field_id} must be required"


def test_the_files_field_names_all_ten_result_files():
    description = field("files")["attributes"]["description"]
    missing = [name for name in TEN_FILES if f"NAME_{name}.jsonl" not in description]
    assert not missing, f"the form does not name these result files: {missing}"


def test_the_form_refuses_a_quick_run_and_edited_files():
    labels = " ".join(o["label"] for o in field("confirm")["attributes"]["options"])
    assert "--quick" in labels
    assert "edited" in labels
    assert all(o["required"] for o in field("confirm")["attributes"]["options"])


def test_the_dropdown_offers_every_way_the_readme_says_to_run_it():
    options = " ".join(field("how")["attributes"]["options"]).lower()
    for route in ("pip", "docker", "clone", "action"):
        assert route in options


def test_the_page_links_the_form_and_the_form_config_links_the_page():
    assert FORM_URL in PAGE, "the Submit a row page must link the issue form"
    urls = [link["url"] for link in CONFIG["contact_links"]]
    assert "https://chipintegrity.org/submit.html" in urls


def test_the_readme_points_at_the_form():
    assert FORM_URL in README


def test_the_home_and_run_pages_point_at_the_page():
    for name in ("index", "run"):
        source = (ROOT / "site" / "pages" / f"{name}.html").read_text()
        assert 'href="submit.html"' in source, f"the {name} page does not link the submit page"


def test_the_submit_page_is_in_the_nav_and_the_page_list():
    import build_site
    assert ("submit", "Submit") in build_site.NAV
    assert "submit" in build_site.PAGES
    assert build_site.PAGES["submit"]["learn"], "every page carries what you will learn"


def test_the_built_site_carries_the_page_and_its_links():
    docs = ROOT / "docs"
    built = (docs / "submit.html").read_text()
    assert FORM_URL in built
    assert FORM_URL.split("?")[0] in built
    assert f"{build_site_url()}/submit.html" in (docs / "sitemap.xml").read_text()
    assert 'href="submit.html"' in (docs / "index.html").read_text()


def build_site_url():
    import build_site
    return build_site.SITE


@pytest.mark.parametrize("page", ["index", "run", "submit"])
def test_no_stale_docker_tag_on_the_site(page):
    """The Docker tag in an instruction must be the version this repo is at."""
    import re as _re
    version = _re.search(r'^__version__ = "([\d.]+)"',
                         (ROOT / "chip_integrity" / "__init__.py").read_text(), _re.M).group(1)
    source = (ROOT / "site" / "pages" / f"{page}.html").read_text()
    for tag in _re.findall(r"ai-chip-integrity:([\d.]+)", source):
        assert tag == version, f"{page}.html pins the Docker image at {tag}, but this is {version}"


def test_every_link_on_the_submit_page_is_real():
    for href in re.findall(r'href="([^"]+)"', PAGE):
        if href.startswith("http") or href.startswith("#"):
            continue
        target, _, anchor = href.partition("#")
        assert (ROOT / "docs" / target).exists(), f"submit.html links {target}, which is not built"
