"""Fetch a representative photograph for each cat and dog breed group of the demo.

The images are looked up from Wikipedia, which for an article about a breed
almost always has a representative photograph as its lead image. The lead image
is hosted on Wikimedia Commons, so its license and author can be read from the
Commons API and recorded alongside the downloaded file.

Many of the breed groups in the survey data are groups rather than single
breeds, for example "Scenthounds" or "Teacup dogs". Those are mapped by hand to
an article about a breed that represents the group, which is noted in the
credits file so that the choice is visible to anyone looking at the demo.

Run from the repository root:
    python scripts/fetch_breed_images.py
"""
# ruff: noqa: T201

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path(__file__).parents[1] / "webui" / "static" / "animal_pics"
CREDITS_PATH = OUTPUT_DIR / "credits.json"

USER_AGENT = "DESDEO-catsanddogs-demo/1.0 (https://github.com/industrial-optimization-group/DESDEO)"

# The image is fetched at a comfortably larger size than it is shown at, and
# then shrunk down. Fetching at the display size directly gives a noticeably
# softer image on high density displays.
THUMB_WIDTH = 640
DISPLAY_WIDTH = 480
JPEG_QUALITY = 82

# Breed group as it appears in the survey data, mapped to the Wikipedia article
# whose lead image represents it. A group that covers several breeds is mapped
# to one representative breed.
CAT_ARTICLES: dict[str, str] = {
    "Abyssinian": "Abyssinian cat",
    "American_Curl": "American Curl",
    "Bengal": "Bengal cat",
    "British": "British Shorthair",
    "Burmese": "Burmese cat",
    "Cornish_Rex": "Cornish Rex",
    "European": "European Shorthair",
    "House_cat": "Cat",
    "Korat": "Korat",
    "Landrace_Cat_Longhair": "Domestic long-haired cat",
    "Landrace_Cat_Shorthair": "Domestic short-haired cat",
    "Maine_Coon": "Maine Coon",
    "Norwegian_Forest_Cat": "Norwegian Forest cat",
    "Ocicat": "Ocicat",
    "Oriental": "Oriental Shorthair",
    "Other": "Cat",
    "Persian_and_Exotic": "Persian cat",
    "Ragdoll": "Ragdoll",
    "Russian_Blue": "Russian Blue cat",
    "Sacred_Birman": "Birman",
    "Siamese_and_Balinese": "Siamese cat",
    "Siberian_and_Neva_Masquerade": "Siberian cat",
    "Somali": "Somali cat",
    "Sphynx_and_Devon_Rex": "Sphynx cat",
    "Turkish_Angora": "Turkish Angora",
    "Turkish_Van": "Turkish Van",
}

DOG_ARTICLES: dict[str, str] = {
    "Australian_Shepherd": "Australian Shepherd",
    "Belgian_shepherd_dogs": "Belgian Shepherd",
    "Bernese_Mountain_Dogs": "Bernese Mountain Dog",
    "Bichon_type_dogs": "Bichon Frise",
    "Border_Collie": "Border Collie",
    "Brachycephalic_dogs": "Pug",
    "Bull_type_terriers": "Staffordshire Bull Terrier",
    "Chinese_Crested_Dog": "Chinese Crested Dog",
    "Collie_Rough": "Rough Collie",
    "Collie_Smooth": "Smooth Collie",
    "Dachshunds": "Dachshund",
    "English_herders": "Old English Sheepdog",
    "European_sighthounds": "Greyhound",
    "Fighting_dogs": "American Staffordshire Terrier",
    "Finnish_Lapphund": "Finnish Lapphund",
    "German_Shepherd_Dog": "German Shepherd",
    "German_spitz_related": "German Spitz",
    "Golden_Retriever": "Golden Retriever",
    "Hunting_terriers": "Fox Terrier",
    "Jack_Russell_Terrier": "Jack Russell Terrier",
    "Japanese_Asian_primitive": "Shiba Inu",
    "Labrador_Retriever": "Labrador Retriever",
    "Lagotto_Romagnolo_Romagna_Water_Dog": "Lagotto Romagnolo",
    "Lapponian_Herder": "Lapponian Herder",
    "Livestock_guardian_dogs": "Pyrenean Mountain Dog",
    "Mastiff_type_dogs": "English Mastiff",
    "Middle_European_herders": "Briard",
    "Middle_European_utility_dogs": "Hovawart",
    "Miniature_Pinscher": "Miniature Pinscher",
    "Miniature_Schnauzer": "Miniature Schnauzer",
    "Mixed_breed": "Mongrel",
    "Northern_companion_spitz": "Pomeranian dog",
    "Northern_hunting_spitz": "Finnish Spitz",
    "Nova_Scotia_Duck_Tolling_Retriever": "Nova Scotia Duck Tolling Retriever",
    "Other_breed": "Dog",
    "Other_companion_dogs": "Cavalier King Charles Spaniel",
    "Parson_type_terriers": "Parson Russell Terrier",
    "Pinschers_Schnauzers": "Standard Schnauzer",
    "Pointers": "Pointer (dog breed)",
    "Poodles": "Poodle",
    "Primitive_sighthounds": "Saluki",
    "Retrievers_flushing_dogs": "English Springer Spaniel",
    "Scenthounds": "Beagle",
    "Schapendoes": "Schapendoes",
    "Shetland_Sheepdog": "Shetland Sheepdog",
    "Sled_dogs": "Siberian Husky",
    "Spanish_Water_Dog": "Spanish Water Dog",
    "Teacup_dogs": "Chihuahua (dog breed)",
    "Welsh_Corgis": "Pembroke Welsh Corgi",
    "Whippet": "Whippet",
    "White_Swiss_Shepherd_Dog": "White Swiss Shepherd Dog",
    "Yard_terriers": "Airedale Terrier",
}


def get_json(url: str) -> dict:
    """Fetch a URL and parse the response as JSON.

    Args:
        url (str): the URL to fetch.

    Returns:
        dict: the parsed response.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace in a string.

    The author field returned by the Commons API is a snippet of HTML.

    Args:
        text (str): the text to clean up.

    Returns:
        str: the text without HTML tags.
    """
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text)).strip()


def lead_image(article: str) -> tuple[str, str] | None:
    """Look up the lead image of a Wikipedia article.

    Args:
        article (str): the title of the article.

    Returns:
        tuple[str, str] | None: the file name of the lead image on Commons and a
            URL to a thumbnail of it, or None if the article has no lead image.
    """
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&format=json&redirects=1"
        f"&titles={urllib.parse.quote(article)}"
        f"&prop=pageimages&piprop=name%7Cthumbnail&pithumbsize={THUMB_WIDTH}"
    )
    pages = get_json(url).get("query", {}).get("pages", {})

    for page in pages.values():
        name = page.get("pageimage")
        thumbnail = page.get("thumbnail", {}).get("source")
        if name and thumbnail:
            return name, thumbnail

    return None


def license_info(file_name: str) -> dict[str, str]:
    """Look up the license and author of a file on Wikimedia Commons.

    Most of the licenses in use on Commons require the author to be credited.
    The author is usually in the `Artist` field, but not always, so the
    `Attribution` and `Credit` fields and finally the account that uploaded the
    file are used as fallbacks. The description page is recorded in every case,
    since linking to it is what makes the attribution checkable.

    Args:
        file_name (str): the name of the file, without the "File:" prefix.

    Returns:
        dict[str, str]: the license, author, whether attribution is required,
            and the description page of the file.
    """
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&format=json"
        f"&titles=File:{urllib.parse.quote(file_name)}"
        "&prop=imageinfo&iiprop=extmetadata%7Curl%7Cuser"
    )
    pages = get_json(url).get("query", {}).get("pages", {})

    for page in pages.values():
        info = page.get("imageinfo")
        if not info:
            continue
        extra = info[0].get("extmetadata", {})

        author = "unknown"
        for field in ("Artist", "Attribution", "Credit"):
            value = strip_html(extra.get(field, {}).get("value", ""))
            if value and value.lower() not in ("", "own work", "unknown"):
                author = value
                break
        if author == "unknown" and info[0].get("user"):
            author = f"{info[0]['user']} (uploader)"

        return {
            "license": strip_html(extra.get("LicenseShortName", {}).get("value", "unknown")),
            "author": author,
            "attribution_required": extra.get("AttributionRequired", {}).get("value", "") == "true",
            "source": info[0].get("descriptionurl", ""),
        }

    return {"license": "unknown", "author": "unknown", "attribution_required": True, "source": ""}


def download(url: str, destination: Path) -> None:
    """Download a file and shrink it to the size the demo shows it at.

    Shrinking keeps the images committed to the repository to a few megabytes
    in total instead of about fifteen. Pillow is not a declared dependency of
    DESDEO, so the image is left at its downloaded size if Pillow is missing.

    Args:
        url (str): the URL to download.
        destination (Path): where to write the downloaded file.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        destination.write_bytes(response.read())

    try:
        from PIL import Image, ImageOps  # noqa: PLC0415
    except ImportError:
        print("  (Pillow not installed, leaving the image at its downloaded size)")
        return

    image = ImageOps.exif_transpose(Image.open(destination)).convert("RGB")
    image.thumbnail((DISPLAY_WIDTH, DISPLAY_WIDTH), Image.LANCZOS)
    image.save(destination, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)


def fetch_all(articles: dict[str, str], animal: str, credits_so_far: dict) -> list[str]:
    """Fetch an image for every breed group of one animal.

    Args:
        articles (dict[str, str]): breed groups mapped to Wikipedia articles.
        animal (str): either "cat" or "dog", recorded in the credits.
        credits_so_far (dict): the credits collected so far, added to in place.

    Returns:
        list[str]: the breed groups no image could be fetched for.
    """
    failed = []

    for breed_group, article in sorted(articles.items()):
        destination = OUTPUT_DIR / f"{breed_group}.jpg"

        try:
            found = lead_image(article)
            if found is None:
                print(f"  no lead image: {breed_group} ({article})")
                failed.append(breed_group)
                continue

            file_name, thumbnail_url = found

            # Downloading again would only shrink an already shrunk image
            # further, so an image that is already there is left alone. Its
            # credits are refreshed regardless.
            if not destination.exists():
                download(thumbnail_url, destination)

            credits_so_far[breed_group] = {
                "animal": animal,
                "file": f"{breed_group}.jpg",
                "represented_by": article,
                "commons_file": file_name,
                **license_info(file_name),
            }
            print(f"  ok: {breed_group} <- {article} ({destination.stat().st_size // 1024} kB)")
        except Exception as error:
            print(f"  failed: {breed_group} ({article}): {error}")
            failed.append(breed_group)

        time.sleep(0.4)

    return failed


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    collected: dict[str, dict] = {}

    print("Fetching cat images...")
    failed_cats = fetch_all(CAT_ARTICLES, "cat", collected)

    print("Fetching dog images...")
    failed_dogs = fetch_all(DOG_ARTICLES, "dog", collected)

    CREDITS_PATH.write_text(json.dumps(collected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nWrote {len(collected)} images to {OUTPUT_DIR}")
    print(f"Credits written to {CREDITS_PATH}")
    if failed_cats or failed_dogs:
        print(f"Failed: {failed_cats + failed_dogs}")
