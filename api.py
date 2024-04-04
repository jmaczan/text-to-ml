def endpoint(url: str, slug: str = "") -> str:
    if url.endswith("/") and slug.startswith("/"):
        return url + slug[1:]
    elif url.endswith("/") or slug.startswith("/"):
        return url + slug
    else:
        return url + "/" + slug
