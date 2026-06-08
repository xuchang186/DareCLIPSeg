def list_pretrained(as_str: bool = False):
    return []


def list_pretrained_tag_models(tag: str):
    return []


def list_pretrained_model_tags(model: str):
    return []


def get_pretrained_url(model: str, tag: str):
    return ''


def download_pretrained(url: str, root: str = ''):
    raise RuntimeError("Automatic pretrained weight download is disabled. Use a local checkpoint path.")
