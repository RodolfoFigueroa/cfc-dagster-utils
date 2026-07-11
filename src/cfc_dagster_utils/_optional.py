from typing import NoReturn


def raise_optional_dependency_error(
    error: ModuleNotFoundError,
    *,
    import_name: str,
    dependency_name: str,
    extra: str,
) -> NoReturn:
    """Raise an actionable error when a direct optional dependency is missing."""
    if error.name != import_name:
        raise error

    message = (
        f"Optional dependency '{dependency_name}' is required for this feature. "
        "Install it with "
        f"`pip install 'cfc_dagster_utils[{extra}]'`."
    )
    raise ImportError(message) from error
