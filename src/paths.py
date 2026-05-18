from pathlib import Path


def get_project_paths():
    """
    Detect the project root by searching upward for the folder containing 'data_raw'.
    Returns a dictionary with the main project paths.
    """
    current_dir = Path.cwd()

    base_dir = None
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "data_raw").exists():
            base_dir = parent
            break

    if base_dir is None:
        raise FileNotFoundError(
            "Could not locate the project folder containing 'data_raw'. "
            "Please ensure the notebook is somewhere inside the project directory."
        )

    data_raw = base_dir / "data_raw"
    data_processed = base_dir / "data_processed"

    resources = base_dir / "resources"
    templates_dir = resources / "templates"

    outputs = base_dir / "outputs"
    tables_dir = outputs / "tables"
    figures_dir = outputs / "figures"
    excel_dir = outputs / "excel"
    intermediate_dir = outputs / "intermediate"

    return {
        "BASE_DIR": base_dir,
        "DATA_RAW": data_raw,
        "DATA_PROCESSED": data_processed,
        "RESOURCES_DIR": resources,
        "TEMPLATES_DIR": templates_dir,
        "OUTPUTS": outputs,
        "TABLES_DIR": tables_dir,
        "FIGURES_DIR": figures_dir,
        "EXCEL_DIR": excel_dir,
        "INTERMEDIATE_DIR": intermediate_dir,
    }


def ensure_project_directories(paths):
    """
    Ensure that the main project directories exist.
    """
    for key in [
        "DATA_PROCESSED",
        "RESOURCES_DIR",
        "TEMPLATES_DIR",
        "OUTPUTS",
        "TABLES_DIR",
        "FIGURES_DIR",
        "EXCEL_DIR",
        "INTERMEDIATE_DIR",
    ]:
        paths[key].mkdir(parents=True, exist_ok=True)


def get_part2_paths():
    """
    Return project paths for Part II (same structure as Part I).
    """
    return get_project_paths()


def ensure_part2_directories(paths):
    """
    Ensure that output directories for Part II exist.
    """
    ensure_project_directories(paths)