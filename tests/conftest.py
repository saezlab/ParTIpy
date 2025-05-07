import logging
import os
import shutil
import sys
import time
from datetime import timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

# for Python < 3.11 compatibility
try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]

# --- Setup ---
PROJECT_ROOT = Path(__file__).parent.parent
LOG_FILE = PROJECT_ROOT / "tests" / "pytest.log"

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_installed_package_versions():
    """Get versions of packages listed in pyproject.toml."""
    try:
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)

        dependencies = pyproject.get("project", {}).get("dependencies", [])
        package_names = [pkg.split("==")[0].split(">=")[0].split("[")[0].strip().lower() for pkg in dependencies]

        versions = {}
        for pkg in package_names:
            try:
                versions[pkg] = version(pkg)
            except PackageNotFoundError:
                try:
                    versions[pkg] = version(pkg.replace("-", "_"))
                except PackageNotFoundError:
                    versions[pkg] = "Not installed"
        return versions

    except FileNotFoundError:
        logging.warning("pyproject.toml not found - skipping dependency version logging")
        return {}
    except tomllib.TOMLDecodeError as e:
        logging.warning(f"Invalid pyproject.toml: {str(e)} - skipping dependency version logging")
        return {}
    except PermissionError:
        logging.warning("Permission denied when reading pyproject.toml - skipping dependency version logging")
        return {}
    except Exception as e:  # noqa: BLE001
        logging.warning(f"Unexpected error reading dependencies: {str(e)} - skipping version logging")
        return {}


def pytest_sessionstart(session):
    """Log environment and package versions at startup."""
    logging.info("=== Environment ===")
    logging.info(f"Python: {sys.version.split()[0]}")  # Just version without build info
    logging.info(f"Platform: {sys.platform}")
    logging.info(f"Pytest: {pytest.__version__}")

    # Log package versions
    package_versions = get_installed_package_versions()
    if package_versions:
        logging.info("=== Dependencies ===")
        max_len = max(len(pkg) for pkg in package_versions)
        for pkg, ver in sorted(package_versions.items()):
            logging.info(f"{pkg:<{max_len}} : {ver}")


@pytest.fixture(scope="session", autouse=True)
def track_test_session():
    """Track total test session duration."""
    start_time = time.time()
    logging.info("=== Test session started ===")

    yield  # Run all tests

    end_time = time.time()
    duration = timedelta(seconds=end_time - start_time)
    logging.info(f"=== Test session completed in {duration} ===")


@pytest.fixture(scope="session", autouse=True)
def clean_pycache():
    """Delete __pycache__ dirs and log to file."""
    deleted = []
    for root, dirs, _ in os.walk(PROJECT_ROOT, topdown=False):
        if "__pycache__" in dirs:
            path = os.path.join(root, "__pycache__")
            shutil.rmtree(path, ignore_errors=True)
            deleted.append(str(Path(path).relative_to(PROJECT_ROOT)))
    if deleted:
        logging.info("=== Cleaned __pycache__ ===")
        logging.info(" | ".join(f"{path}" for path in sorted(deleted)))


def pytest_runtest_logreport(report):
    """Log test results with duration and failure details."""
    if report.when == "call":
        level = logging.INFO if report.passed else logging.ERROR
        duration = f" ({report.duration:.3f}s)" if hasattr(report, "duration") else ""
        logging.log(level, f"{report.nodeid} - {'PASSED' if report.passed else 'FAILED'}{duration}")

        if report.failed and report.longrepr:
            logging.error(
                "Failure traceback:\n"
                + "\n".join(
                    line
                    for line in str(report.longrepr).splitlines()
                    if line.strip()  # Skip empty lines
                )
            )
