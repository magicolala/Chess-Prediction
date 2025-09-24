import tomllib
from pathlib import Path


def test_pyproject_hatch_packages_configured():
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as fh:
        config = tomllib.load(fh)

    wheel_config = (
        config.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel")
    )
    assert wheel_config is not None, "tool.hatch.build.targets.wheel must be configured"

    packages = wheel_config.get("packages")
    assert packages, "wheel target must declare packages to include"
    assert "oracle" in packages, "oracle package must be included in the build"
