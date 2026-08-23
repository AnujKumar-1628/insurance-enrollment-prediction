from setuptools import setup, find_packages


def get_requirements(file_path: str) -> list[str]:
    """
    Read project dependencies from requirements.txt.
    """
    requirements = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line and not line.startswith("-"):
                requirements.append(line)

    return requirements


setup(
    name="insurance-enrollment-prediction",
    version="0.1.0",
    description=(
        "Machine learning pipeline for predicting "
        "voluntary insurance enrollment"
    ),
    author="Anuj Kumar",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements("requirements.txt"),
)