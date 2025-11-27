"""Setup script for Multi-Agent Translation Pipeline package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = []

setup(
    name="multi-agent-translation-pipeline",
    version="1.0.0",
    author="MSc Project",
    author_email="",
    description="Multi-Agent Translation Pipeline for Semantic Drift Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multi-agent-translation-pipeline",
    packages=find_packages(where=".", include=["src", "src.*"]),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "pylint>=2.17.0",
            "mypy>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-experiment=scripts.run_real_experiment:main",
            "run-analysis=scripts.run_real_analysis:main",
            "generate-graphs=scripts.generate_real_graphs:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md", "*.txt"],
    },
    zip_safe=False,
    keywords="multi-agent translation semantic-drift nlp ai research",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/multi-agent-translation-pipeline/issues",
        "Source": "https://github.com/yourusername/multi-agent-translation-pipeline",
        "Documentation": "https://github.com/yourusername/multi-agent-translation-pipeline/blob/main/README.md",
    },
)
