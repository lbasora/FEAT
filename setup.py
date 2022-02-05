from setuptools import setup, find_packages

setup(
    name="feat",
    version="0.1",
    description="Fuel Estimation in Air Transportation (FEAT)",
    license="MIT",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "openap", "statsmodels", "tqdm", "traffic"],
    python_requires=">=3.7.1",
    include_package_data=True,
    package_data={
        "feat.data": ["seats.csv"],
    },
)
