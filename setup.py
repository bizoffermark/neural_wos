from setuptools import find_packages, setup

setup(
    name="wos",
    version="0.0.1",
    author="Hong Chul Nam",
    description="Neural Walk-on-Sphere for elliptic PDEs",
    packages=find_packages(include=["wos"]),
    python_requires=">=3.10",
    license="MIT",
    zip_safe=True,
    install_requires=[
        # hydra
        "hydra-core==1.2.0",
        "hydra-joblib-launcher==1.2.0",
        "hydra-submitit-launcher==1.2.0",
        # logging, eval, and plotting
        "pandas==2.1.3",
        "matplotlib==3.8.1",
        "wandb==0.16.0",
        "plotly==5.18.0",
        "tqdm==4.66.1",
    ],
    extras_require={
        "interactive": ["jupyter==1.0.0", "kaleido==0.2.1", "seaborn==0.13.0"],
        "dev": ["isort==5.10.1", "black==22.10.0"],
        "torch": ["torch", "torchvision"],
        "dill": ["dill==0.3.7"],
    },
)
