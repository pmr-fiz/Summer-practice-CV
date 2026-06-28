from setuptools import setup, find_packages

setup(
    name="medical_cv_project",
    version="1.0.0",
    description="Детектирование опухолей головного мозга на МРТ (сравнение 6 моделей)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2",
        "torchvision>=0.17",
        "ultralytics>=8.2",
        "effdet==0.4.1",
        "timm>=0.9.12",
        "transformers>=4.40",
        "opencv-python",
        "pandas",
        "numpy",
        "matplotlib",
        "pyyaml",
        "tqdm",
        "pycocotools",
    ],
)
