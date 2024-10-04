import umpypkg as pkg

packages = [
    'scipy==1.8.0',
    'matplotlib==3.7.0',
    'scikit-learn==1.3.0',
    'seaborn==0.12.2',
]

for package in packages:
    pkg.install(package)
