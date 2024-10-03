import umpypkg as pkg

packages = [
    'numpy',
    'pandas',
    'scipy',
    'matplotlib',
    'scikit-learn',
    'seaborn',
    'tkinter'
]

for package in packages:
    pkg.install(package)