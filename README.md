![GitHub](https://img.shields.io/github/license/Magica-Chen/co-locationship)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/Magica-Chen/co-locationship)
![GitHub last commit](https://img.shields.io/github/last-commit/Magica-Chen/co-locationship)
[![DOI](https://zenodo.org/badge/309197285.svg)](https://zenodo.org/badge/latestdoi/309197285)
![GitHub Repo stars](https://img.shields.io/github/stars/Magica-Chen/gptp_multi_output?style=social)
![Twitter](https://img.shields.io/twitter/follow/MagicaChen?style=social)

# co-locationship

Leveraging information transfer in social and co-location networks to improve predictability in human mobility

The code was tested on Python 3.6.

## Install (via pypi version)

```
pip install colocationship
```

## Install (via GitHub)

```
git clone https://github.com/Magica-Chen/co-locationship.git

cd co-locationship
```

The dependencies package are shown in `requirements.txt`, also, you can run 

```
pip install -r requirements.txt
```

After installing all dependencies clone the repository and do (inside the top directory):

```
pip install . 
```

This will install a copy of the code as a package. If you want to install a package that links to the cloned code, run

```
pip install --editable .
```

This makes changes to the source files in the cloned directory immediately available to everything that imports the package.


After installation, anywhere on the system you can then import the package:

```python
import colocationship as cl
```

## Dataset

All processed datasets (Weeplaces, BrightKite, Gowalla) we used in this repo can be found in [Google Drive](https://drive.google.com/drive/folders/1C71Atf4x7eTAEazAPehih5_zkBqqfX4M?usp=sharing).

## Usage

The example please refers to `example/example_weeplaces.ipynb`.

## Citation

Chen, Z., Kelty, S., Evsukoff, A.G. et al. Contrasting social and non-social sources of predictability in human mobility. Nat Commun 13, 1922 (2022).

@article{chen2022constrasting,
  title={Contrasting social and non-social sources of predictability in human mobility},
  author={Chen, Zexun and Kelty, Sean and Evsukoff, Alexandre G. and Welles, Brooke Foucault and Bagrow, James P and Menezes, Ronaldo and Ghoshal, Gourab},
  journal={Nature communications},
  volume={13},
  number={1},
  pages={1--9},
  year={2022},
  publisher={Nature Publishing Group}
}
