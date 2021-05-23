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

Or you can use `pip` directly (currently this package only publishes in `Testpypi`), 

```
pip install -i https://test.pypi.org/simple/ colocationship==0.0.1
```

After installation, anywhere on the system you can then import the package:

```python
import colocationship as cl
```

## Dataset

All processed datasets (Weeplaces, BrightKite, Gowalla) we used in this repo can be found in [Google Drive](https://drive.google.com/drive/folders/1C71Atf4x7eTAEazAPehih5_zkBqqfX4M?usp=sharing).

## Usage

The example please refers to `example/example_weeplaces.ipynb`.

## Contributing

PRs accepted.

## License

MIT © Zexun Chen