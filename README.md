# Drzewo decyzyjne z ruletką

## Wymagania wstępne

Należy zacząć od zainstalowania Pythona w wersji 3.13.x. Jedną z możliwości jest pobranie instalatora ze strony [python.org](https://www.python.org/downloads/).

Następnie należy upewnić się, że na komputerze zainstalowane jest [poetry](https://python-poetry.org/), zalecamy użycie [pipx](https://pipx.pypa.io):

```shell
pipx install poetry
```

## Jak zacząć pracę z kodem?

Sklonować repozytorium:

```shell
git clone https://github.com/Depermitto/decision-tree
cd decision-tree
```

Zainstalować wszystkie zależności:

```shell
poetry install --no-root
```

Ostatnim krokiem jest uruchomienie aplikacji, można to zrobić za pomocą komendy:

```shell
poetry run python src/main.py
```
