# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

Pokemon-themed Python project in early setup phase. Uses **Poetry** for dependency management.

## Development Setup

```bash
poetry install          # Install dependencies
poetry run python ...   # Run Python scripts
```

- Python: >=3.11
- Package manager: Poetry

## Datasets

Three image datasets are downloaded under `dataset/`:

### 1. HybridShivam/Pokemon (GitHub) — `dataset/HybridShivam-Pokemon/`
- **1160+ high-quality PNG images** — Official Sugimori Artwork (up to Gen VIII)
- Includes alternate forms: Mega, Gmax, Hisuian, etc.
- Naming: `{species_id}.png` (zero-padded to 3 digits), e.g. `001.png`, `006-Mega-X.png`
- **PokeAPI dataset** in `src/dataSet/`:
  - JSON: `pokemon.json`, `pokemon-species.json`, `ability.json`, `move.json`, `evolution-chain.json`
  - CSV: `moves.csv`, `machines.csv`, `pokemon-moves.csv`
- Source: Bulbapedia (scraped), PokeAPI

### 2. arenagrenade — `dataset/arenagrenade-pokemon-images/Pokemon Dataset/`
- **898 PNG images** — All Pokemon from Pokedex (Gen I–VIII)
- Named by lowercase English name: `bulbasaur.png`, `charizard.png`
- Includes some alternate forms: `aegislash-shield.png`, `deoxys-attack.png`
- License: CC0-1.0

### 3. kvpratama — `dataset/kvpratama-pokemon-images/`
- **819 JPG images** in `pokemon_jpg/pokemon_jpg/` — numbered `1.jpg` to `819.jpg`
- **819 PNG images** in `pokemon/pokemon/` — numbered `1.png` to `819.png`
- License: CC0-1.0

## Key Differences Between Datasets

| Feature | HybridShivam | arenagrenade | kvpratama |
|---------|-------------|-------------|-----------|
| Count | 1160+ (with forms) | 898 | 819 |
| Quality | Highest (Sugimori) | Medium | Medium |
| Naming | `{id}.png` | `{name}.png` | `{id}.jpg/png` |
| Forms | Yes (Mega, Gmax, etc.) | Some | No |
| PokeAPI data | Yes | No | No |
