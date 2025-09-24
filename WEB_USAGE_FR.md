# Utilisation de l'interface web d'Oracle (Mode Gratuit)

Ce guide explique comment lancer et utiliser l'interface web d'Oracle sur votre machine locale. Le mode gratuit signifie que vous n'avez pas besoin d'une clé API pour un service LLM payant. Vous utiliserez uniquement Stockfish pour l'analyse, ou un modèle de langue local si vous en avez un de configuré.

## Prérequis

1.  **Python 3.11+**: Assurez-vous que Python est installé sur votre système. Vous pouvez le télécharger depuis [python.org](https://www.python.org/).
2.  **Stockfish**: Vous devez disposer d'une version de l'exécutable de Stockfish. Vous pouvez le télécharger depuis le [site officiel de Stockfish](https://stockfishchess.org/download/). Notez bien le chemin d'accès vers l'exécutable (par exemple, `C:\stockfish\stockfish.exe` sous Windows ou `/usr/local/bin/stockfish` sous Linux/macOS).

## Installation

1.  **Clonez le dépôt**: Si ce n'est pas déjà fait, clonez le dépôt `Chess-Predict` sur votre machine locale.
    ```bash
    git clone https://github.com/YoshaIglesias/Chess-Predict.git
    cd Chess-Predict
    ```

2.  **Créez un environnement virtuel**: Il est recommandé d'utiliser un environnement virtuel pour isoler les dépendances du projet.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
    ```

3.  **Installez les dépendances**: Installez les bibliothèques Python nécessaires en utilisant `pip`.
    ```bash
    pip install .
    ```
    Cette commande installe les dépendances de base définies dans le fichier `pyproject.toml`.

## Lancement du serveur web

1.  Une fois les dépendances installées, lancez le serveur web en exécutant le script `Oracle_web.py`:
    ```bash
    python Oracle_web.py
    ```

2.  Le serveur devrait démarrer et écouter sur `http://127.0.0.1:8000`. Vous verrez un message similaire à celui-ci dans votre console :
    ```
     * Running on http://127.0.0.1:8000
    ```

## Utilisation de l'interface web

1.  **Ouvrez votre navigateur**: Rendez-vous à l'adresse [http://127.0.0.1:8000](http://127.0.0.1:8000).

2.  **Configurez Stockfish**:
    - Dans le champ "Chemin Stockfish", entrez le chemin complet vers l'exécutable de Stockfish que vous avez téléchargé. C'est l'étape la plus importante pour que l'analyse fonctionne.

3.  **Entrez le PGN**:
    - Collez la partie au format PGN (Portable Game Notation) dans la grande zone de texte "PGN". La partie doit être arrêtée au coup où vous souhaitez obtenir une prédiction.

4.  **Réglez les paramètres (optionnel)**:
    - **Elo**: Ajustez le niveau Elo des joueurs pour affiner l'analyse.
    - **Cadence**: Choisissez la cadence de la partie (bullet, blitz, rapid, classical).

5.  **Lancez l'analyse**:
    - Cliquez sur le bouton "Analyser".

6.  **Résultats**:
    - Les résultats de l'analyse s'afficheront sous le bouton. Vous verrez le score attendu et un tableau des coups possibles avec leur probabilité.

## Mode gratuit et LLM

L'interface web vous permet de configurer un backend LLM (Grand Modèle de Langage) pour des analyses plus poussées. Cependant, pour une utilisation gratuite, **vous pouvez laisser la section "Configuration LLM" vide**. L'analyse se basera alors sur Stockfish.

Si vous disposez d'un modèle LLM compatible (format GGUF pour llama.cpp ou un modèle sur Hugging Face pour Transformers) et de la configuration matérielle nécessaire, vous pouvez l'utiliser localement sans frais d'API. Pour cela, installez d'abord les dépendances optionnelles :
```bash
pip install .[llm]
```
Ensuite, dans l'interface web, sélectionnez le backend approprié et remplissez les chemins et informations demandés.
