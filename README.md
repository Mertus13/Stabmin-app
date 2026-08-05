# Prédiction de la Stabilité des Pentes

Dashboard interactif de prédiction de la stabilité des pentes en exploitation minière à ciel ouvert, basé sur un modèle XGBoost.

Réalisé dans le cadre du rapport de stage — UI2M (Exploitation minière).

## Fonctionnalités

- Formulaire de saisie des paramètres géotechniques (cohésion, angle de frottement, géométrie de la pente, etc.)
- Prédiction en temps réel (Stable / Instable) avec indice de confiance
- Visualisations interactives (jauge, probabilités, profil radar, importance des variables)
- Export du rapport complet au format PDF

## Utilisation en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Modèle

- Algorithme : XGBoost Classifier
- Variables d'entrée : poids unitaire, cohésion, angle de frottement interne, angle de pente, hauteur de pente, ratio de pression interstitielle, type de renforcement
- Données : dataset synthétique (Kaggle, 10 000 observations)

## Auteur

Mertus YANOGO
