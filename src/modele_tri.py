"""
Modèle d'IA qui oriente automatiquement chaque ticket vers la bonne équipe
et estime son niveau d'urgence à partir du texte du message.

Méthode : représentation TF-IDF du texte (mots et morceaux de mots)
+ régression logistique. Simple, rapide et explicable.

Usage :
    python src/modele_tri.py   -> entraîne, évalue et enregistre models/modele_tri.joblib
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import FeatureUnion, Pipeline

RACINE = Path(__file__).resolve().parents[1]
MODELE = RACINE / "models" / "modele_tri.joblib"

# Les 9 premiers mois servent à l'apprentissage, les 3 derniers au test :
# on évalue le modèle sur des tickets "futurs", comme en conditions réelles.
DATE_COUPURE = "2026-08-01"

# En dessous de ce niveau de confiance, le ticket est laissé à un agent
SEUIL_CONFIANCE = 0.60


def creer_pipeline():
    texte = FeatureUnion(
        [
            ("mots", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode", lowercase=True)),
            ("caracteres", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, strip_accents="unicode", lowercase=True)),
        ]
    )
    return Pipeline([("texte", texte), ("classifieur", LogisticRegression(max_iter=2000, C=4.0))])


def decouper(df):
    entrainement = df[df["date_creation"] < DATE_COUPURE]
    test = df[df["date_creation"] >= DATE_COUPURE]
    return entrainement, test


def entrainer(df):
    entrainement, _ = decouper(df)
    modeles = {}
    for cible in ["equipe_cible", "priorite"]:
        pipeline = creer_pipeline()
        pipeline.fit(entrainement["message"], entrainement[cible])
        modeles[cible] = pipeline
    return modeles


def predire(modeles, messages):
    """Renvoie l'équipe proposée, la priorité proposée et la confiance du modèle pour chaque message."""
    messages = pd.Series(messages)
    proba_equipe = modeles["equipe_cible"].predict_proba(messages)
    resultat = pd.DataFrame(
        {
            "equipe_predite": modeles["equipe_cible"].classes_[proba_equipe.argmax(axis=1)],
            "confiance": proba_equipe.max(axis=1).round(3),
            "priorite_predite": modeles["priorite"].predict(messages),
        },
        index=messages.index,
    )
    resultat["tri_automatique"] = resultat["confiance"] >= SEUIL_CONFIANCE
    return resultat


def evaluer(modeles, test):
    pred = predire(modeles, test["message"])
    auto = pred["tri_automatique"]
    return {
        "precision_equipe": accuracy_score(test["equipe_cible"], pred["equipe_predite"]),
        "precision_priorite": accuracy_score(test["priorite"], pred["priorite_predite"]),
        "part_tri_automatique": auto.mean(),
        "precision_equipe_tri_auto": accuracy_score(test.loc[auto, "equipe_cible"], pred.loc[auto, "equipe_predite"]),
        "rapport_equipe": classification_report(test["equipe_cible"], pred["equipe_predite"], digits=3),
        "rapport_priorite": classification_report(test["priorite"], pred["priorite_predite"], digits=3),
        "predictions": pred,
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(RACINE / "src"))
    from analyse import charger_tickets

    tickets = charger_tickets()
    entrainement, test = decouper(tickets)
    modeles = entrainer(tickets)
    res = evaluer(modeles, test)

    print(f"Entraînement : {len(entrainement)} tickets | Test : {len(test)} tickets")
    print(f"Bonne équipe trouvée : {res['precision_equipe']:.1%}")
    print(f"Bonne priorité trouvée : {res['precision_priorite']:.1%}")
    print(f"Tickets triés automatiquement (confiance >= {SEUIL_CONFIANCE:.0%}) : {res['part_tri_automatique']:.1%}")
    print(f"Bonne équipe parmi ces tickets : {res['precision_equipe_tri_auto']:.1%}")

    MODELE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modeles, MODELE)
    print(f"Modèle enregistré : {MODELE.relative_to(RACINE)}")
