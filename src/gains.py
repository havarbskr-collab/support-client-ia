"""
Estimation des gains du tri automatique, sur la période de test (tickets jamais vus par le modèle).

Hypothèses, toutes mesurées dans les données sauf indication contraire :
1. Un ticket trié automatiquement n'a plus besoin d'être lu par un agent pour être orienté :
   on économise son temps de tri manuel.
2. Les tickets dont le modèle n'est pas sûr (confiance < seuil) restent triés par un agent,
   avec le même taux d'erreur qu'aujourd'hui.
3. Chaque erreur d'orientation, humaine ou automatique, entraîne une réorientation
   qui retarde la première réponse du retard moyen observé dans les données.

Usage :
    python src/gains.py
"""

import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from analyse import charger_tickets  # noqa: E402
from modele_tri import decouper, entrainer, predire  # noqa: E402


def retard_moyen_reorientation(df):
    """Retard moyen de première réponse causé par une réorientation, à priorité égale."""
    moyennes = df.groupby(["priorite", "reoriente"])["delai_premiere_reponse_h"].mean().unstack()
    ecarts = moyennes[1] - moyennes[0]
    poids = df["priorite"].value_counts(normalize=True)
    return float((ecarts * poids).sum())


def simuler(test, predictions, retard_h):
    t = test.join(predictions)
    auto = t["tri_automatique"]

    # Situation actuelle
    erreurs_avant = t["reoriente"].astype(bool)

    # Avec l'IA : erreurs du modèle sur les tickets triés seuls + erreurs humaines sur les autres
    erreurs_ia = auto & (t["equipe_predite"] != t["equipe_cible"])
    erreurs_humaines = ~auto & t["reoriente"].astype(bool)
    erreurs_apres = erreurs_ia | erreurs_humaines

    # Délai de première réponse estimé après : on retire le retard des réorientations évitées
    # et on l'ajoute pour les nouvelles erreurs du modèle
    reponse_apres = t["delai_premiere_reponse_h"].copy()
    reponse_apres[erreurs_avant & ~erreurs_apres] -= retard_h
    reponse_apres[~erreurs_avant & erreurs_apres] += retard_h
    reponse_apres = reponse_apres.clip(lower=0.1)

    nb_mois = t["mois"].nunique()
    return {
        "nb_mois": nb_mois,
        "tickets_par_mois": len(t) / nb_mois,
        "part_tri_auto": auto.mean(),
        "heures_tri_avant": t["temps_tri_manuel_min"].sum() / 60 / nb_mois,
        "heures_tri_apres": t.loc[~auto, "temps_tri_manuel_min"].sum() / 60 / nb_mois,
        "erreurs_avant": erreurs_avant.sum() / nb_mois,
        "erreurs_apres": erreurs_apres.sum() / nb_mois,
        "taux_erreur_avant": erreurs_avant.mean(),
        "taux_erreur_apres": erreurs_apres.mean(),
        "heures_attente_evitees": (erreurs_avant.sum() - erreurs_apres.sum()) * retard_h / nb_mois,
        "sla_avant": t["sla_respecte"].mean(),
        "sla_apres": (reponse_apres <= t["sla_h"]).mean(),
    }


def tableau_gains(r):
    lignes = [
        ("Temps de tri par mois (heures)", r["heures_tri_avant"], r["heures_tri_apres"]),
        ("Tickets mal orientés par mois", r["erreurs_avant"], r["erreurs_apres"]),
        ("Taux de tickets mal orientés (%)", 100 * r["taux_erreur_avant"], 100 * r["taux_erreur_apres"]),
        ("SLA de 1re réponse respecté (%)", 100 * r["sla_avant"], 100 * r["sla_apres"]),
    ]
    t = pd.DataFrame(lignes, columns=["Indicateur", "Aujourd'hui", "Avec le tri IA"]).set_index("Indicateur")
    t["Écart"] = t["Avec le tri IA"] - t["Aujourd'hui"]
    return t.round(1)


if __name__ == "__main__":
    tickets = charger_tickets()
    _, test = decouper(tickets)
    modeles = entrainer(tickets)
    retard = retard_moyen_reorientation(tickets)
    res = simuler(test, predire(modeles, test["message"]), retard)

    print(f"Retard moyen causé par une réorientation : {retard:.1f} h")
    print(f"Tickets triés automatiquement : {res['part_tri_auto']:.0%}\n")
    print(tableau_gains(res).to_string())
    print(f"\nHeures d'attente client évitées par mois : {res['heures_attente_evitees']:.0f} h")
