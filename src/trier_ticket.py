"""
Trie un nouveau ticket : équipe à qui l'envoyer, niveau d'urgence et confiance du modèle.
Si le modèle n'est pas assez sûr, le ticket est laissé à un agent.

Usage :
    python src/trier_ticket.py "Notre commande CMD-12345 n'est toujours pas livrée, c'est urgent"
"""

import sys
from pathlib import Path

import joblib

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from modele_tri import MODELE, SEUIL_CONFIANCE, entrainer, predire  # noqa: E402


def charger_modele():
    if MODELE.exists():
        return joblib.load(MODELE)
    from analyse import charger_tickets

    print("Premier lancement : entraînement du modèle...")
    modeles = entrainer(charger_tickets())
    MODELE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modeles, MODELE)
    return modeles


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    message = " ".join(sys.argv[1:])
    r = predire(charger_modele(), [message]).iloc[0]

    print(f"Message   : {message}")
    print(f"Équipe    : {r['equipe_predite']} (confiance {r['confiance']:.0%})")
    print(f"Priorité  : {r['priorite_predite']}")
    if r["tri_automatique"]:
        print("Décision  : envoi automatique à l'équipe")
    else:
        print(f"Décision  : confiance inférieure à {SEUIL_CONFIANCE:.0%}, à vérifier par un agent")


if __name__ == "__main__":
    main()
