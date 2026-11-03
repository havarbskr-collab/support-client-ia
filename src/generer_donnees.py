"""
Génère un jeu de données simulé de tickets de support client pour TechDistrib,
un distributeur informatique B2B fictif.

Les données sont entièrement synthétiques : aucune entreprise ni aucun client réel.
Elles reproduisent des situations courantes d'un service client B2B :
retards de livraison, erreurs de facturation, retours SAV, tickets mal orientés...

Usage :
    python src/generer_donnees.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

GRAINE = 42
NB_TICKETS = 6000
DEBUT = pd.Timestamp("2025-11-01")
FIN = pd.Timestamp("2026-10-31")

RACINE = Path(__file__).resolve().parents[1]
SORTIE = RACINE / "data" / "tickets_support.csv"

# Catégorie du ticket -> équipe qui doit le traiter
EQUIPES = {
    "Livraison": "Logistique",
    "Retour et SAV": "SAV",
    "Facturation": "Comptabilité clients",
    "Commande et devis": "Administration des ventes",
    "Technique": "Support technique",
    "Accès au portail": "Support technique",
}

# Fréquence de chaque catégorie
POIDS_CATEGORIES = {
    "Livraison": 0.27,
    "Commande et devis": 0.20,
    "Facturation": 0.17,
    "Retour et SAV": 0.15,
    "Technique": 0.13,
    "Accès au portail": 0.08,
}

# Délai moyen de résolution par catégorie (en heures ouvrées)
RESOLUTION_MOYENNE = {
    "Livraison": 20,
    "Commande et devis": 10,
    "Facturation": 26,
    "Retour et SAV": 44,
    "Technique": 30,
    "Accès au portail": 5,
}

PRODUITS = [
    "écrans Dell 27 pouces", "PC portables Lenovo ThinkPad", "imprimantes HP LaserJet",
    "switchs Cisco 24 ports", "stations d'accueil USB-C", "MacBook Air", "casques Jabra",
    "serveurs Dell PowerEdge", "disques SSD Samsung", "bornes Wi-Fi Ubiquiti",
    "claviers Logitech", "onduleurs APC", "tablettes Samsung", "vidéoprojecteurs Epson",
]
LOGICIELS = ["licences Microsoft 365", "licences Windows 11 Pro", "licences antivirus ESET", "licences Adobe Acrobat"]

MODELES = {
    "Livraison": [
        "Bonjour, notre commande {cmd} de {p} devait arriver le {j} et nous n'avons toujours rien reçu.",
        "Le colis de la commande {cmd} est indiqué livré mais personne ne l'a réceptionné chez nous.",
        "Le suivi transporteur de {cmd} est bloqué depuis {n} jours, pouvez-vous relancer ?",
        "Livraison incomplète pour {cmd} : il manque {n} {p} sur le bon de livraison.",
        "Pouvez-vous nous donner une nouvelle date de livraison pour {cmd} ?",
        "Le carton est arrivé abîmé et la commande {cmd} ({p}) a été livrée en retard.",
        "Nous attendons toujours l'expédition de {cmd}, le délai annoncé est dépassé.",
        "Le livreur est passé en dehors des horaires d'ouverture, merci de reprogrammer la livraison de {cmd}.",
    ],
    "Commande et devis": [
        "Bonjour, pouvez-vous nous envoyer un devis pour {n} {p} ?",
        "Nous souhaitons modifier la commande {cmd} et ajouter {n} {p}.",
        "Merci d'annuler la commande {cmd}, nous avons commandé en double.",
        "Quel est le délai de disponibilité des {p} ? Nous devons équiper {n} postes.",
        "Le devis {dev} a expiré, pouvez-vous le renouveler aux mêmes conditions ?",
        "Pouvons-nous bénéficier d'un tarif dégressif pour {n} {p} ?",
        "Je voudrais confirmer la commande issue du devis {dev}.",
        "Les {p} sont en rupture sur le site, avez-vous une alternative à proposer ?",
    ],
    "Facturation": [
        "La facture {fac} ne correspond pas au devis, le montant est plus élevé.",
        "Nous avons été facturés deux fois pour la commande {cmd} (factures {fac} et suivante).",
        "Merci de nous envoyer un avoir pour le retour de {p} du mois dernier.",
        "Notre numéro de TVA est erroné sur la facture {fac}, pouvez-vous la corriger ?",
        "Nous avons réglé la facture {fac} mais nous recevons encore des relances de paiement.",
        "Pouvez-vous nous transmettre le duplicata de la facture {fac} ?",
        "L'adresse de facturation de notre société a changé, la facture {fac} est à refaire.",
        "Le prix des {p} sur la facture ne correspond pas à notre contrat cadre.",
    ],
    "Retour et SAV": [
        "Un appareil de la commande {cmd} ({p}) ne s'allume plus, nous souhaitons un échange.",
        "Erreur de préparation sur la commande {cmd} : nous avons reçu des {p} au lieu de la référence prévue, comment faire le retour ?",
        "Nous avons besoin d'un numéro de retour (RMA) pour {n} {p} en panne.",
        "Les {p} sont sous garantie mais leur réparation n'avance pas.",
        "Un appareil de la commande {cmd} ({p}) est arrivé cassé, que faire ?",
        "Nous attendons depuis {n} semaines le remplacement des {p} partis en SAV.",
        "Nous voulons retourner {n} {p} qui ne correspondent pas à notre besoin.",
        "Panne récurrente sur les {p} de notre dernière commande, demande de prise en charge sous garantie.",
    ],
    "Technique": [
        "Les {p} ne se connectent pas à notre réseau, avez-vous une procédure de configuration ?",
        "Problème d'activation des {lg}, la clé produit est refusée.",
        "Les pilotes des {p} ne sont pas compatibles avec notre version de Windows.",
        "Nous n'arrivons pas à configurer les {p}, la documentation est incomplète.",
        "Mise à jour du firmware impossible sur les {p}, message d'erreur à l'installation.",
        "Les {p} redémarrent en boucle après la mise à jour.",
        "Besoin d'aide pour le déploiement des {p} sur nos {n} sites.",
        "Les {p} fonctionnent au ralenti depuis l'installation, pouvez-vous nous aider au diagnostic ?",
    ],
    "Accès au portail": [
        "Je n'arrive plus à me connecter au portail client, mot de passe refusé.",
        "Merci de créer un accès au portail pour un nouveau collègue des achats.",
        "Mon compte sur l'espace client est bloqué après plusieurs tentatives.",
        "Je ne reçois pas l'e-mail de réinitialisation du mot de passe de l'espace client.",
        "Le portail affiche une erreur quand je veux télécharger mes factures.",
        "Pouvez-vous supprimer l'accès au portail d'un collaborateur qui a quitté l'entreprise ?",
        "Impossible de valider mon panier sur le portail, la page reste blanche.",
        "Les droits de mon compte portail ne me permettent pas de voir les commandes de mon service.",
    ],
}

# Phrases qui ajoutent une notion d'urgence
URGENCE_FORTE = [
    " C'est urgent, notre production est à l'arrêt.",
    " URGENT : nos équipes ne peuvent pas travailler.",
    " Merci de traiter en priorité, c'est bloquant pour tout le service.",
    " Situation critique, un client final attend.",
]
URGENCE_MOYENNE = [
    " Nous en avons besoin rapidement.",
    " Merci de revenir vers nous avant la fin de semaine.",
    " Cela commence à gêner nos utilisateurs.",
]
URGENCE_FAIBLE = [
    " Pas d'urgence.",
    " Quand vous aurez un moment.",
    " Rien de bloquant, c'est pour information.",
]
POLITESSES = ["", "", " Merci d'avance.", " Cordialement.", " Bonne journée."]

# Messages vagues : le texte ne dit pas de quoi il s'agit vraiment.
# Seul un échange avec le client permet de savoir quelle équipe doit intervenir.
MESSAGES_VAGUES = [
    "Bonjour, j'ai un problème avec la commande {cmd}, pouvez-vous me rappeler ?",
    "Merci de me contacter au sujet de notre dossier {cmd}.",
    "Suite à notre échange téléphonique, je reviens vers vous concernant {cmd}.",
    "Nous avons un souci avec les {p} de notre commande, qui peut m'aider ?",
    "Je relance ma demande de la semaine dernière, toujours sans réponse.",
    "Pouvez-vous faire le point sur notre dossier en cours ?",
    "Rien ne va avec notre dernière commande, merci de nous rappeler rapidement.",
    "Bonjour, pouvez-vous transmettre ma demande au bon service ?",
]

PROPORTION_VAGUES = 0.09
PROPORTION_MELANGES = 0.15


def remplir(rng, modele):
    return modele.format(
        cmd=f"CMD-{rng.integers(10000, 99999)}",
        p=rng.choice(PRODUITS),
        n=int(rng.integers(2, 40)),
        j=f"{rng.integers(1, 28)}/{rng.integers(1, 12):02d}",
        fac=f"FAC-{rng.integers(100000, 999999)}",
        dev=f"DEV-{rng.integers(1000, 9999)}",
        lg=rng.choice(LOGICIELS),
    )


def generer_texte(rng, categorie, priorite):
    """Renvoie le message du client et son type : clair, mélangé (deux sujets) ou vague."""
    tirage = rng.random()
    if tirage < PROPORTION_VAGUES:
        type_message = "vague"
        texte = remplir(rng, rng.choice(MESSAGES_VAGUES))
    else:
        texte = remplir(rng, rng.choice(MODELES[categorie]))
        type_message = "clair"
        if tirage < PROPORTION_VAGUES + PROPORTION_MELANGES:
            # Le client évoque aussi un second sujet qui relève d'une autre équipe
            type_message = "mélangé"
            autre = rng.choice([c for c in MODELES if c != categorie])
            second = remplir(rng, rng.choice(MODELES[autre])).removeprefix("Bonjour, ")
            texte += " Par ailleurs, " + second[0].lower() + second[1:]

    if priorite == "Urgente":
        texte += rng.choice(URGENCE_FORTE)
    elif priorite == "Haute" and rng.random() < 0.7:
        texte += rng.choice(URGENCE_MOYENNE)
    elif priorite == "Basse" and rng.random() < 0.5:
        texte += rng.choice(URGENCE_FAIBLE)
    elif priorite == "Normale" and rng.random() < 0.05:
        texte += rng.choice(URGENCE_MOYENNE)
    texte += rng.choice(POLITESSES)
    # Une partie des clients écrit sans majuscules
    if rng.random() < 0.10:
        texte = texte.lower()
    return texte, type_message


def heures_ouvrees(rng, moyenne, taille=None):
    """Durées positives et asymétriques (loi log-normale), en heures ouvrées."""
    sigma = 0.8
    mu = np.log(moyenne) - sigma**2 / 2
    return rng.lognormal(mu, sigma, taille)


def main():
    rng = np.random.default_rng(GRAINE)

    # Dates de création : plus de tickets le lundi et en fin de trimestre
    jours = pd.bdate_range(DEBUT, FIN)
    poids_jours = np.ones(len(jours))
    poids_jours[jours.dayofweek == 0] *= 1.6
    poids_jours[jours.month.isin([3, 6, 9, 12]) & (jours.day >= 20)] *= 1.4
    poids_jours /= poids_jours.sum()
    dates = rng.choice(jours, size=NB_TICKETS, p=poids_jours)
    heures = rng.integers(8 * 60, 18 * 60, NB_TICKETS)
    dates = pd.to_datetime(dates) + pd.to_timedelta(heures, unit="m")

    categories = rng.choice(
        list(POIDS_CATEGORIES), size=NB_TICKETS, p=list(POIDS_CATEGORIES.values())
    )
    priorites = rng.choice(
        ["Basse", "Normale", "Haute", "Urgente"], size=NB_TICKETS, p=[0.18, 0.52, 0.21, 0.09]
    )
    canaux = rng.choice(["E-mail", "Portail client", "Téléphone"], size=NB_TICKETS, p=[0.55, 0.30, 0.15])
    types_clients = rng.choice(
        ["PME", "Grand compte", "Secteur public"], size=NB_TICKETS, p=[0.55, 0.30, 0.15]
    )

    lignes = []
    for i in range(NB_TICKETS):
        cat, prio = categories[i], priorites[i]
        texte, type_message = generer_texte(rng, cat, prio)
        equipe = EQUIPES[cat]

        # Tri manuel actuel : un agent lit le ticket et l'oriente vers une équipe.
        # Les messages mélangés ou vagues sont plus souvent mal orientés.
        proba_erreur = {"clair": 0.10, "mélangé": 0.30, "vague": 0.45}[type_message]
        mal_oriente = rng.random() < proba_erreur
        if mal_oriente:
            autres = [e for e in set(EQUIPES.values()) if e != equipe]
            equipe_initiale = rng.choice(sorted(autres))
        else:
            equipe_initiale = equipe

        temps_tri_min = round(float(np.clip(rng.lognormal(np.log(3.5), 0.45), 1, 20)), 1)

        # Délai de première réponse : dépend de l'urgence, du jour et d'une éventuelle réorientation
        base_reponse = {"Urgente": 1.0, "Haute": 2.5, "Normale": 5.0, "Basse": 8.0}[prio]
        if dates[i].dayofweek == 0:
            base_reponse *= 1.4
        reponse = heures_ouvrees(rng, base_reponse)
        retard_reorientation = heures_ouvrees(rng, 7.0) if mal_oriente else 0.0
        reponse += retard_reorientation

        resolution = reponse + heures_ouvrees(rng, RESOLUTION_MOYENNE[cat])
        if prio == "Urgente":
            resolution *= 0.6

        # Satisfaction : baisse avec le délai de résolution et après une réorientation
        score = 4.6 - 0.025 * resolution - (0.5 if mal_oriente else 0) + rng.normal(0, 0.7)
        satisfaction = int(np.clip(round(score), 1, 5))

        lignes.append(
            {
                "id_ticket": f"T{100000 + i}",
                "date_creation": dates[i].strftime("%Y-%m-%d %H:%M"),
                "canal": canaux[i],
                "type_client": types_clients[i],
                "message": texte,
                "categorie": cat,
                "priorite": prio,
                "equipe_cible": equipe,
                "equipe_initiale": equipe_initiale,
                "reoriente": int(mal_oriente),
                "temps_tri_manuel_min": temps_tri_min,
                "delai_premiere_reponse_h": round(float(reponse), 1),
                "delai_resolution_h": round(float(resolution), 1),
                "satisfaction": satisfaction,
            }
        )

    df = pd.DataFrame(lignes).sort_values("date_creation").reset_index(drop=True)
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SORTIE, index=False, encoding="utf-8")
    print(f"{len(df)} tickets générés -> {SORTIE.relative_to(RACINE)}")


if __name__ == "__main__":
    main()
