"""
Indicateurs du service client et reporting Excel.

Usage :
    python src/analyse.py   -> crée reports/reporting_support.xlsx
"""

from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
DONNEES = RACINE / "data" / "tickets_support.csv"
RAPPORTS = RACINE / "reports"

# Engagement de délai de première réponse (SLA), en heures ouvrées
SLA_REPONSE_H = {"Urgente": 2, "Haute": 4, "Normale": 8, "Basse": 16}
ORDRE_PRIORITES = ["Urgente", "Haute", "Normale", "Basse"]


def charger_tickets(chemin=DONNEES):
    df = pd.read_csv(chemin, parse_dates=["date_creation"])
    df["mois"] = df["date_creation"].dt.to_period("M").astype(str)
    df["jour_semaine"] = df["date_creation"].dt.day_name(locale=None)
    df["sla_h"] = df["priorite"].map(SLA_REPONSE_H)
    df["sla_respecte"] = df["delai_premiere_reponse_h"] <= df["sla_h"]
    return df


def kpi_globaux(df):
    return pd.Series(
        {
            "Tickets": len(df),
            "Tickets par mois (moyenne)": round(len(df) / df["mois"].nunique()),
            "Délai médian de 1re réponse (h)": df["delai_premiere_reponse_h"].median(),
            "Délai médian de résolution (h)": df["delai_resolution_h"].median(),
            "SLA de 1re réponse respecté (%)": round(100 * df["sla_respecte"].mean(), 1),
            "Tickets mal orientés (%)": round(100 * df["reoriente"].mean(), 1),
            "Satisfaction moyenne (/5)": round(df["satisfaction"].mean(), 2),
            "Temps de tri manuel moyen (min)": round(df["temps_tri_manuel_min"].mean(), 1),
        }
    )


def kpi_par(df, colonne):
    t = df.groupby(colonne).agg(
        tickets=("id_ticket", "count"),
        reponse_mediane_h=("delai_premiere_reponse_h", "median"),
        resolution_mediane_h=("delai_resolution_h", "median"),
        sla_respecte_pct=("sla_respecte", "mean"),
        mal_orientes_pct=("reoriente", "mean"),
        satisfaction=("satisfaction", "mean"),
    )
    t["sla_respecte_pct"] = (100 * t["sla_respecte_pct"]).round(1)
    t["mal_orientes_pct"] = (100 * t["mal_orientes_pct"]).round(1)
    t["satisfaction"] = t["satisfaction"].round(2)
    return t.sort_values("tickets", ascending=False)


def effet_reorientation(df):
    """Compare les tickets bien orientés et mal orientés dès le départ."""
    t = df.groupby("reoriente").agg(
        tickets=("id_ticket", "count"),
        reponse_moyenne_h=("delai_premiere_reponse_h", "mean"),
        resolution_moyenne_h=("delai_resolution_h", "mean"),
        sla_respecte_pct=("sla_respecte", "mean"),
        satisfaction=("satisfaction", "mean"),
    )
    t.index = t.index.map({0: "Bien orienté", 1: "Mal orienté"})
    t["sla_respecte_pct"] = (100 * t["sla_respecte_pct"]).round(1)
    return t.round(2)


def creer_reporting_excel(df, chemin=RAPPORTS / "reporting_support.xlsx"):
    """Reporting prêt à partager : synthèse, détail par équipe et par mois, tickets hors délai."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    hors_delai = df.loc[
        ~df["sla_respecte"],
        ["id_ticket", "date_creation", "categorie", "priorite", "equipe_cible",
         "delai_premiere_reponse_h", "sla_h", "reoriente"],
    ].sort_values("delai_premiere_reponse_h", ascending=False)

    with pd.ExcelWriter(chemin, engine="openpyxl") as xls:
        kpi_globaux(df).rename("Valeur").to_frame().to_excel(xls, sheet_name="Synthèse")
        kpi_par(df, "equipe_cible").to_excel(xls, sheet_name="Par équipe")
        kpi_par(df, "categorie").to_excel(xls, sheet_name="Par catégorie")
        kpi_par(df, "mois").sort_index().to_excel(xls, sheet_name="Par mois")
        hors_delai.to_excel(xls, sheet_name="Tickets hors SLA", index=False)

        for feuille in xls.book.worksheets:
            for colonne in feuille.columns:
                largeur = max(len(str(c.value)) if c.value is not None else 0 for c in colonne)
                feuille.column_dimensions[colonne[0].column_letter].width = min(largeur + 2, 45)
    return chemin


if __name__ == "__main__":
    tickets = charger_tickets()
    print(kpi_globaux(tickets).to_string())
    print(f"\nReporting créé : {creer_reporting_excel(tickets).relative_to(RACINE)}")
