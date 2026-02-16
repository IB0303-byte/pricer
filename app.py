import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Calculateur de Prix des Titres - Circulaire 02/04",
    page_layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("📊 Calculateur de Prix des Titres")
st.markdown("Conformément à la circulaire n°02/04 du CDVM - Annexe: Formules d'actualisation")

# Fonctions de calcul
def is_leap_year(year):
    """Vérifie si une année est bissextile"""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def jours_dans_annee(date_eval):
    """Retourne le nombre de jours dans l'année (365 ou 366)"""
    return 366 if is_leap_year(date_eval.year) else 365

def calcul_maturite_residuelle(date_echeance, date_evaluation):
    """Calcule la maturité résiduelle en jours"""
    return (date_echeance - date_evaluation).days

def calcul_prix_titre_court_terme(N, tf, tr, Mi, Mr):
    """
    Formule (1): Titres de créances de maturité initiale ≤ 1 an
    P = N * (1 + tf * Mi/360) / (1 + tr * Mr/360)
    """
    numerateur = 1 + tf * Mi / 360
    denominateur = 1 + tr * Mr / 360
    return N * numerateur / denominateur

def calcul_prix_titre_maturite_residuelle_courte(N, tf, tr, Mr, A):
    """
    Formule (2): Titres de maturité initiale > 1 an, maturité résiduelle < 1 an
    P = N * (1 + tf) / (1 + tr * Mr/360)
    """
    numerateur = 1 + tf
    denominateur = 1 + tr * Mr / 360
    return N * numerateur / denominateur

def calcul_prix_ligne_postérieure_un_flux(N, tf, tr, Mi, Mr, A):
    """
    Formule (3): Ligne postérieure à un seul flux
    P = N * (1 + tf * Mi/A) / (1 + tr * Mr/360)
    """
    numerateur = 1 + tf * Mi / A
    denominateur = 1 + tr * Mr / 360
    return N * numerateur / denominateur

def calcul_prix_ligne_normale(N, tf, tr, n, nj, A):
    """
    Formule (4.1): Ligne normale avec plusieurs coupons
    P = N/(1+tr)^(nj/A) * [somme(tf/(1+tr)^(i-1)) + 1/(1+tr)^(n-1)]
    """
    facteur_actualisation = (1 + tr) ** (nj / A)
    somme_coupons = 0
    for i in range(1, n + 1):
        somme_coupons += tf / ((1 + tr) ** (i - 1))
    somme_coupons += 1 / ((1 + tr) ** (n - 1))
    return N * somme_coupons / facteur_actualisation

def calcul_prix_titre(N, tf, tr, date_emission, date_echeance, date_evaluation, 
                     type_titre="Etat", spread=0.0, nb_coupons=None):
    """
    Calcule le prix d'un titre selon les formules de la circulaire
    """
    # Calcul du taux d'actualisation avec spread si nécessaire
    taux_actu = tr + spread if type_titre != "Etat" else tr
    
    # Calcul des maturités
    Mi = (date_echeance - date_emission).days
    Mr = (date_echeance - date_evaluation).days
    A = jours_dans_annee(date_evaluation)
    
    # Détermination du type de formule à utiliser
    if Mi <= 365:
        # Formule (1): Maturité initiale ≤ 1 an
        return calcul_prix_titre_court_terme(N, tf, taux_actu, Mi, Mr)
    
    elif Mr <= 365:
        # Formule (2) ou (3): Maturité résiduelle < 1 an
        # Vérifier si c'est une ligne postérieure à un seul flux
        if nb_coupons == 1:
            return calcul_prix_ligne_postérieure_un_flux(N, tf, taux_actu, Mi, Mr, A)
        else:
            return calcul_prix_titre_maturite_residuelle_courte(N, tf, taux_actu, Mr, A)
    
    else:
        # Formules (4.x): Maturité résiduelle > 1 an
        nj = (date_echeance - date_evaluation).days
        
        # Détermination du nombre de coupons restants
        if nb_coupons is None:
            # Calcul approximatif du nombre de coupons
            annees_restantes = Mr / 365
            nb_coupons = int(np.ceil(annees_restantes))
        
        return calcul_prix_ligne_normale(N, tf, taux_actu, nb_coupons, nj, A)

# Sidebar pour les paramètres globaux
with st.sidebar:
    st.header("⚙️ Paramètres")
    
    # Option pour uploader un fichier ou saisir manuellement
    input_method = st.radio(
        "Méthode de saisie",
        ["Saisie manuelle", "Import fichier Excel/CSV"]
    )
    
    # Courbe des taux BAM (simulée)
    st.subheader("📈 Courbe des taux BAM")
    taux_bam_3m = st.number_input("Taux 3 mois (%)", min_value=0.0, max_value=100.0, value=3.0, step=0.1) / 100
    taux_bam_6m = st.number_input("Taux 6 mois (%)", min_value=0.0, max_value=100.0, value=3.2, step=0.1) / 100
    taux_bam_1a = st.number_input("Taux 1 an (%)", min_value=0.0, max_value=100.0, value=3.5, step=0.1) / 100
    taux_bam_2a = st.number_input("Taux 2 ans (%)", min_value=0.0, max_value=100.0, value=3.8, step=0.1) / 100
    taux_bam_3a = st.number_input("Taux 3 ans (%)", min_value=0.0, max_value=100.0, value=4.0, step=0.1) / 100
    taux_bam_5a = st.number_input("Taux 5 ans (%)", min_value=0.0, max_value=100.0, value=4.3, step=0.1) / 100
    taux_bam_10a = st.number_input("Taux 10 ans (%)", min_value=0.0, max_value=100.0, value=4.8, step=0.1) / 100

# Fonction pour obtenir le taux BAM selon la maturité
def get_taux_bam(maturite_jours, taux_dict):
    """Interpolation linéaire des taux BAM"""
    maturites = [90, 180, 365, 730, 1095, 1825, 3650]  # en jours
    taux = [taux_dict['3m'], taux_dict['6m'], taux_dict['1a'], 
            taux_dict['2a'], taux_dict['3a'], taux_dict['5a'], taux_dict['10a']]
    
    if maturite_jours <= maturites[0]:
        return taux[0]
    elif maturite_jours >= maturites[-1]:
        return taux[-1]
    else:
        # Interpolation linéaire
        for i in range(len(maturites) - 1):
            if maturites[i] <= maturite_jours <= maturites[i+1]:
                ratio = (maturite_jours - maturites[i]) / (maturites[i+1] - maturites[i])
                return taux[i] + ratio * (taux[i+1] - taux[i])

# Interface principale
if input_method == "Saisie manuelle":
    st.header("📝 Saisie manuelle des titres")
    
    col1, col2 = st.columns(2)
    
    with col1:
        isin = st.text_input("Code ISIN", value="MA0000012345")
        date_evaluation = st.date_input("Date d'évaluation", value=date.today())
        date_emission = st.date_input("Date d'émission", value=date.today() - relativedelta(years=1))
        date_echeance = st.date_input("Date d'échéance", value=date.today() + relativedelta(years=5))
        nominal = st.number_input("Nominal (DH)", min_value=1000.0, value=100000.0, step=1000.0)
    
    with col2:
        taux_facial = st.number_input("Taux facial (%)", min_value=0.0, max_value=100.0, value=4.0, step=0.1) / 100
        type_titre = st.selectbox("Type d'émetteur", ["Etat", "Privé garanti par l'Etat", "Privé"])
        spread = 0.0
        if type_titre != "Etat":
            spread = st.number_input("Spread / Prime de risque (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1) / 100
        nb_coupons = st.number_input("Nombre de coupons restants", min_value=1, max_value=50, value=5, step=1)
    
    # Calcul de la maturité résiduelle
    Mr = (date_echeance - date_evaluation).days
    Mi = (date_echeance - date_emission).days
    
    st.info(f"📊 Maturité initiale: {Mi} jours | Maturité résiduelle: {Mr} jours")
    
    # Détermination du taux BAM
    taux_dict = {
        '3m': taux_bam_3m,
        '6m': taux_bam_6m,
        '1a': taux_bam_1a,
        '2a': taux_bam_2a,
        '3a': taux_bam_3a,
        '5a': taux_bam_5a,
        '10a': taux_bam_10a
    }
    
    tr = get_taux_bam(Mr, taux_dict)
    
    st.write(f"📌 Taux de référence BAM appliqué: {tr*100:.2f}%")
    
    if st.button("💰 Calculer le prix", type="primary"):
        prix = calcul_prix_titre(
            nominal, taux_facial, tr, date_emission, 
            date_echeance, date_evaluation, type_titre, 
            spread, nb_coupons
        )
        
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            st.metric("Prix calculé", f"{prix:,.2f} DH")
        with col_result2:
            st.metric("Valeur nominale", f"{nominal:,.2f} DH")
        with col_result3:
            st.metric("Différence", f"{prix - nominal:,.2f} DH")

else:
    st.header("📁 Import de fichier")
    
    uploaded_file = st.file_uploader("Choisir un fichier Excel ou CSV", type=['xlsx', 'csv'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.subheader("Aperçu des données importées")
            st.dataframe(df.head())
            
            # Mapping des colonnes
            st.subheader("Configuration des colonnes")
            col_mapping = {}
            
            columns = df.columns.tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                col_mapping['ISIN'] = st.selectbox("Colonne ISIN", columns, key='isin_col')
                col_mapping['date_emission'] = st.selectbox("Colonne Date émission", columns, key='date_em_col')
                col_mapping['date_echeance'] = st.selectbox("Colonne Date échéance", columns, key='date_ech_col')
                col_mapping['nominal'] = st.selectbox("Colonne Nominal", columns, key='nominal_col')
            
            with col2:
                col_mapping['taux_facial'] = st.selectbox("Colonne Taux facial", columns, key='tf_col')
                col_mapping['type_titre'] = st.selectbox("Colonne Type émetteur", columns, key='type_col')
                col_mapping['spread'] = st.selectbox("Colonne Spread (si applicable)", columns, key='spread_col')
            
            date_evaluation = st.date_input("Date d'évaluation pour tous les titres", value=date.today())
            
            if st.button("🚀 Calculer tous les prix"):
                taux_dict = {
                    '3m': taux_bam_3m,
                    '6m': taux_bam_6m,
                    '1a': taux_bam_1a,
                    '2a': taux_bam_2a,
                    '3a': taux_bam_3a,
                    '5a': taux_bam_5a,
                    '10a': taux_bam_10a
                }
                
                results = []
                
                for idx, row in df.iterrows():
                    try:
                        # Conversion des dates
                        date_emission = pd.to_datetime(row[col_mapping['date_emission']])
                        date_echeance = pd.to_datetime(row[col_mapping['date_echeance']])
                        
                        Mr = (date_echeance - date_evaluation).days
                        tr = get_taux_bam(Mr, taux_dict)
                        
                        prix = calcul_prix_titre(
                            N=float(row[col_mapping['nominal']]),
                            tf=float(row[col_mapping['taux_facial']]) / 100 if float(row[col_mapping['taux_facial']]) > 1 else float(row[col_mapping['taux_facial']]),
                            tr=tr,
                            date_emission=date_emission,
                            date_echeance=date_echeance,
                            date_evaluation=date_evaluation,
                            type_titre=row[col_mapping['type_titre']],
                            spread=float(row[col_mapping['spread']]) / 100 if pd.notna(row[col_mapping['spread']]) else 0.0,
                            nb_coupons=None
                        )
                        
                        results.append({
                            'ISIN': row[col_mapping['ISIN']],
                            'Prix calculé': prix,
                            'Nominal': float(row[col_mapping['nominal']]),
                            'Différence': prix - float(row[col_mapping['nominal']]),
                            'Maturité résiduelle (jours)': Mr,
                            'Taux BAM appliqué': tr
                        })
                    except Exception as e:
                        st.error(f"Erreur sur la ligne {idx+1}: {str(e)}")
                
                if results:
                    results_df = pd.DataFrame(results)
                    
                    st.subheader("📊 Résultats des calculs")
                    st.dataframe(results_df.style.format({
                        'Prix calculé': '{:,.2f}',
                        'Nominal': '{:,.2f}',
                        'Différence': '{:,.2f}',
                        'Taux BAM appliqué': '{:.2%}'
                    }))
                    
                    # Graphiques
                    col_g1, col_g2 = st.columns(2)
                    
                    with col_g1:
                        fig = px.bar(results_df, x='ISIN', y='Prix calculé', 
                                   title="Prix par titre", color='ISIN')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_g2:
                        fig2 = px.scatter(results_df, x='Maturité résiduelle (jours)', 
                                        y='Prix calculé', text='ISIN',
                                        title="Relation Prix vs Maturité")
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Export
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger les résultats (CSV)",
                        data=csv,
                        file_name=f"resultats_prix_titres_{date.today()}.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier: {str(e)}")

# Section d'aide et documentation
with st.expander("📖 Aide et documentation"):
    st.markdown("""
    ### Formules utilisées (Circulaire n°02/04 - Annexe)
    
    **1. Titres de maturité initiale ≤ 1 an:**
    - P = N × (1 + tf × Mi/360) / (1 + tr × Mr/360)
    
    **2. Titres de maturité initiale > 1 an, maturité résiduelle < 1 an:**
    - P = N × (1 + tf) / (1 + tr × Mr/360)
    
    **3. Ligne postérieure à un seul flux:**
    - P = N × (1 + tf × Mi/A) / (1 + tr × Mr/360)
    
    **4. Ligne normale (plusieurs coupons):**
    - P = N/(1+tr)^(nj/A) × [∑(tf/(1+tr)^(i-1)) + 1/(1+tr)^(n-1)]
    
    **Légende:**
    - N: Nominal
    - tf: Taux facial
    - tr: Taux de rendement (taux BAM + spread le cas échéant)
    - Mi: Maturité initiale (jours)
    - Mr: Maturité résiduelle (jours)
    - A: 365 ou 366 jours selon l'année
    - nj: Jours jusqu'au prochain coupon
    
    ### Règles importantes:
    - Les titres d'État: taux BAM uniquement
    - Titres privés garantis par l'État: taux BAM + prime de liquidité
    - Autres titres privés: taux BAM + prime de risque
    - Interpolation linéaire pour les maturités non cotées
    """)

# Pied de page
st.markdown("---")
st.markdown("© 2024 - Calculateur conforme à la circulaire CDVM n°02/04")
