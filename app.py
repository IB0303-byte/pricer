import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

# Configuration de la page
st.set_page_config(
    page_title="Valorisation des Titres - Circulaire 02/04",
    page_icon="💰",
    layout="wide"
)

# Titre principal
st.title("🏦 Valorisation des Titres de Créances")
st.markdown("**Conforme à la Circulaire CDVM N° 02/04**")
st.markdown("---")

# Fonctions de calcul selon les formules de la circulaire

def calculer_jours_residuels(date_evaluation, date_echeance):
    """Calcule le nombre de jours résiduels"""
    return (date_echeance - date_evaluation).days

def est_annee_bissextile(annee):
    """Vérifie si une année est bissextile"""
    return (annee % 4 == 0 and annee % 100 != 0) or (annee % 400 == 0)

def get_A(date_evaluation):
    """Retourne 366 si année bissextile, 365 sinon"""
    return 366 if est_annee_bissextile(date_evaluation.year) else 365

def calculer_jours_prochain_coupon(date_evaluation, date_echeance, mi_jours):
    """Calcule le nombre de jours jusqu'au prochain coupon"""
    # Pour simplifier, on suppose des coupons annuels
    # Le prochain coupon est à la date anniversaire suivante
    annee_courante = date_evaluation.year
    mois_echeance = date_echeance.month
    jour_echeance = date_echeance.day
    
    # Chercher la prochaine date de coupon
    try:
        prochaine_date_coupon = datetime(annee_courante, mois_echeance, jour_echeance)
    except ValueError:
        # Gérer le cas du 29 février
        prochaine_date_coupon = datetime(annee_courante, mois_echeance, 28)
    
    if prochaine_date_coupon <= date_evaluation:
        try:
            prochaine_date_coupon = datetime(annee_courante + 1, mois_echeance, jour_echeance)
        except ValueError:
            prochaine_date_coupon = datetime(annee_courante + 1, mois_echeance, 28)
    
    return (prochaine_date_coupon - date_evaluation).days

def calculer_nombre_coupons(date_evaluation, date_echeance):
    """Calcule le nombre de coupons restants"""
    annees_restantes = (date_echeance - date_evaluation).days / 365.25
    return max(1, math.ceil(annees_restantes))

# Formule (1) : MI <= 1 an
def formule_1(N, Mi, Mr, tf, tr):
    """
    Titres de maturité initiale <= 365 jours
    P = N × (1 + tf × Mi/360) / (1 + tr × Mr/360)
    """
    P = N * (1 + tf * Mi / 360) / (1 + tr * Mr / 360)
    return P

# Formule (2) : MI > 1 an et MR <= 1 an (ligne normale)
def formule_2(N, tf, tr, Mr):
    """
    MI > 1 an, MR <= 365 jours (ligne normale)
    P = N × (1 + tf) / (1 + tr × Mr/360)
    """
    P = N * (1 + tf) / (1 + tr * Mr / 360)
    return P

# Formule (3) : MI > 1 an et MR <= 1 an (ligne postérieure à un seul flux)
def formule_3(N, Mi, tf, tr, Mr, A):
    """
    MI > 1 an, MR <= 365 jours (ligne postérieure à un seul flux)
    P = N × (1 + tf × Mi/A) / (1 + tr × Mr/360)
    """
    P = N * (1 + tf * Mi / A) / (1 + tr * Mr / 360)
    return P

# Formule (4.1) : MI > 1 an et MR > 1 an (ligne normale)
def formule_4_1(N, tf, tr, n, nj, A):
    """
    MI > 1 an, MR > 365 jours (ligne normale)
    P = N × [tf × Σ(1/(1+tr)^(i-1+nj/A)) + 1/(1+tr)^(n-1+nj/A)]
    """
    somme_coupons = 0
    for i in range(1, n + 1):
        exposant = i - 1 + nj / A
        somme_coupons += tf / ((1 + tr) ** exposant)
    
    # Valeur nominale à l'échéance
    exposant_final = n - 1 + nj / A
    valeur_nominale = 1 / ((1 + tr) ** exposant_final)
    
    P = N * (somme_coupons + valeur_nominale)
    return P

# Formule (4.2) : Ligne postérieure à un seul flux, MR > 1 an
def formule_4_2(N, Mi, tf, tr, nj, A):
    """
    Ligne postérieure à un seul flux, MR > 365 jours
    P = N × (1 + tf × Mi/A) / (1 + tr)^(nj/A)
    """
    P = N * (1 + tf * Mi / A) / ((1 + tr) ** (nj / A))
    return P

# Formule (4.3) : Ligne postérieure à plusieurs flux
def formule_4_3(N, tf, tr, n, nj, A, Dc1, Dem):
    """
    Ligne postérieure à plusieurs flux (si date éval < date détachement 1er coupon)
    P = N × [tf × (Dc1 - Dem)/A / (1 + tr)^(nj/A) + tf × Σ + 1/(1+tr)^(n-1+nj/A)]
    """
    jours_premier_coupon = (Dc1 - Dem).days
    
    # Premier coupon proratisé
    premier_coupon = tf * jours_premier_coupon / A / ((1 + tr) ** (nj / A))
    
    # Coupons suivants
    somme_coupons = 0
    for i in range(2, n + 1):
        exposant = i - 1 + nj / A
        somme_coupons += tf / ((1 + tr) ** exposant)
    
    # Valeur nominale
    exposant_final = n - 1 + nj / A
    valeur_nominale = 1 / ((1 + tr) ** exposant_final)
    
    P = N * (premier_coupon + somme_coupons + valeur_nominale)
    return P

def calculer_prix_titre(
    isin, date_emission, date_echeance, mi_jours, mr_jours,
    taux_facial, spread, taux_bam, nominal, date_evaluation,
    type_emetteur="Etat", type_ligne="Normale", date_premier_coupon=None
):
    """
    Fonction principale pour calculer le prix d'un titre
    """
    # Convertir les taux en décimaux
    tf = taux_facial / 100
    p = spread / 100  # Prime de risque ou spread
    tA = taux_bam / 100
    
    # Taux de rendement
    if type_emetteur == "Etat":
        tr = tA
    else:
        tr = tA + p
    
    A = get_A(date_evaluation)
    
    # Déterminer quelle formule utiliser
    prix = 0
    formule_utilisee = ""
    
    # Cas 1 : MI <= 365 jours
    if mi_jours <= 365:
        prix = formule_1(nominal, mi_jours, mr_jours, tf, tr)
        formule_utilisee = "Formule (1) : MI ≤ 1 an"
    
    # Cas 2 : MI > 365 jours
    else:
        # Sous-cas 2.1 : MR <= 365 jours
        if mr_jours <= 365:
            if type_ligne == "Postérieure - Un seul flux":
                prix = formule_3(nominal, mi_jours, tf, tr, mr_jours, A)
                formule_utilisee = "Formule (3) : MI > 1 an, MR ≤ 1 an, Ligne postérieure (1 flux)"
            else:
                prix = formule_2(nominal, tf, tr, mr_jours)
                formule_utilisee = "Formule (2) : MI > 1 an, MR ≤ 1 an, Ligne normale"
        
        # Sous-cas 2.2 : MR > 365 jours
        else:
            n = calculer_nombre_coupons(date_evaluation, date_echeance)
            nj = calculer_jours_prochain_coupon(date_evaluation, date_echeance, mi_jours)
            
            if type_ligne == "Postérieure - Un seul flux":
                prix = formule_4_2(nominal, mi_jours, tf, tr, nj, A)
                formule_utilisee = "Formule (4.2) : MI > 1 an, MR > 1 an, Ligne postérieure (1 flux)"
            
            elif type_ligne == "Postérieure - Plusieurs flux" and date_premier_coupon:
                if date_evaluation < date_premier_coupon:
                    prix = formule_4_3(nominal, tf, tr, n, nj, A, date_premier_coupon, date_emission)
                    formule_utilisee = "Formule (4.3) : Ligne postérieure (plusieurs flux)"
                else:
                    prix = formule_4_1(nominal, tf, tr, n, nj, A)
                    formule_utilisee = "Formule (4.1) : MI > 1 an, MR > 1 an, Ligne normale (après 1er coupon)"
            
            else:  # Ligne normale
                prix = formule_4_1(nominal, tf, tr, n, nj, A)
                formule_utilisee = "Formule (4.1) : MI > 1 an, MR > 1 an, Ligne normale"
    
    return prix, formule_utilisee, tr * 100

# Interface Streamlit

# Sidebar pour les paramètres globaux
with st.sidebar:
    st.header("⚙️ Paramètres")
    date_evaluation = st.date_input(
        "Date d'évaluation",
        value=datetime.now(),
        help="Date à laquelle le titre est évalué"
    )
    
    st.markdown("---")
    st.info("💡 **Note**: Les calculs sont conformes à la Circulaire CDVM N° 02/04")

# Tabs pour différentes fonctionnalités
tab1, tab2, tab3 = st.tabs(["📊 Calcul Unique", "📁 Calcul Multiple", "📖 Documentation"])

with tab1:
    st.header("Calcul du prix d'un titre")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Informations du titre")
        isin = st.text_input("Code ISIN", value="MA0000000001", help="Code ISIN du titre")
        
        date_emission = st.date_input(
            "Date d'émission",
            value=datetime.now() - timedelta(days=730),
            help="Date d'émission du titre"
        )
        
        date_echeance = st.date_input(
            "Date d'échéance",
            value=datetime.now() + timedelta(days=730),
            help="Date d'échéance du titre"
        )
        
        nominal = st.number_input(
            "Nominal (DH)",
            min_value=0.0,
            value=100000.0,
            step=1000.0,
            help="Valeur nominale du titre"
        )
        
        taux_facial = st.number_input(
            "Taux facial (%)",
            min_value=0.0,
            max_value=100.0,
            value=3.5,
            step=0.01,
            format="%.2f",
            help="Taux facial du titre"
        )
    
    with col2:
        st.subheader("Paramètres de valorisation")
        
        taux_bam = st.number_input(
            "Taux BAM (%)",
            min_value=0.0,
            max_value=100.0,
            value=2.5,
            step=0.01,
            format="%.2f",
            help="Taux de référence Bank Al-Maghrib"
        )
        
        type_emetteur = st.selectbox(
            "Type d'émetteur",
            ["Etat", "Garanti par l'Etat", "Privé"],
            help="Nature de l'émetteur du titre"
        )
        
        spread = st.number_input(
            "Spread / Prime (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.5 if type_emetteur != "Etat" else 0.0,
            step=0.01,
            format="%.2f",
            help="Prime de risque ou de liquidité",
            disabled=(type_emetteur == "Etat")
        )
        
        type_ligne = st.selectbox(
            "Type de ligne",
            ["Normale", "Postérieure - Un seul flux", "Postérieure - Plusieurs flux"],
            help="Nature de la ligne d'émission"
        )
        
        date_premier_coupon = None
        if type_ligne == "Postérieure - Plusieurs flux":
            date_premier_coupon = st.date_input(
                "Date de détachement du 1er coupon",
                value=date_emission + timedelta(days=450),
                help="Date de détachement du premier coupon"
            )
    
    # Calcul automatique des maturités
    mi_jours = (date_echeance - date_emission).days
    mr_jours = (date_echeance - datetime.combine(date_evaluation, datetime.min.time())).days
    
    # Affichage des maturités calculées
    st.markdown("---")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.metric("Maturité Initiale (MI)", f"{mi_jours} jours")
    with col_info2:
        st.metric("Maturité Résiduelle (MR)", f"{mr_jours} jours")
    
    # Bouton de calcul
    if st.button("🔢 Calculer le prix", type="primary", use_container_width=True):
        if mr_jours <= 0:
            st.error("⚠️ La date d'échéance doit être postérieure à la date d'évaluation!")
        else:
            with st.spinner("Calcul en cours..."):
                prix, formule, taux_rendement = calculer_prix_titre(
                    isin, date_emission, date_echeance, mi_jours, mr_jours,
                    taux_facial, spread, taux_bam, nominal, 
                    datetime.combine(date_evaluation, datetime.min.time()),
                    type_emetteur, type_ligne, 
                    datetime.combine(date_premier_coupon, datetime.min.time()) if date_premier_coupon else None
                )
            
            st.success("✅ Calcul effectué avec succès!")
            
            # Résultats
            st.markdown("---")
            st.subheader("📈 Résultats")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric(
                    "Prix du titre",
                    f"{prix:,.2f} DH",
                    delta=f"{((prix/nominal - 1) * 100):.2f}%"
                )
            
            with col_res2:
                st.metric(
                    "Taux de rendement",
                    f"{taux_rendement:.3f}%"
                )
            
            with col_res3:
                st.metric(
                    "Prix pour 100 DH",
                    f"{(prix/nominal * 100):.4f} DH"
                )
            
            # Détails du calcul
            with st.expander("📋 Détails du calcul"):
                st.write(f"**Formule utilisée:** {formule}")
                st.write(f"**Type d'émetteur:** {type_emetteur}")
                st.write(f"**Type de ligne:** {type_ligne}")
                st.write(f"**Taux BAM:** {taux_bam:.2f}%")
                if type_emetteur != "Etat":
                    st.write(f"**Spread:** {spread:.2f}%")
                st.write(f"**Taux de rendement (tr):** {taux_rendement:.3f}%")
                st.write(f"**Maturité initiale:** {mi_jours} jours ({mi_jours/365:.2f} ans)")
                st.write(f"**Maturité résiduelle:** {mr_jours} jours ({mr_jours/365:.2f} ans)")

with tab2:
    st.header("Calcul multiple à partir d'un fichier Excel")
    
    st.info("""
    📝 **Format du fichier Excel requis:**
    - ISIN
    - Date_Emission (format: YYYY-MM-DD)
    - Date_Echeance (format: YYYY-MM-DD)
    - Nominal
    - Taux_Facial (%)
    - Taux_BAM (%)
    - Type_Emetteur (Etat / Garanti par l'Etat / Privé)
    - Spread (%)
    - Type_Ligne (Normale / Postérieure - Un seul flux / Postérieure - Plusieurs flux)
    - Date_Premier_Coupon (optionnel, format: YYYY-MM-DD)
    """)
    
    # Télécharger un template
    if st.button("📥 Télécharger un template Excel"):
        template_data = {
            'ISIN': ['MA0000000001', 'MA0000000002'],
            'Date_Emission': ['2022-01-15', '2021-06-20'],
            'Date_Echeance': ['2027-01-15', '2026-06-20'],
            'Nominal': [100000, 50000],
            'Taux_Facial': [3.5, 4.2],
            'Taux_BAM': [2.5, 2.5],
            'Type_Emetteur': ['Etat', 'Privé'],
            'Spread': [0, 0.8],
            'Type_Ligne': ['Normale', 'Normale'],
            'Date_Premier_Coupon': ['', '']
        }
        df_template = pd.DataFrame(template_data)
        
        # Convertir en Excel
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_template.to_excel(writer, index=False, sheet_name='Titres')
        
        st.download_button(
            label="💾 Télécharger le template",
            data=output.getvalue(),
            file_name="template_titres.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    uploaded_file = st.file_uploader(
        "Charger un fichier Excel",
        type=['xlsx', 'xls'],
        help="Fichier contenant les informations des titres"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            
            st.write("**Aperçu des données chargées:**")
            st.dataframe(df.head())
            
            if st.button("🔢 Calculer tous les prix", type="primary"):
                resultats = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in df.iterrows():
                    status_text.text(f"Traitement du titre {idx + 1}/{len(df)}...")
                    
                    try:
                        date_em = pd.to_datetime(row['Date_Emission'])
                        date_ech = pd.to_datetime(row['Date_Echeance'])
                        
                        mi = (date_ech - date_em).days
                        mr = (date_ech - pd.Timestamp(date_evaluation)).days
                        
                        date_pc = None
                        if pd.notna(row.get('Date_Premier_Coupon', None)) and row.get('Date_Premier_Coupon') != '':
                            date_pc = pd.to_datetime(row['Date_Premier_Coupon'])
                        
                        prix, formule, tr = calculer_prix_titre(
                            row['ISIN'],
                            date_em,
                            date_ech,
                            mi,
                            mr,
                            row['Taux_Facial'],
                            row['Spread'],
                            row['Taux_BAM'],
                            row['Nominal'],
                            pd.Timestamp(date_evaluation),
                            row['Type_Emetteur'],
                            row['Type_Ligne'],
                            date_pc
                        )
                        
                        resultats.append({
                            'ISIN': row['ISIN'],
                            'Nominal': row['Nominal'],
                            'Prix': prix,
                            'Prix_100DH': (prix / row['Nominal']) * 100,
                            'Taux_Rendement': tr,
                            'Formule': formule,
                            'MI_jours': mi,
                            'MR_jours': mr
                        })
                    
                    except Exception as e:
                        st.warning(f"Erreur pour le titre {row['ISIN']}: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / len(df))
                
                status_text.text("Calcul terminé!")
                
                df_resultats = pd.DataFrame(resultats)
                
                st.success(f"✅ {len(resultats)} titres calculés avec succès!")
                
                st.subheader("📊 Résultats")
                st.dataframe(df_resultats, use_container_width=True)
                
                # Télécharger les résultats
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df_resultats.to_excel(writer, index=False, sheet_name='Résultats')
                
                st.download_button(
                    label="💾 Télécharger les résultats",
                    data=output.getvalue(),
                    file_name=f"resultats_valorisation_{date_evaluation}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier: {str(e)}")

with tab3:
    st.header("📖 Documentation")
    
    st.markdown("""
    ## Vue d'ensemble
    
    Cette application permet de calculer le prix des titres de créances conformément à la 
    **Circulaire CDVM N° 02/04** relative aux conditions d'évaluation des valeurs apportées 
    à un OPCVM ou détenues par lui.
    
    ## Formules d'actualisation
    
    ### 1. Titres de maturité initiale ≤ 1 an (Formule 1)
    
    ```
    P = N × (1 + tf × Mi/360) / (1 + tr × Mr/360)
    ```
    
    ### 2. MI > 1 an et MR ≤ 1 an
    
    **Ligne normale (Formule 2):**
    ```
    P = N × (1 + tf) / (1 + tr × Mr/360)
    ```
    
    **Ligne postérieure à un seul flux (Formule 3):**
    ```
    P = N × (1 + tf × Mi/A) / (1 + tr × Mr/360)
    ```
    
    ### 3. MI > 1 an et MR > 1 an
    
    **Ligne normale (Formule 4.1):**
    ```
    P = N × [tf × Σ(1/(1+tr)^(i-1+nj/A)) + 1/(1+tr)^(n-1+nj/A)]
    ```
    
    **Ligne postérieure à un seul flux (Formule 4.2):**
    ```
    P = N × (1 + tf × Mi/A) / (1 + tr)^(nj/A)
    ```
    
    **Ligne postérieure à plusieurs flux (Formule 4.3):**
    ```
    P = N × [tf × (Dc1 - Dem)/A / (1 + tr)^(nj/A) + tf × Σ + 1/(1+tr)^(n-1+nj/A)]
    ```
    
    ## Variables
    
    - **P**: Prix du titre (DH)
    - **N**: Nominal (DH)
    - **Mi**: Maturité initiale (jours)
    - **Mr**: Maturité résiduelle (jours)
    - **tf**: Taux facial (décimal)
    - **tr**: Taux de rendement (décimal) = Taux BAM + Spread (pour émetteurs privés)
    - **A**: 366 si année bissextile, 365 sinon
    - **n**: Nombre de coupons restants
    - **nj**: Nombre de jours jusqu'au prochain coupon
    
    ## Types d'émetteurs
    
    1. **Etat**: tr = Taux BAM
    2. **Garanti par l'Etat**: tr = Taux BAM + Prime de liquidité
    3. **Privé**: tr = Taux BAM + Prime de risque
    
    ## Types de lignes
    
    - **Normale**: Emission standard avec coupons réguliers
    - **Postérieure à un seul flux**: Une seule échéance pour le coupon et le nominal
    - **Postérieure à plusieurs flux**: Premier coupon calculé sur une durée > 1 an
    
    ## Références
    
    - Circulaire CDVM N° 02/04 du 02 juillet 2004
    - Arrêté du Ministre de l'Economie et des Finances n° 160/04
    - Bank Al-Maghrib - Courbe des taux de référence
    """)
    
    with st.expander("⚖️ Cadre réglementaire"):
        st.markdown("""
        La présente application implémente les dispositions de:
        
        - **Dahir portant loi n°1-93-212** relatif au CDVM
        - **Dahir 1-04-17 portant loi n° 23-01**
        - **Dahir portant loi n°1-93-213** relatif aux OPCVM
        - **Arrêté n° 160-04 du 22 janvier 2004**
        
        Les sanctions en cas de non-respect des règles d'évaluation sont prévues 
        à l'article 124 du Dahir portant loi n°1-93-213.
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    💰 Application de Valorisation des Titres | Conforme à la Circulaire CDVM N° 02/04<br>
    Développé pour le respect de la réglementation marocaine sur les OPCVM
    </div>
    """,
    unsafe_allow_html=True
)
