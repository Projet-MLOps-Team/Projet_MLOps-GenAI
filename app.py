import streamlit as st
import pandas as pd
import json
from typing import Any, Dict

from agent import build_agent, chat, ml_predict  # ton fichier agent.py

# ========== CONFIG STREAMLIT ==========
st.set_page_config(
    page_title="GENAI – Banking Lab",
    page_icon="🤖",
    layout="wide"
)

# ========== SESSION STATE ==========
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": "..."}]

if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

agent = st.session_state.agent

# ========= PAGE HEADER GLOBAL =========
st.title("GENAI – Banking Lab")

# ========= NAVIGATION PAR ONGLET EN HAUT =========
tab_eda, tab_ml, tab_chat = st.tabs(["📊 EDA", "🔮 Prédiction ML", "💬 Chatbot"])

# ==================== PAGE 1 : EDA ====================
with tab_eda:
    st.header("📊 Analyse Exploratoire – Risque Crédit")

    st.markdown(
        """
        Explore les caractéristiques des clients et comprends les patterns associés au **risque de défaut**.
        """
    )

    # ================= CHARGEMENT CSV =================
    uploaded_file = st.file_uploader("📂 Charger un fichier CSV (dataset crédit)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_df = df
    else:
        df = st.session_state.uploaded_df if st.session_state.uploaded_df is not None else None

    if df is None:
        st.info("👉 Charge un fichier CSV pour commencer l'analyse.")
        st.stop()

    st.success(f"Dataset chargé : **{df.shape[0]} lignes**, **{df.shape[1]} colonnes**")

    # ================= APERCU =================
    st.markdown("### 👀 Aperçu du dataset")
    st.dataframe(df.head(), use_container_width=True)

    # ================= INDICATEURS GLOBAUX =================
    default_rate = df["default"].mean() * 100
    colA, colB, colC = st.columns(3)
    colA.metric("Taux de défaut global", f"{default_rate:.1f} %")
    colB.metric("Clients sains", f"{(df['default']==0).sum()}")
    colC.metric("Clients en défaut", f"{(df['default']==1).sum()}")

    st.markdown("---")

    # ================= DISTRIBUTIONS PAR DEFAUT =================
    st.markdown("## 📈 Variables clés vs défaut")

    numeric_cols = [
        "fico_score", "debt_ratio", "income", "years_employed",
        "loan_amt_outstanding", "total_debt_outstanding"
    ]

    var = st.selectbox("Choisis une variable à explorer :", numeric_cols)

    import altair as alt
    chart = alt.Chart(df).mark_bar(opacity=0.7).encode(
        x=alt.X(var, bin=alt.Bin(maxbins=30)),
        y="count()",
        color=alt.Color("default:N", legend=alt.Legend(title="Default (0=OK, 1=Défaut)"))
    ).properties(width=650, height=350)

    st.altair_chart(chart)

    st.markdown("---")

    # ================= CORRÉLATION =================
    st.markdown("## 🔗 Matrice de corrélation")

    corr = df.corr(numeric_only=True)
    st.dataframe(corr.style.background_gradient(cmap="Reds"), use_container_width=True)

    # Top variables explicatives
    st.markdown("### 🥇 Variables les plus corrélées avec le défaut")

    corr_default = corr["default"].drop("default").sort_values(ascending=False)
    st.bar_chart(corr_default)

    st.markdown("---")

    # ================= SCATTERPLOT =================
    st.markdown("## 🧭 Scatterplot – localiser les zones à risque")

    x_var = st.selectbox("Axe X", numeric_cols, index=2)
    y_var = st.selectbox("Axe Y", numeric_cols, index=0)

    scatter = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
        x=x_var,
        y=y_var,
        color=alt.Color("default:N", legend=alt.Legend(title="Défaut")),
        tooltip=["income", "fico_score", "debt_ratio", "default"]
    ).properties(width=750, height=450)

    st.altair_chart(scatter)

    st.success("Analyse EDA terminée ✔️")


# ==================== PAGE 2 : FORMULAIRE PRÉDICTION ML ====================
with tab_ml:
    st.header("🔮 Prédiction de risque via le modèle ML (.pkl sur S3)")

    st.markdown(
        """
        Remplis ce **questionnaire** : nous estimons ensuite le risque de défaut du client,  
        et nous t’affichons une explication claire et visuelle.
        """
    )

    col_left, col_right = st.columns([1, 1])

    # ========================= FORMULAIRE =========================
    with col_left:
        st.markdown("### 🎯 Profil client / crédit")

        credit_lines = st.number_input(
            "Lignes de crédit ouvertes (credit_lines_outstanding)",
            min_value=0, max_value=50, value=5
        )

        loan_amt = st.number_input(
            "Montant du prêt en cours (€) – loan_amt_outstanding",
            min_value=0, max_value=1_000_000, value=15_000, step=1_000
        )

        total_debt = st.number_input(
            "Dette totale actuelle (€) – total_debt_outstanding",
            min_value=0, max_value=1_000_000, value=25_000, step=1_000
        )

        income = st.number_input(
            "Revenu annuel (€) – income",
            min_value=1, max_value=1_000_000, value=60_000, step=1_000
        )

        years = st.number_input(
            "Ancienneté dans l'emploi (années) – years_employed",
            min_value=0, max_value=50, value=10
        )

        fico = st.number_input(
            "Score FICO – fico_score",
            min_value=300, max_value=850, value=720
        )

        debt_ratio = total_debt / income if income > 0 else 0.0
        st.metric("Debt ratio calculé", f"{debt_ratio:.2f}")

        default_payload = {
            "credit_lines_outstanding": credit_lines,
            "loan_amt_outstanding": loan_amt,
            "total_debt_outstanding": total_debt,
            "income": income,
            "years_employed": years,
            "fico_score": fico,
            "debt_ratio": debt_ratio
        }

    # ========================= JSON EDITABLE =========================
    with col_right:
        st.markdown("### 🧾 Payload JSON (optionnel)")

        st.caption("Tu peux garder ce JSON tel quel ou l’ajuster manuellement avant la prédiction.")

        payload_str = st.text_area(
            "Payload envoyé à `ml_predict` :",
            value=json.dumps(default_payload, indent=2),
            height=260
        )

        lancer = st.button("🚀 Lancer la prédiction ML", type="primary")

    # ========================= PRÉDICTION & AFFICHAGE UX =========================
    if lancer:
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError as e:
            st.error(f"JSON invalide : {e}")
            payload = None

        if payload is not None:
            with st.spinner("Analyse du risque par le modèle…"):
                try:
                    raw = ml_predict.invoke({"payload": payload})
                except Exception as e:
                    st.error(f"Erreur lors de l’appel de ml_predict : {e}")
                    raw = None

            if raw is not None:
                # On essaye de parser le JSON retourné par le tool
                prediction = None
                try:
                    parsed = json.loads(raw)
                    prediction = parsed.get("prediction", {})
                except Exception:
                    prediction = None

                if prediction is None or not isinstance(prediction, dict):
                    st.error("La réponse du modèle n’est pas dans le format attendu.")
                    st.code(raw, language="json")
                else:
                    label_name = prediction.get("label_name", "Résultat inconnu")
                    risk_level = prediction.get("risk_level", "inconnu")
                    proba_default = prediction.get("proba_default", None)
                    explanation = prediction.get("explanation", "")
                    features_used = prediction.get("features_used", [])

                    # --------- Traduction du niveau de risque en jauge ----------
                    if isinstance(proba_default, (float, int)):
                        proba_pct = max(0.0, min(float(proba_default), 1.0)) * 100
                    else:
                        # fallback selon risk_level
                        mapping = {"faible": 15.0, "modéré": 35.0, "élevé": 70.0}
                        proba_pct = mapping.get(risk_level, 50.0)

                    # Couleur / emoji selon le risque
                    if risk_level == "faible":
                        emoji = "🟢"
                        texte_risque = "Risque faible"
                    elif risk_level == "modéré":
                        emoji = "🟠"
                        texte_risque = "Risque modéré"
                    elif risk_level == "élevé":
                        emoji = "🔴"
                        texte_risque = "Risque élevé"
                    else:
                        emoji = "⚪"
                        texte_risque = "Risque non déterminé"

                    st.markdown("---")
                    st.subheader("🧠 Résultat de l’analyse du modèle")

                    # Bloc résumé pour un client
                    col_r1, col_r2 = st.columns([2, 1])
                    with col_r1:
                        st.markdown(
                            f"""
                            **Verdict : {emoji} {label_name}**  
                            **Niveau de risque : {texte_risque}**
                            """
                        )
                        if isinstance(proba_default, (float, int)):
                            st.markdown(
                                f"Le modèle estime une probabilité de défaut d’environ **{proba_pct:.1f}%**."
                            )
                        if explanation:
                            st.markdown(f"📝 *{explanation}*")

                    with col_r2:
                        st.markdown("### 📊 Jauge de risque")
                        st.progress(int(proba_pct))

                    # Features utilisées – version simple
                    if features_used:
                        st.markdown("### 🔍 Variables prises en compte")
                        st.write(", ".join(features_used))

                    # Détails techniques en expander
                    with st.expander("🔧 Détails techniques / JSON brut"):
                        st.markdown("**Réponse brute du tool `ml_predict` :**")
                        st.code(raw, language="json")
                        try:
                            st.markdown("**Vue JSON parsée :**")
                            st.json(parsed)
                        except Exception:
                            pass

    st.markdown("---")
    st.caption(
        "💡 Astuce : cette page sert pour les utilisateurs métier. "
        "Les développeurs peuvent récupérer le payload et la réponse brute dans l’expander."
    )


# ==================== PAGE 3 : CHATBOT ====================
with tab_chat:
    st.header("💬 Chat avec l’agent (web + RAG + ML)")

    st.markdown(
        """
        Exemple de requêtes :
        - *“Résume-moi les frais de tenue de compte pour un non résident.”*  
        - *“Utilise `rag_search` pour extraire les tarifs de découvert.”*  
        - *“Appelle `ml_predict` avec {'credit_lines_outstanding': 5, ...} et explique le résultat.”*
        """
    )

    # Affichage de l'historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Champ d'entrée
    prompt = st.chat_input("Pose une question à l’agent…")

    if prompt:
        # 1. Ajout du message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Appel agent AVEC L’HISTORIQUE COMPLET
        with st.chat_message("assistant"):
            with st.spinner("L’agent réfléchit…"):
                try:
                    answer = chat(agent, st.session_state.messages)
                except Exception as e:
                    answer = f"❌ ERREUR agent: {e}"

                st.markdown(answer)

        # 3. Ajout de la réponse assistant dans la mémoire
        st.session_state.messages.append({"role": "assistant", "content": answer})
