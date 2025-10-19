#!/bin/bash
# setup_and_run.sh

echo "🔧 Installation des dépendances..."
pip install streamlit pandas numpy scikit-learn plotly joblib python-dotenv openai tavily-python

echo "📁 Vérification du fichier de données..."
if [ ! -f "Loan_Data.csv" ]; then
    echo "❌ ERREUR: Loan_Data.csv introuvable!"
    exit 1
fi

echo "🔑 Vérification des clés API..."
if [ ! -f ".env" ]; then
    echo "⚠️  Fichier .env non trouvé. Création..."
    cat > .env << EOF
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
MLFLOW_TRACKING_URI=http://localhost:5000
EOF
    echo "⚠️  Veuillez éditer .env avec vos vraies clés API!"
    exit 1
fi

echo "🚀 Lancement de l'application..."
streamlit run app.py
