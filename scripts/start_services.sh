#!/bin/bash
# Script de démarrage des services RAG (API + Interface Streamlit)

set -e

echo "🚀 Démarrage des services Puls-Events..."

# Démarrage de l'API FastAPI en arrière-plan
echo "📡 Lancement de l'API FastAPI sur le port 8000..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Attendre que l'API soit prête
echo "⏳ Attente du démarrage de l'API..."
sleep 5

# Vérification de l'API
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    echo "⏳ L'API n'est pas encore prête, nouvelle tentative dans 2 secondes..."
    sleep 2
done

echo "✅ API FastAPI prête sur http://localhost:8000"

# Démarrage de Streamlit
echo "🎨 Lancement de l'interface Streamlit sur le port 8001..."
streamlit run src/streamlit_app.py --server.port 8001 --server.address 0.0.0.0 &
STREAMLIT_PID=$!

echo "✅ Streamlit prêt sur http://localhost:8001"

echo ""
echo "=========================================="
echo "🎭 Puls-Events - Tous les services sont prêts!"
echo "=========================================="
echo "📡 API FastAPI : http://localhost:8000"
echo "🎨 Interface Web : http://localhost:8001"
echo "📊 Health Check : http://localhost:8000/health"
echo "=========================================="
echo ""
echo "Pour arrêter les services, appuyez sur Ctrl+C"

# Fonction de nettoyage lors de l'arrêt
cleanup() {
    echo ""
    echo "🛑 Arrêt des services..."
    kill $API_PID 2>/dev/null || true
    kill $STREAMLIT_PID 2>/dev/null || true
    echo "✅ Services arrêtés"
    exit 0
}

# Capture du signal d'interruption
trap cleanup SIGINT SIGTERM

# Attendre que les processus se terminent
wait
