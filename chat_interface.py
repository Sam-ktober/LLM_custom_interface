import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st
import requests
import json
import sys
from datetime import datetime
from pathlib import Path
import re
import subprocess
import torch
import psutil
import plotly.graph_objects as go
from typing import Tuple, Optional

# Configuration de la page
st.set_page_config(
    page_title="Chat IA + Studio Audio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION DES DOSSIERS ET CHEMINS
# ============================================================================
SAVE_DIR = Path("saved_chats")
AUDIO_OUTPUT_DIR = Path("outputs/audio")
FISH_SPEECH_PATH = Path(r"C:\Users\sampl\fish-speech")

# Création des dossiers au démarrage
SAVE_DIR.mkdir(exist_ok=True)
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CSS personnalisé
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    h1 {
        text-align: center;
        color: #6366f1;
        margin-bottom: 1rem;
    }
    .emotion-tag {
        background-color: #ff6b6b;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        display: inline-block;
        margin: 0.25rem;
        font-weight: bold;
    }
    /* Style pour les jauges */
    .gauge-container {
        padding: 0.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS DE MONITORING GPU/RAM
# ============================================================================

def get_gpu_usage() -> Tuple[float, float]:
    """
    Récupère l'utilisation de la VRAM via pynvml
    
    Returns:
        Tuple[float, float]: (VRAM utilisée en MB, VRAM totale en MB)
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        used_mb = info.used / (1024 ** 2)
        total_mb = info.total / (1024 ** 2)
        
        pynvml.nvmlShutdown()
        return used_mb, total_mb
    except Exception as e:
        # Fallback si pynvml n'est pas disponible
        if torch.cuda.is_available():
            used_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
            total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            return used_mb, total_mb
        return 0.0, 8192.0

def get_ram_usage() -> float:
    """
    Récupère l'utilisation de la RAM système
    
    Returns:
        float: Pourcentage d'utilisation de la RAM
    """
    try:
        return psutil.virtual_memory().percent
    except Exception:
        return 0.0

def calculate_context_percentage(tokens: int, max_tokens: int = 16000) -> float:
    """
    Calcule le pourcentage de remplissage du contexte
    
    Args:
        tokens: Nombre de tokens actuels
        max_tokens: Limite du contexte (défaut 16k)
    
    Returns:
        float: Pourcentage (0-100)
    """
    if max_tokens <= 0:
        return 0.0
    return min((tokens / max_tokens) * 100, 100.0)

def create_gauge(value: float, max_value: float, title: str, color: str = "#6366f1") -> go.Figure:
    """
    Crée une jauge circulaire Plotly
    
    Args:
        value: Valeur actuelle
        max_value: Valeur maximale
        title: Titre de la jauge
        color: Couleur de la jauge
    
    Returns:
        go.Figure: Figure Plotly
    """
    percentage = (value / max_value * 100) if max_value > 0 else 0
    
    # Choix de la couleur selon le remplissage
    if percentage < 50:
        gauge_color = "#10b981"  # Vert
    elif percentage < 80:
        gauge_color = "#f59e0b"  # Orange
    else:
        gauge_color = "#ef4444"  # Rouge
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14, 'color': '#ffffff'}},
        number={'suffix': f" / {max_value:.0f}", 'font': {'size': 16, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "#ffffff"},
            'bar': {'color': gauge_color, 'thickness': 0.75},
            'bgcolor': "rgba(0,0,0,0.1)",
            'borderwidth': 2,
            'bordercolor': "#ffffff",
            'steps': [
                {'range': [0, max_value * 0.5], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [max_value * 0.5, max_value * 0.8], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [max_value * 0.8, max_value], 'color': 'rgba(239, 68, 68, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#ffffff", 'family': "Arial"},
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_percentage_gauge(percentage: float, title: str) -> go.Figure:
    """
    Crée une jauge de pourcentage
    
    Args:
        percentage: Valeur en pourcentage (0-100)
        title: Titre de la jauge
    
    Returns:
        go.Figure: Figure Plotly
    """
    # Choix de la couleur
    if percentage < 50:
        gauge_color = "#10b981"
    elif percentage < 80:
        gauge_color = "#f59e0b"
    else:
        gauge_color = "#ef4444"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14, 'color': '#ffffff'}},
        number={'suffix': "%", 'font': {'size': 18, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#ffffff"},
            'bar': {'color': gauge_color, 'thickness': 0.75},
            'bgcolor': "rgba(0,0,0,0.1)",
            'borderwidth': 2,
            'bordercolor': "#ffffff",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [50, 80], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [80, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': percentage
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#ffffff", 'family': "Arial"},
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def render_monitors(tokens: int = 0):
    st.sidebar.markdown("### 📊 Monitoring Ressources")
    vram_used, vram_total = get_gpu_usage()
    ram_percent = get_ram_usage()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        # CORRECTION ICI : width="stretch" au lieu de use_container_width=True
        st.plotly_chart(create_gauge(vram_used, vram_total, "VRAM (MB)"), config={'displayModeBar': False})
    with col2:
        # CORRECTION ICI
        st.plotly_chart(create_gauge(ram_percent, 100, "RAM Sys (%)"), config={'displayModeBar': False})
        
    st.sidebar.caption(f"🎮 GPU: {vram_used:.0f} MB | 📝 Ctx: {tokens:,} tok")
    st.sidebar.divider()



def check_fish_speech_installed() -> bool:
    """Vérifie l'installation et tente de localiser le moteur"""
    if not FISH_SPEECH_PATH.exists():
        return False
    # On vérifie si on trouve au moins le dossier du package python
    return (FISH_SPEECH_PATH / "fish_speech").exists() or (FISH_SPEECH_PATH / "tools").exists()

def find_inference_script(root_path: Path) -> Path:
    """
    Cherche récursivement le bon script d'inférence.
    Fish-Speech change souvent de point d'entrée (tools/inference.py, entrypoint/inference.py, etc.)
    """
    # 1. Liste des candidats connus (par ordre de probabilité pour la v1.4/Main)
    candidates = [
        root_path / "tools" / "inference.py",            # Standard v1.4
        root_path / "fish_speech" / "entrypoint" / "inference.py", # Nouvelle structure v1.5
        root_path / "tools" / "run_webui.py",            # Fallback WebUI (souvent dispo)
        root_path / "inference.py",                      # Racine
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
            
    # 2. Recherche désespérée : tout fichier s'appelant "inference.py"
    found = list(root_path.rglob("inference.py"))
    if found:
        # On évite les fichiers internes de test ou vqgan/llama isolés si possible,
        # on cherche celui qui est le plus proche de la racine ou dans tools/
        # Priorité : tools > entrypoint > root
        for f in found:
            if "tools" in str(f) and "vqgan" not in str(f): return f
        return found[0] # On prend le premier trouvé
        
    raise FileNotFoundError(f"Impossible de trouver un script d'inférence (inference.py) dans {root_path}")

def get_emotion_tags():
    """Liste des balises émotionnelles supportées"""
    return {
        "Normal": "",
        "Cri puissant": "(shouting)",
        "Hurlement": "(screaming)",
        "Pleurs intenses": "(crying loudly)",
        "Colère": "(angry)",
        "Excitation": "(excited)",
        "Peur": "(fearful)",
        "Tristesse": "(sad)",
        "Chuchotement": "(whispering)",
    }

def format_text_with_emotion(text: str, emotion_tag: str, max_intensity: bool = False) -> str:
    """Formate le texte pour maximiser l'effet émotionnel"""
    if max_intensity:
        text = text.upper()
        # Ajoute des points d'exclamation sans doubler s'ils existent déjà
        if not text.rstrip().endswith('!'):
            text = f"{text} !!!"
    
    if emotion_tag:
        # L'espace après le tag est crucial
        return f"{emotion_tag} {text}"
    return text







def generate_audio_fish_speech(text: str, voice_seed: str = "default", output_path: Optional[Path] = None) -> Path:
    """Version Bridge : Appelle notre script personnalisé pour éviter les erreurs de chemin"""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = AUDIO_OUTPUT_DIR / f"audio_{timestamp}.wav"
    
    # Chemin vers notre script de secours
    bridge_script = FISH_SPEECH_PATH / "bridge.py"
    
    if not bridge_script.exists():
        raise Exception("❌ Le fichier 'bridge.py' est manquant dans le dossier Fish-Speech !")

    # Commande simplifiée
    cmd = [
        sys.executable, str(bridge_script),
        str(output_path),
        text
    ]
    
    # On nettoie l'environnement pour éviter les conflits
    env = os.environ.copy()
    env["PYTHONPATH"] = str(FISH_SPEECH_PATH)
    
    # Exécution
    try:
        process = subprocess.run(
            cmd,
            cwd=str(FISH_SPEECH_PATH),
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if process.returncode != 0:
            raise Exception(f"Erreur Bridge (Logs): {process.stdout}\n{process.stderr}")
            
        if not output_path.exists():
             raise Exception("Le script bridge a fini mais aucun fichier n'a été créé.")
             
        return output_path
        
    except Exception as e:
        raise Exception(f"❌ Échec génération : {e}")
    





def list_generated_audios():
    """Liste tous les fichiers audio générés"""
    audio_files = []
    for filepath in AUDIO_OUTPUT_DIR.glob("*.wav"):
        try:
            audio_files.append({
                "filepath": filepath,
                "filename": filepath.name,
                "timestamp": datetime.fromtimestamp(filepath.stat().st_mtime),
                "size": filepath.stat().st_size / 1024
            })
        except Exception: continue
    audio_files.sort(key=lambda x: x["timestamp"], reverse=True)
    return audio_files

# ============================================================================
# FONCTIONS DE GESTION DES CONVERSATIONS
# ============================================================================

def sanitize_filename(text: str) -> str:
    """Nettoie un texte pour en faire un nom de fichier valide"""
    cleaned = re.sub(r'[^\w\s-]', '', text)
    cleaned = re.sub(r'\s+', '_', cleaned)
    return cleaned[:50]

def generate_filename(messages: list) -> str:
    """Génère un nom de fichier basé sur le premier message utilisateur"""
    if not messages:
        return f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    first_user_msg = None
    for msg in messages:
        if msg.get("role") == "user":
            first_user_msg = msg.get("content", "")
            break
    
    if first_user_msg:
        preview = first_user_msg[:50]
        filename = sanitize_filename(preview)
        return f"{filename}_{datetime.now().strftime('%H%M%S')}"
    
    return f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def quick_save(messages, model, system_prompt, context_mode, custom_name=None):
    """Sauvegarde rapide de la conversation"""
    if not messages:
        return None
    
    if custom_name:
        filename = f"{sanitize_filename(custom_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    else:
        filename = f"{generate_filename(messages)}.json"
    
    filepath = SAVE_DIR / filename
    
    save_data = {
        "model": model,
        "system_prompt": system_prompt,
        "context_mode": context_mode,
        "save_date": datetime.now().isoformat(),
        "total_messages": len(messages),
        "messages": messages
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    return filepath

def list_saved_conversations():
    """Liste toutes les conversations sauvegardées"""
    saved_files = []
    
    for filepath in SAVE_DIR.glob("*.json"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                metadata = {
                    "filename": filepath.name,
                    "filepath": filepath,
                    "save_date": data.get("save_date", "Date inconnue"),
                    "model": data.get("model", "Modèle inconnu"),
                    "total_messages": data.get("total_messages", 0),
                    "first_message": ""
                }
                
                for msg in data.get("messages", []):
                    if msg.get("role") == "user":
                        metadata["first_message"] = msg.get("content", "")[:60] + "..."
                        break
                
                saved_files.append((filepath, metadata))
        except (json.JSONDecodeError, Exception):
            continue
    
    saved_files.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
    
    return saved_files

def load_conversation_from_file(filepath):
    """Charge une conversation depuis un fichier local"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "messages" not in data or not isinstance(data["messages"], list):
            return None
        
        return data
    except Exception:
        return None

def delete_saved_conversation(filepath):
    """Supprime une conversation sauvegardée"""
    try:
        filepath.unlink()
        return True
    except Exception:
        return False

@st.cache_data(ttl=60)
def get_available_models():
    """Récupère la liste des modèles installés sur Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [(model["name"], model.get("size", 0)) for model in models]
        return []
    except requests.exceptions.RequestException:
        return []

def estimate_tokens(text: str) -> int:
    """Estimation grossière du nombre de tokens (1 token ≈ 4 caractères)"""
    return len(text) // 4

def calculate_context_tokens(messages: list) -> int:
    """Calcule le nombre approximatif de tokens dans l'historique"""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))
    return total

def limit_context(messages, max_messages=None, max_tokens=None):
    """Limite l'historique pour ne pas dépasser la fenêtre de contexte"""
    if not messages:
        return []
    
    if max_messages and len(messages) > max_messages:
        return messages[-max_messages:]
    
    if max_tokens:
        filtered = []
        current_tokens = 0
        
        for msg in reversed(messages):
            msg_tokens = estimate_tokens(msg.get("content", ""))
            if current_tokens + msg_tokens <= max_tokens:
                filtered.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        return filtered
    
    return messages

def load_imported_conversation(data, mode="replace"):
    """Charge une conversation dans st.session_state"""
    if mode == "replace":
        st.session_state.messages = data["messages"]
    elif mode == "append":
        st.session_state.messages.extend(data["messages"])
    
    if "model" in data:
        st.session_state.model = data["model"]
    if "system_prompt" in data:
        st.session_state.system_prompt = data["system_prompt"]
    if "context_mode" in data:
        st.session_state.context_mode = data["context_mode"]
    
    st.success(f"✅ Conversation chargée : {len(data['messages'])} message(s)")
    st.rerun()

# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

# Titre principal
st.title("🤖 Chat IA + 🎙️ Studio Audio")

# Création des onglets
tab_chat, tab_audio = st.tabs(["💬 Chat IA", "🎙️ Studio Audio"])

# ============================================================================
# SIDEBAR (Commune aux deux onglets)
# ============================================================================
with st.sidebar:
    # Calcul des tokens pour le monitoring
    current_tokens = calculate_context_tokens(st.session_state.get("messages", []))
    
    # Affichage des jauges de monitoring
    render_monitors(current_tokens)
    
    st.header("⚙️ Configuration")
    
    # Quick Reload
    st.subheader("⚡ Quick Reload")
    
    saved_conversations = list_saved_conversations()
    
    if saved_conversations:
        conversation_options = ["--- Nouvelle conversation ---"]
        conversation_map = {}
        
        for filepath, metadata in saved_conversations:
            display_name = f"{metadata['first_message'][:40]}... ({metadata['total_messages']} msgs)"
            conversation_options.append(display_name)
            conversation_map[display_name] = filepath
        
        selected_conversation = st.selectbox(
            "Charger une conversation",
            conversation_options,
            help="Sélectionnez une conversation sauvegardée"
        )
        
        if selected_conversation != "--- Nouvelle conversation ---":
            if st.session_state.get("last_selected") != selected_conversation:
                st.session_state.last_selected = selected_conversation
                
                filepath = conversation_map[selected_conversation]
                data = load_conversation_from_file(filepath)
                
                if data:
                    load_imported_conversation(data, mode="replace")
        else:
            st.session_state.last_selected = None
        
        if selected_conversation != "--- Nouvelle conversation ---":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Recharger", use_container_width=True):
                    filepath = conversation_map[selected_conversation]
                    data = load_conversation_from_file(filepath)
                    if data:
                        load_imported_conversation(data, mode="replace")
            
            with col2:
                if st.button("🗑️ Supprimer", use_container_width=True):
                    filepath = conversation_map[selected_conversation]
                    if delete_saved_conversation(filepath):
                        st.success("✅ Conversation supprimée")
                        st.rerun()
    else:
        st.info("📁 Aucune conversation sauvegardée")
    
    st.divider()
    
    # Sélection du modèle
    available_models = get_available_models()
    
    if not available_models:
        st.warning("⚠️ Aucun modèle détecté")
        selected_model = st.text_input("Modèle (manuel)", value="qwen2.5-coder:7b")
    else:
        model_options = [f"{name} ({size // (1024**3):.1f} GB)" if size > 0 else name 
                        for name, size in available_models]
        model_names = [name for name, _ in available_models]
        
        selected_index = st.selectbox(
            "📦 Modèle",
            range(len(model_options)),
            format_func=lambda i: model_options[i]
        )
        selected_model = model_names[selected_index]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"✅ {len(available_models)} modèle(s)")
        with col2:
            if st.button("🔄"):
                st.cache_data.clear()
                st.rerun()
    
    if "model" not in st.session_state:
        st.session_state.model = selected_model
    else:
        if st.session_state.model != selected_model:
            st.session_state.model = selected_model
            st.session_state.messages = []
    
    st.divider()
    
    # Gestion du contexte
    st.subheader("🧠 Contexte")
    
    context_mode = st.radio(
        "Mode",
        ["Complet", "Limité (messages)", "Limité (tokens)", "Fenêtre glissante"],
        help="Gère la quantité d'historique"
    )
    
    st.session_state.context_mode = context_mode
    
    if context_mode == "Limité (messages)":
        max_messages = st.slider("Messages", 2, 50, 20, 2)
        st.session_state.max_context_messages = max_messages
    elif context_mode == "Limité (tokens)":
        max_tokens_context = st.slider("Tokens max", 1000, 16000, 4000, 500)
        st.session_state.max_context_tokens = max_tokens_context
    elif context_mode == "Fenêtre glissante":
        window_size = st.slider("Fenêtre", 4, 30, 10, 2)
        st.session_state.window_size = window_size * 2
    
    st.divider()
    
    # System Prompt
    st.subheader("🎭 Personnalité")
    
    default_prompts = {
        "Par défaut": "",
        "Expert Python": "Tu es un expert en Python. Réponds de manière concise et technique.",
        "Code Reviewer": "Tu es un expert en révision de code.",
        "Debugger": "Tu es un expert en débogage.",
    }
    
    prompt_choice = st.selectbox("Mode", list(default_prompts.keys()))
    
    if prompt_choice == "Personnalisé":
        system_prompt = st.text_area("Instructions", value=st.session_state.get("system_prompt", ""))
    else:
        system_prompt = default_prompts[prompt_choice]
    
    st.session_state.system_prompt = system_prompt
    
    st.divider()
    
    # Paramètres avancés
    with st.expander("🔧 Paramètres avancés"):
        temperature = st.slider("Température", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Tokens max", 100, 8000, 2000, 100)
        top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05)
        use_streaming = st.checkbox("Streaming", value=True)
        st.session_state.use_streaming = use_streaming
    
    st.divider()
    
    # Quick Save
    st.subheader("💾 Quick Save")
    
    if st.session_state.get("messages"):
        use_custom_name = st.checkbox("Nom personnalisé", value=False)
        
        if use_custom_name:
            custom_name = st.text_input("Nom", placeholder="Ex: Projet Flask")
        else:
            custom_name = None
        
        if st.button("💾 Sauvegarder", use_container_width=True, type="primary"):
            filepath = quick_save(
                st.session_state.messages,
                st.session_state.model,
                st.session_state.get("system_prompt", ""),
                st.session_state.get("context_mode", "Complet"),
                custom_name
            )
            
            if filepath:
                st.success(f"✅ `{filepath.name}`")
                st.rerun()
    else:
        st.info("💬 Aucune conversation")
    
    st.divider()
    
    # Actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Nouvelle", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_selected = None
            st.rerun()
    
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

# ============================================================================
# ONGLET 1 : CHAT IA
# ============================================================================
with tab_chat:
    # Initialisation
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Affichage de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Zone de saisie
    if prompt := st.chat_input("💭 Posez votre question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                messages_to_send = []
                
                if st.session_state.system_prompt:
                    messages_to_send.append({
                        "role": "system",
                        "content": st.session_state.system_prompt
                    })
                
                # Limitation du contexte
                context_mode = st.session_state.get("context_mode", "Complet")
                
                if context_mode == "Complet":
                    filtered_messages = st.session_state.messages
                elif context_mode == "Limité (messages)":
                    max_msg = st.session_state.get("max_context_messages", 20)
                    filtered_messages = limit_context(st.session_state.messages, max_messages=max_msg)
                elif context_mode == "Limité (tokens)":
                    max_tok = st.session_state.get("max_context_tokens", 4000)
                    filtered_messages = limit_context(st.session_state.messages, max_tokens=max_tok)
                elif context_mode == "Fenêtre glissante":
                    window = st.session_state.get("window_size", 20)
                    filtered_messages = st.session_state.messages[-window:] if len(st.session_state.messages) > window else st.session_state.messages
                else:
                    filtered_messages = st.session_state.messages
                
                messages_to_send.extend(filtered_messages)
                
                if len(filtered_messages) < len(st.session_state.messages):
                    truncated = len(st.session_state.messages) - len(filtered_messages)
                    st.info(f"ℹ️ {truncated} message(s) omis")
                
                url = "http://localhost:11434/api/chat"
                payload = {
                    "model": st.session_state.model,
                    "messages": messages_to_send,
                    "stream": st.session_state.get("use_streaming", True),
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "top_p": top_p
                    }
                }
                
                if st.session_state.get("use_streaming", True):
                    with requests.post(url, json=payload, stream=True, timeout=120) as r:
                        r.raise_for_status()
                        
                        for line in r.iter_lines():
                            if line:
                                try:
                                    chunk = json.loads(line.decode('utf-8'))
                                    
                                    if "message" in chunk:
                                        content = chunk["message"].get("content", "")
                                        full_response += content
                                        message_placeholder.markdown(full_response + "▌")
                                    
                                    if chunk.get("done", False):
                                        break
                                        
                                except json.JSONDecodeError:
                                    continue
                else:
                    payload["stream"] = False
                    r = requests.post(url, json=payload, timeout=120)
                    r.raise_for_status()
                    response_data = r.json()
                    full_response = response_data.get("message", {}).get("content", "")
                    message_placeholder.markdown(full_response)
                
                message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })
                
                # Force le refresh des jauges
                st.rerun()
                
            except requests.exceptions.ConnectionError:
                st.error("❌ Impossible de se connecter à Ollama")
            except requests.exceptions.Timeout:
                st.error("⏱️ Timeout")
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ Erreur HTTP : {e}")
            except Exception as e:
                st.error(f"❌ Erreur : {e}")

# ============================================================================
# ONGLET 2 : STUDIO AUDIO
# ============================================================================
with tab_audio:
    st.header("🎙️ Studio Audio - Fish-Speech 1.4")
    
    # Vérification de Fish-Speech
    fish_installed = check_fish_speech_installed()
    
    if not fish_installed:
        st.error(f"⚠️ Fish-Speech non trouvé dans `{FISH_SPEECH_PATH}`")
        st.info("""
        **Vérifiez l'installation :**
        1. Fish-Speech doit être dans `C:\\Users\\sampl\\fish-speech`
        2. Le dossier `tools` doit exister
        3. Lancez depuis l'environnement virtuel Fish-Speech
        """)
    else:
        st.success(f"✅ Fish-Speech détecté dans `{FISH_SPEECH_PATH}`")
    
    # Configuration GPU
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        st.success(f"🎮 GPU : {gpu_name} ({vram:.1f} GB)")
    else:
        st.warning("⚠️ CPU mode (plus lent)")
    
    st.divider()
    
    # Interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✍️ Texte")
        
        text_input = st.text_area(
            "Écrivez votre texte",
            height=150,
            placeholder="Ex: ARRÊTEZ! JE NE PEUX PLUS!",
            help="Le texte sera converti en audio"
        )
        
        if text_input:
            st.caption(f"📝 {len(text_input)} caractères")
    
    with col2:
        st.subheader("🎭 Config")
        
        emotions = get_emotion_tags()
        selected_emotion_name = st.selectbox(
            "Émotion",
            list(emotions.keys()),
            index=1
        )
        
        emotion_tag = emotions[selected_emotion_name]
        
        if emotion_tag:
            st.markdown(f'<div class="emotion-tag">{emotion_tag}</div>', unsafe_allow_html=True)
        
        max_intensity = st.checkbox("🔥 Intensité Max", value=False)
        
        voice_seed = st.selectbox(
            "Voix",
            ["default", "angry_male", "excited_female", "shouting_male"]
        )
    
    st.divider()
    
    # Aperçu
    if text_input:
        formatted_text = format_text_with_emotion(text_input, emotion_tag, max_intensity)
        st.subheader("👁️ Aperçu")
        st.code(formatted_text, language=None)
    
    st.divider()
    
    # Génération
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        generate_button = st.button(
            "🔥 Générer le Cri",
            use_container_width=True,
            type="primary",
            disabled=not text_input or not fish_installed
        )
    
    if generate_button and text_input:
        with st.spinner("🎵 Génération en cours... (30-60s)"):
            try:
                formatted_text = format_text_with_emotion(text_input, emotion_tag, max_intensity)
                
                audio_path = generate_audio_fish_speech(formatted_text, voice_seed)
                
                if audio_path.exists():
                    st.success("✅ Audio généré!")
                    
                    st.subheader("🔊 Résultat")
                    st.audio(str(audio_path), format="audio/wav")
                    
                    file_size = audio_path.stat().st_size / 1024
                    st.caption(f"📁 `{audio_path.name}` ({file_size:.1f} KB)")
                    
                    with open(audio_path, "rb") as f:
                        st.download_button(
                            "⬇️ Télécharger",
                            f,
                            file_name=audio_path.name,
                            mime="audio/wav"
                        )
                else:
                    st.error("❌ Fichier non généré")
                    
            except Exception as e:
                st.error(f"❌ {str(e)}")
                st.info("💡 Vérifiez que Fish-Speech 1.4 est installé")
    
    st.divider()
    
    # Historique
    st.subheader("📚 Historique")
    
    generated_audios = list_generated_audios()
    
    if generated_audios:
        st.caption(f"{len(generated_audios)} fichier(s)")
        
        cols = st.columns(3)
        
        for idx, audio_info in enumerate(generated_audios[:9]):
            with cols[idx % 3]:
                st.caption(f"🕒 {audio_info['timestamp'].strftime('%H:%M:%S')}")
                st.audio(str(audio_info['filepath']), format="audio/wav")
                st.caption(f"{audio_info['size']:.1f} KB")
    else:
        st.info("📭 Aucun audio généré")

# Footer
st.divider()
st.caption(f"🚀 Ollama • 🎙️ Fish-Speech • GPU: RTX 4060-Ti")
