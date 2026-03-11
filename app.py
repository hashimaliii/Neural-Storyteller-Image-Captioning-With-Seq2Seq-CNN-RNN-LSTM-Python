import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import pickle
import re

# ===========================
# Page Configuration
# ===========================
st.set_page_config(
    page_title="Neural Storyteller",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# Custom CSS - Gemini Dark Theme
# ===========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@300;400;500;600&family=Google+Sans+Display:wght@300;400;500&display=swap');

    /* ── Reset & Base ── */
    * { box-sizing: border-box; }

    .stApp {
        background-color: #0D0E0F;
        font-family: 'Google Sans', 'Segoe UI', sans-serif;
        color: #E3E3E3;
        overflow-y: auto !important;
    }

    /* ── Ensure columns don't clip ── */
    [data-testid="stVerticalBlock"] {
        overflow: visible !important;
    }

    /* ── Main content padding ── */
    .block-container {
        padding: 2rem 2.5rem 3rem 2.5rem !important;
        max-width: 1400px !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #111314 !important;
        border-right: 1px solid #2A2B2D !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding: 2rem 1.5rem !important;
    }

    /* ── Hero Header ── */
    .hero-sub {
        font-size: 1rem;
        font-weight: 400;
        color: #8E9499;
        max-width: 520px;
        line-height: 1.6;
    }

    /* ── Section Labels ── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #5F6368;
        margin-bottom: 1rem;
    }

    /* ── Glass Cards ── */
    .glass-card {
        background: linear-gradient(145deg, #17181A 0%, #131415 100%);
        border: 1px solid #2A2B2D;
        border-radius: 20px;
        padding: 1.75rem;
        margin-bottom: 1.25rem;
    }

    /* ── Upload Zone ── */
    [data-testid="stFileUploadDropzone"] {
        background-color: #131415 !important;
        border: 1.5px dashed #2E3134 !important;
        border-radius: 16px !important;
        transition: all 0.25s ease !important;
        min-height: 140px !important;
    }

    [data-testid="stFileUploadDropzone"]:hover {
        background-color: #1A1C1E !important;
        border-color: #4285F4 !important;
    }

    [data-testid="stFileUploadDropzone"] svg {
        fill: #4285F4 !important;
        color: #4285F4 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #8E9499 !important;
        font-size: 0.9rem !important;
    }

    /* ── Uploaded image ── */
    [data-testid="stImage"] > img {
        border-radius: 16px !important;
        border: 1px solid #2A2B2D !important;
        width: 100% !important;
    }

    /* ── Caption Output Box ── */
    .caption-card {
        background: linear-gradient(145deg, #17181A 0%, #131415 100%);
        border: 1px solid #2A2B2D;
        border-radius: 20px;
        padding: 2rem 2rem 1.75rem 2rem;
        margin-bottom: 1.25rem;
        position: relative;
        overflow: hidden;
    }

    .caption-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #4285F4, #9B72CB, #D96570);
        border-radius: 20px 20px 0 0;
    }

    .caption-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(66, 133, 244, 0.12);
        border: 1px solid rgba(66, 133, 244, 0.25);
        border-radius: 100px;
        padding: 0.25rem 0.75rem;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #A8C7FA;
        margin-bottom: 1.25rem;
    }

    .caption-chip::before {
        content: '';
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #4285F4;
        box-shadow: 0 0 6px #4285F4;
    }

    .caption-text {
        font-size: 1.45rem;
        font-weight: 400;
        color: #E8EAED;
        line-height: 1.55;
        letter-spacing: -0.3px;
    }

    /* ── Stat Pills ── */
    .stats-row {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin-top: 1.5rem;
        padding-top: 1.25rem;
        border-top: 1px solid #222426;
    }

    .stat-pill {
        background: #1A1C1E;
        border: 1px solid #2A2B2D;
        border-radius: 100px;
        padding: 0.35rem 0.9rem;
        font-size: 0.78rem;
        color: #8E9499;
        display: flex;
        gap: 6px;
        align-items: center;
    }

    .stat-pill span.val {
        color: #C4C7C5;
        font-weight: 500;
    }

    /* ── Placeholder State ── */
    .placeholder-card {
        background: linear-gradient(145deg, #17181A 0%, #131415 100%);
        border: 1px solid #2A2B2D;
        border-radius: 20px;
        padding: 3.5rem 2rem;
        text-align: center;
    }

    .placeholder-icon {
        width: 56px;
        height: 56px;
        margin: 0 auto 1.25rem;
        background: linear-gradient(135deg, rgba(66,133,244,0.15), rgba(155,114,203,0.15));
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(66,133,244,0.2);
        font-size: 1.5rem;
    }

    .placeholder-title {
        font-size: 1rem;
        font-weight: 500;
        color: #C4C7C5;
        margin-bottom: 0.5rem;
    }

    .placeholder-body {
        font-size: 0.88rem;
        color: #5F6368;
        line-height: 1.6;
        max-width: 280px;
        margin: 0 auto;
    }

    /* ── Architecture Info Cards ── */
    .arch-card {
        background: #131415;
        border: 1px solid #222426;
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.6rem;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .arch-label {
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #5F6368;
    }

    .arch-value {
        font-size: 0.92rem;
        font-weight: 500;
        color: #C4C7C5;
    }

    /* ── Generate Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #4285F4 0%, #6B4EE6 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.65rem 2rem !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.3px !important;
        width: 100% !important;
        margin-top: 1rem !important;
        transition: opacity 0.2s ease, transform 0.1s ease !important;
        box-shadow: 0 4px 20px rgba(66, 133, 244, 0.25) !important;
    }

    .stButton > button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 28px rgba(66, 133, 244, 0.35) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ── Download Button ── */
    .stDownloadButton > button {
        background: transparent !important;
        color: #A8C7FA !important;
        border: 1px solid #2A2B2D !important;
        border-radius: 14px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        width: 100% !important;
        margin-top: 0.5rem !important;
        transition: all 0.2s ease !important;
    }

    .stDownloadButton > button:hover {
        background: rgba(168, 199, 250, 0.06) !important;
        border-color: #4285F4 !important;
    }

    /* ── Sliders ── */
    [data-testid="stSlider"] > div > div > div > div {
        background: linear-gradient(90deg, #4285F4, #9B72CB) !important;
    }

    [data-testid="stSlider"] [role="slider"] {
        background: #A8C7FA !important;
        border: 2px solid #0D0E0F !important;
        box-shadow: 0 0 0 2px #4285F4 !important;
    }

    .stSlider [data-testid="stMarkdownContainer"] p {
        color: #8E9499 !important;
        font-size: 0.82rem !important;
    }

    /* ── Sidebar headings ── */
    [data-testid="stSidebar"] h3 {
        font-size: 0.72rem !important;
        font-weight: 500 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: #5F6368 !important;
        margin-bottom: 1.25rem !important;
        margin-top: 0.5rem !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li {
        color: #8E9499 !important;
        font-size: 0.88rem !important;
    }

    [data-testid="stSidebar"] strong {
        color: #C4C7C5 !important;
    }

    hr {
        border: none !important;
        border-top: 1px solid #1F2022 !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #4285F4 !important;
    }

    /* ── Hide branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Column gap ── */
    [data-testid="column"] {
        padding: 0 0.75rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# Model Classes
# ===========================
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
    def __len__(self): return len(self.itos)
    @staticmethod
    def tokenize(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

class EncoderCNN(nn.Module):
    def __init__(self, hidden_size=512):
        super(EncoderCNN, self).__init__()
        self.linear = nn.Linear(2048, hidden_size)
        self.relu = nn.ReLU()
    def forward(self, cached_features):
        return self.relu(self.linear(cached_features))

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    def forward(self, features, captions):
        captions = captions[:, :-1]
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(inputs)
        return self.linear(hiddens)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, features, captions):
        return self.decoder(self.encoder(features), captions)

# ===========================
# Backend Functions
# ===========================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet = nn.Sequential(*list(resnet.children())[:-1]).to(device)
        resnet.eval()
        encoder = EncoderCNN(hidden_size=512)
        decoder = DecoderRNN(embed_size=512, hidden_size=512, vocab_size=len(vocab), num_layers=1)
        model = Seq2Seq(encoder, decoder).to(device)
        model.load_state_dict(torch.load('best_flickr30k_model.pth', map_location=device))
        model.eval()
        return vocab, resnet, model, device
    except Exception as e:
        st.error(f"System Error: {str(e)}")
        return None, None, None, None

def beam_search(model, feature, vocab, device, beam_width=3, max_len=20):
    start_token = vocab.stoi["<SOS>"]
    sequences = [([start_token], 0.0, None)]
    for _ in range(max_len):
        all_candidates = []
        for seq, score, h_state in sequences:
            if seq[-1] == vocab.stoi["<EOS>"]:
                all_candidates.append((seq, score, h_state))
                continue
            inputs = torch.tensor([seq[-1]]).to(device)
            embeds = model.decoder.embed(inputs)
            if h_state is None:
                output, h_state = model.decoder.lstm(feature.view(1, 1, -1), None)
            else:
                output, h_state = model.decoder.lstm(embeds.view(1, 1, -1), h_state)
            logits = model.decoder.linear(output.squeeze(1))
            log_probs = torch.log_softmax(logits, dim=1)
            top_probs, top_indices = log_probs.topk(beam_width)
            for i in range(beam_width):
                candidate = (seq + [top_indices[0][i].item()], score + top_probs[0][i].item(), h_state)
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = ordered[:beam_width]
        if all(s[0][-1] == vocab.stoi["<EOS>"] for s in sequences):
            break
    best_seq = sequences[0][0]
    caption = [vocab.itos[i] for i in best_seq if vocab.itos[i] not in ["<SOS>", "<EOS>", "<PAD>"]]
    return " ".join(caption)

# ===========================
# Main Application
# ===========================
def main():

    # ── Sidebar ──────────────────────────────────────
    with st.sidebar:
        st.markdown("### Configuration")

        beam_width = st.slider("Beam Search Width", min_value=1, max_value=5, value=3)
        max_length = st.slider("Max Token Length", min_value=10, max_value=30, value=20)

        st.markdown("---")
        st.markdown("### Architecture")

        st.markdown("""
        <div class="arch-card">
            <span class="arch-label">Feature Extractor</span>
            <span class="arch-value">ResNet-50</span>
        </div>
        <div class="arch-card">
            <span class="arch-label">Projection</span>
            <span class="arch-value">2048 &rarr; 512 Linear</span>
        </div>
        <div class="arch-card">
            <span class="arch-label">Generator</span>
            <span class="arch-value">Single-layer LSTM</span>
        </div>
        <div class="arch-card">
            <span class="arch-label">Training Dataset</span>
            <span class="arch-value">Flickr30k</span>
        </div>
        <div class="arch-card">
            <span class="arch-label">Decoding Strategy</span>
            <span class="arch-value">Beam Search</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### How it works")
        st.markdown("""
        1. ResNet-50 extracts a 2048-dim feature vector from the image.
        2. A linear projection maps it to the 512-dim LSTM hidden space.
        3. The decoder generates tokens one-by-one, guided by beam search to find the most probable sequence.
        """)

    # ── Hero ──────────────────────────────────────────
    # Canvas renders the gradient title since Streamlit strips background-clip:text
    st.markdown("""
    <div style="padding:2.5rem 0 2rem 0; border-bottom:1px solid #1F2022; margin-bottom:2.5rem;">
        <div style="font-size:0.72rem;font-weight:500;letter-spacing:2.5px;text-transform:uppercase;color:#8E9499;margin-bottom:0.75rem;font-family:'Google Sans','Segoe UI',sans-serif;">
            Upload any image and the model will generate a natural-language description using a CNN encoder and LSTM decoder.
        </div>
    </div>
    <script>
    (function() {
        function drawTitle() {
            var canvas = document.getElementById('heroCanvas');
            if (!canvas) { setTimeout(drawTitle, 100); return; }
            var dpr = window.devicePixelRatio || 1;
            var w = canvas.parentElement.clientWidth || 700;
            canvas.width = w * dpr;
            canvas.height = 80 * dpr;
            canvas.style.width = w + 'px';
            canvas.style.height = '80px';
            var ctx = canvas.getContext('2d');
            ctx.scale(dpr, dpr);
            var fontSize = Math.min(Math.floor(w / 7.5), 62);
            ctx.font = '400 ' + fontSize + 'px "Google Sans Display","Google Sans","Segoe UI",sans-serif';
            var grad = ctx.createLinearGradient(0, 0, w * 0.75, 0);
            grad.addColorStop(0,    '#4285F4');
            grad.addColorStop(0.45, '#9B72CB');
            grad.addColorStop(1,    '#D96570');
            ctx.fillStyle = grad;
            ctx.fillText('Neural Storyteller', 0, fontSize);
        }
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', drawTitle);
        } else {
            drawTitle();
        }
        window.addEventListener('resize', drawTitle);
    })();
    </script>
    """, unsafe_allow_html=True)

    # ── Load models ───────────────────────────────────
    with st.spinner("Loading model weights..."):
        vocab, resnet, model, device = load_models()

    if model is None or vocab is None:
        st.stop()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # ── Two-column layout ─────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT: Input ───────────────────────────────────
    with col_left:
        st.markdown('<div class="section-label">Input Image</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Select an image file",
            type=['png', 'jpg', 'jpeg'],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True)

            # Image metadata pills
            w, h = image.size
            st.markdown(f"""
            <div style="display:flex; gap:0.6rem; flex-wrap:wrap; margin-top:0.85rem; margin-bottom:0.25rem;">
                <div class="stat-pill">{uploaded_file.name}</div>
                <div class="stat-pill"><span class="val">{w} &times; {h}</span>&nbsp;px</div>
                <div class="stat-pill">Format&nbsp;<span class="val">{image.format or uploaded_file.type.split("/")[-1].upper()}</span></div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Generate Caption"):
                with st.spinner("Running inference..."):
                    img_tensor = transform(image).unsqueeze(0).to(device)
                    with torch.no_grad():
                        raw_feature = resnet(img_tensor).view(1, -1)
                        encoded_feature = model.encoder(raw_feature)
                        caption = beam_search(
                            model,
                            encoded_feature.squeeze(0),
                            vocab,
                            device,
                            beam_width=beam_width,
                            max_len=max_length
                        )
                    st.session_state.generated_caption = caption
                    st.session_state.beam_used = beam_width
                    st.session_state.maxlen_used = max_length
                    word_count = len(caption.split())
                    st.session_state.word_count = word_count
        else:
            # Empty state hint
            st.markdown("""
            <div style="background:#131415; border:1.5px dashed #222426; border-radius:16px; padding:2.5rem; text-align:center; margin-top:0.5rem;">
                <div style="font-size:0.9rem; color:#5F6368; line-height:1.7;">
                    Drag and drop a PNG, JPG, or JPEG file here,<br>or click to browse.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── RIGHT: Output ─────────────────────────────────
    with col_right:
        st.markdown('<div class="section-label">Generated Caption</div>', unsafe_allow_html=True)

        if hasattr(st.session_state, 'generated_caption'):
            caption = st.session_state.generated_caption
            word_count = st.session_state.get('word_count', len(caption.split()))
            beam_used = st.session_state.get('beam_used', beam_width)
            maxlen_used = st.session_state.get('maxlen_used', max_length)

            st.markdown(f"""
            <div class="caption-card">
                <div class="caption-chip">Sequence Prediction</div>
                <div class="caption-text">{caption.capitalize()}.</div>
                <div class="stats-row">
                    <div class="stat-pill">Words&nbsp;<span class="val">{word_count}</span></div>
                    <div class="stat-pill">Beam width&nbsp;<span class="val">{beam_used}</span></div>
                    <div class="stat-pill">Max tokens&nbsp;<span class="val">{maxlen_used}</span></div>
                    <div class="stat-pill">Device&nbsp;<span class="val">{'GPU' if torch.cuda.is_available() else 'CPU'}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.download_button(
                label="Export as .txt",
                data=caption,
                file_name="caption.txt",
                mime="text/plain"
            )

        else:
            st.markdown("""
            <div class="placeholder-card">
                <div class="placeholder-icon">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="#4285F4" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <div class="placeholder-title">No output yet</div>
                <div class="placeholder-body">Upload an image and press Generate Caption to see the model's prediction appear here.</div>
            </div>
            """, unsafe_allow_html=True)

            # Show model readiness info when idle
            st.markdown("""
            <div class="glass-card" style="margin-top:1.25rem;">
                <div class="section-label" style="margin-bottom:0.75rem;">Model Status</div>
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.5rem;">
                    <div style="width:8px;height:8px;border-radius:50%;background:#34A853;box-shadow:0 0 8px #34A853;flex-shrink:0;"></div>
                    <span style="font-size:0.9rem;color:#C4C7C5;">Weights loaded and ready</span>
                </div>
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.5rem;">
                    <div style="width:8px;height:8px;border-radius:50%;background:#34A853;box-shadow:0 0 8px #34A853;flex-shrink:0;"></div>
                    <span style="font-size:0.9rem;color:#C4C7C5;">Vocabulary index available</span>
                </div>
                <div style="display:flex; align-items:center; gap:10px;">
                    <div style="width:8px;height:8px;border-radius:50%;background:#{'34A853' if torch.cuda.is_available() else 'F9AB00'};box-shadow:0 0 8px #{'34A853' if torch.cuda.is_available() else 'F9AB00'};flex-shrink:0;"></div>
                    <span style="font-size:0.9rem;color:#C4C7C5;">Running on {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU — no GPU detected'}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()