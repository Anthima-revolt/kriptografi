import streamlit as st
import pandas as pd
import numpy as np
import math
from collections import Counter
from PIL import Image
import io
import matplotlib.pyplot as plt

st.markdown("""
<style>

/* ==============================
   GLOBAL BACKGROUND & ANIMATION
   ============================== */
body {
    background: radial-gradient(circle at top, #0B0F14, #020617);
    overflow-x: hidden;
    animation: pulseBG 14s ease-in-out infinite;
}

/* Floating hacker particles */
body::after {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(#00E67655 1px, transparent 1px),
        radial-gradient(#00E67622 1px, transparent 1px);
    background-size: 120px 120px, 60px 60px;
    animation: floatParticles 45s linear infinite;
    pointer-events: none;
    z-index: 0;
}

/* Animated scanline */
body::before {
    content: "";
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        to bottom,
        rgba(0, 230, 118, 0.04),
        rgba(0, 230, 118, 0.04) 1px,
        transparent 1px,
        transparent 5px
    );
    animation: scanMove 8s linear infinite;
    pointer-events: none;
    z-index: 1;
}

/* ==============================
   FOREGROUND FIX
   ============================== */
section[data-testid="stAppViewContainer"] {
    position: relative;
    z-index: 2;
}

/* ==============================
   SIDEBAR STYLE
   ============================== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B0F14, #020617);
    border-right: 1px solid #00E67633;
}

/* ==============================
   CARD / CONTAINER EFFECT
   ============================== */
div[data-testid="stVerticalBlock"] > div {
    background: #0F172A;
    border-radius: 14px;
    padding: 1.2rem;
    border: 1px solid #00E67622;
    box-shadow: 0 0 25px rgba(0, 230, 118, 0.05);
}

/* ==============================
   BUTTON GLOW
   ============================== */
button {
    background: linear-gradient(90deg, #00E676, #00C853) !important;
    color: black !important;
    border-radius: 10px !important;
    font-weight: 600;
    box-shadow: 0 0 15px rgba(0, 230, 118, 0.4);
    transition: all 0.2s ease-in-out;
}

button:hover {
    box-shadow: 0 0 25px rgba(0, 230, 118, 0.9);
    transform: scale(1.03);
}

/* ==============================
   INPUT / TEXTAREA (TERMINAL)
   ============================== */
input, textarea {
    background-color: #020617 !important;
    color: #00E676 !important;
    border: 1px solid #00E67644 !important;
}

/* ==============================
   ANIMATIONS
   ============================== */
@keyframes floatParticles {
    from {
        background-position: 0 0, 0 0;
    }
    to {
        background-position: 600px 1200px, -600px 600px;
    }
}

@keyframes scanMove {
    from {
        background-position: 0 0;
    }
    to {
        background-position: 0 200px;
    }
}

@keyframes pulseBG {
    0%, 100% {
        filter: brightness(1);
    }
    50% {
        filter: brightness(1.05);
    }
}

</style>
""", unsafe_allow_html=True)

# Import modul custom
# Pastikan file sbox_generator.py, sbox_test.py, dan aes_engine.py ada di folder yang sama
from sbox_generator import SBOXES, INV_SBOXES, AFFINE_128_DICT
from sbox_test import get_single_sbox_metrics
import aes_engine as aes

# ==========================================
# --- HELPER FUNCTIONS: IMAGE ANALYSIS ---
# ==========================================
def calculate_entropy(data_bytes):
    """Menghitung Shannon Entropy dari byte array."""
    if not data_bytes: return 0
    counts = Counter(data_bytes)
    total = len(data_bytes)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def get_correlation_coefficient_and_scatter(data, width, height, num_samples=3000):
    """
    Menghitung koefisien korelasi DAN mengembalikan data untuk scatter plot.
    Fokus pada Horizontal Correlation (sebelahan kiri-kanan).
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    
    # Ambil data visual saja
    target_len = width * height * 3 
    if len(arr) > target_len:
        arr = arr[:target_len]
    
    N = len(arr)
    if N < 2: return 0, [], []

    x_h, y_h = [], [] 

    # Random Sampling untuk Scatter Plot & Kalkulasi Cepat
    max_idx = N - 2
    if max_idx <= 0: return 0, [], []
    
    indices = np.random.choice(max_idx, min(max_idx, num_samples), replace=False)

    for i in indices:
        x_h.append(arr[i])
        y_h.append(arr[i+1])

    def calc_corr(x, y):
        if len(x) < 2: return 0
        if np.std(x) == 0 or np.std(y) == 0: return 0 
        return np.corrcoef(x, y)[0, 1]

    coeff = calc_corr(x_h, y_h)
    return coeff, x_h, y_h

def plot_histogram(data_bytes, title, color):
    """Membuat plot histogram distribusi piksel."""
    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(arr, bins=256, range=(0, 256), color=color, alpha=0.7, density=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlim([0, 256])
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    return fig

def plot_scatter(x, y, title, color):
    """Membuat scatter plot untuk korelasi piksel."""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(x, y, s=1, c=color, alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlim([0, 255])
    ax.set_ylim([0, 255])
    ax.set_xlabel("Pixel (x, y)")
    ax.set_ylabel("Pixel (x, y+1)")
    return fig

def calculate_npcr_uaci(c1_bytes, c2_bytes):
    """Menghitung NPCR dan UACI."""
    arr1 = np.frombuffer(c1_bytes, dtype=np.uint8)
    arr2 = np.frombuffer(c2_bytes, dtype=np.uint8)
    
    min_len = min(len(arr1), len(arr2))
    arr1 = arr1[:min_len]
    arr2 = arr2[:min_len]

    if min_len == 0: return 0, 0

    diff_count = np.count_nonzero(arr1 != arr2)
    npcr = (diff_count / min_len) * 100

    abs_diff = np.abs(arr1.astype(int) - arr2.astype(int))
    uaci = (np.sum(abs_diff) / (255 * min_len)) * 100

    return npcr, uaci
# ==========================================
# --- STREAMLIT APP ---
# ==========================================

st.set_page_config(layout="wide", page_title="S-Box & Image Crypto Analysis")
st.title('S-Box Construction Affine Matrix Exploration')

# 1. Select S-box
selected_sbox_name = st.selectbox("Select an S-box:", list(SBOXES.keys()))

col_affine, col_sbox = st.columns(2)
with col_affine:
    st.subheader("Affine Transformation Matrix")
    M = AFFINE_128_DICT[selected_sbox_name]

    df_affine = pd.DataFrame(
        M,
        columns=[f"b{i}" for i in range(8)],
        index=[f"r{i}" for i in range(8)]
    )
    st.dataframe(df_affine)

# 2. Function to format S-box for display
def format_sbox_grid(sbox):
    grid_str = ""
    for i in range(16):
        row = sbox[i * 16:(i + 1) * 16]
        grid_str += " ".join(f"{v:02X}" for v in row) + "\n"
    return grid_str

# 3. Display S-box grid
with col_sbox:
    st.subheader(f"Hexadecimal S-box: {selected_sbox_name}")
    if selected_sbox_name:
        selected_sbox = SBOXES[selected_sbox_name]
        st.code(format_sbox_grid(selected_sbox))

# 4 & 5. Display cryptographic properties
st.header(f"Cryptographic Properties for S-box: {selected_sbox_name}")

@st.cache_data(show_spinner=True)
def get_metrics_wrapper(sbox):
    return get_single_sbox_metrics(sbox)

current_metrics = get_metrics_wrapper(selected_sbox)
metrics_df = pd.DataFrame([current_metrics])
st.dataframe(metrics_df, use_container_width=True)

# --- Section 6: Encryption & Decryption ---
st.divider()
st.header(f"AES Text Encryption & Decryption Simulation")
st.write(f"Menggunakan S-Box: **{selected_sbox_name}**")

tab1, tab2 = st.tabs(["🔒 Encryption", "🔓 Decryption"])

with tab1:
    st.subheader("Enkripsi Plaintext")
    st.info(f"S-Box yang digunakan: {selected_sbox_name}")
    col1, col2 = st.columns(2)
    with col1:
        user_text = st.text_area("Input Plaintext:", "Halo Dunia", key="encrypt_text")
    with col2:
        user_key = st.text_input("Key (Max 16 chars):", "kuncirahasia1234", key="encrypt_key")

    if st.button("🔒 Encrypt Now", key="btn_encrypt"):
        if not user_key:
            st.error("Key tidak boleh kosong!")
        else:
            key_bytes = list(user_key.encode('utf-8').ljust(16, b'\0')[:16])
            round_keys = aes.simple_key_expansion(key_bytes, selected_sbox)
            with st.spinner("Mengenkripsi..."):
                cipher_output = aes.encrypt_full_text(user_text, round_keys, selected_sbox)
            hex_output = "".join(f"{b:02x}" for b in cipher_output)
            st.success("✅ Enkripsi Berhasil!")
            st.text_input("Ciphertext (Hexadecimal Output):", value=hex_output)

with tab2:
    st.subheader("Dekripsi Ciphertext")
    col1, col2 = st.columns(2)
    with col1:
        cipher_input = st.text_area("Input Ciphertext (Hexadecimal):", key="decrypt_cipher_hex", height=100)
    with col2:
        decrypt_key = st.text_input("Decryption Key:", "kuncirahasia1234", key="decrypt_key")

    if st.button("🔓 Decrypt Now", key="btn_decrypt"):
        if not decrypt_key or not cipher_input:
            st.error("Input tidak boleh kosong!")
        else:
            try:
                hex_clean = cipher_input.replace(" ", "").replace("\n", "").strip()
                cipher_bytes = bytes.fromhex(hex_clean)
                if len(cipher_bytes) % 16 != 0:
                    st.error(f"❌ Panjang Hex tidak valid.")
                else:
                    key_bytes = list(decrypt_key.encode('utf-8').ljust(16, b'\0')[:16])
                    round_keys = aes.simple_key_expansion(key_bytes, selected_sbox)
                    inv_sbox = INV_SBOXES[selected_sbox_name]
                    plaintext_full = []
                    for i in range(0, len(cipher_bytes), 16):
                        block = list(cipher_bytes[i:i + 16])
                        decrypted_block = aes.aes_decrypt_block(block, 10, round_keys, inv_sbox)
                        plaintext_full.extend(decrypted_block)
                    plaintext_bytes = bytes(plaintext_full)
                    unpadded = aes.unpad_text(plaintext_bytes)
                    st.success("✅ Dekripsi Berhasil!")
                    st.text_area("Plaintext Output:", value=unpadded.decode('utf-8', errors='replace'), height=100)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.divider()
st.header("🖼️ Image Encryption & Security Analysis")
st.write(f"Menggunakan S-Box: **{selected_sbox_name}**")

tab_img1, tab_img2 = st.tabs(["🔒 Encrypt & Analyze", "🔓 Decrypt Image"])

st.header("🖼️ Image Encryption, Histogram & Correlation Analysis")
st.write(f"Menggunakan S-Box: **{selected_sbox_name}**")

tab_img1, tab_img2 = st.tabs(["🔒 Encrypt & Analyze", "🔓 Decrypt Image"])

with tab_img1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info("Upload gambar untuk melihat analisis Histogram dan Correlation Scatter Plot perbandingan antara Plain Image dan Cipher Image.")
        uploaded_img = st.file_uploader("Upload Gambar:", type=['png', 'jpg', 'jpeg'], key="img_encrypt")
        img_key = st.text_input("Encryption Key:", "kuncirahasia1234", key="img_encrypt_key")
        run_analysis = st.checkbox("Jalankan Analisis NPCR & UACI (Lebih lambat)", value=True)
    
    with col2:
        if uploaded_img:
            st.image(uploaded_img, caption="Preview Input", width=300)

    if st.button("🔒 Encrypt & Generate Analysis", key="btn_encrypt_img"):
        if not img_key or uploaded_img is None:
            st.error("Key dan Gambar wajib diisi!")
        else:
            with st.spinner("Processing Encryption & Generating Plots..."):
                # --- A. LOAD & PREPARE ---
                img = Image.open(uploaded_img).convert("RGB")
                img_bytes = img.tobytes()
                mode, size = img.mode, img.size
                
                key_bytes = list(img_key.encode('utf-8').ljust(16, b'\0')[:16])
                round_keys = aes.simple_key_expansion(key_bytes, selected_sbox)

                # Padding
                padding_len = 16 - (len(img_bytes) % 16)
                padded_img = img_bytes + bytes([padding_len] * padding_len)

                # --- B. ENCRYPT (C1) ---
                encrypted_bytes_c1 = []
                for i in range(0, len(padded_img), 16):
                    block = list(padded_img[i:i + 16])
                    encrypted_block = aes.aes_encrypt_block(block, 10, round_keys, selected_sbox)
                    encrypted_bytes_c1.extend(encrypted_block)
                c1_bytes = bytes(encrypted_bytes_c1)

                # --- C. ANALISIS VISUAL (Histogram & Scatter) ---
                # 1. Plain Image Analysis
                plain_corr, plain_x, plain_y = get_correlation_coefficient_and_scatter(img_bytes, size[0], size[1])
                fig_hist_plain = plot_histogram(img_bytes, "Histogram: Plain Image", "green")
                fig_scat_plain = plot_scatter(plain_x, plain_y, f"Correlation: Plain (Coeff: {plain_corr:.4f})", "green")
                
                # 2. Cipher Image Analysis
                cipher_corr, cipher_x, cipher_y = get_correlation_coefficient_and_scatter(c1_bytes, size[0], size[1])
                fig_hist_cipher = plot_histogram(c1_bytes, "Histogram: Cipher Image", "red")
                fig_scat_cipher = plot_scatter(cipher_x, cipher_y, f"Correlation: Cipher (Coeff: {cipher_corr:.4f})", "red")
                
                # --- D. ANALISIS NUMERIK (Entropy, NPCR, UACI) ---
                analysis_metrics = {}
                analysis_metrics["Entropy (Plain)"] = calculate_entropy(img_bytes)
                analysis_metrics["Entropy (Cipher)"] = calculate_entropy(c1_bytes)
                analysis_metrics["Correlation Coeff (Plain)"] = plain_corr
                analysis_metrics["Correlation Coeff (Cipher)"] = cipher_corr
                
                if run_analysis:
                    # Enkripsi kedua untuk NPCR (Flip 1 bit)
                    p2_list = list(padded_img)
                    p2_list[-1] = p2_list[-1] ^ 1
                    encrypted_bytes_c2 = []
                    for i in range(0, len(p2_list), 16):
                        encrypted_block = aes.aes_encrypt_block(list(p2_list[i:i+16]), 10, round_keys, selected_sbox)
                        encrypted_bytes_c2.extend(encrypted_block)
                    c2_bytes = bytes(encrypted_bytes_c2)
                    
                    npcr, uaci = calculate_npcr_uaci(c1_bytes, c2_bytes)
                    analysis_metrics["NPCR"] = npcr
                    analysis_metrics["UACI"] = uaci

                # --- E. DISPLAY RESULTS ---
                st.success("✅ Encryption Complete!")

                # Row 1: Images Visual
                st.subheader("1. Visual Comparison")
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    st.image(img, caption="Plain Image", use_column_width=True)
                with col_v2:
                    # Create visualization from cipher bytes
                    try:
                        display_bytes = c1_bytes[:size[0]*size[1]*3]
                        encrypted_img_viz = Image.frombytes('RGB', size, display_bytes, 'raw')
                        st.image(encrypted_img_viz, caption="Cipher Image (Visualized)", use_column_width=True)
                    except:
                        st.warning("Visualisasi Cipher gagal (size mismatch).")

                # Row 2: Histograms
                st.subheader("2. Histogram Analysis")
                st.caption("Histogram yang ideal untuk Cipher Image harus datar (uniform distribution), menandakan serangan statistik sulit dilakukan.")
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    st.pyplot(fig_hist_plain)
                with col_h2:
                    st.pyplot(fig_hist_cipher)

                # Row 3: Correlation Scatter Plots
                st.subheader("3. Correlation Analysis (Horizontal)")
                st.caption("Plain image biasanya memiliki pola diagonal (korelasi tinggi). Cipher image harus menyebar acak (korelasi mendekati 0).")
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.pyplot(fig_scat_plain)
                with col_c2:
                    st.pyplot(fig_scat_cipher)

                # Row 4: Numeric Metrics
                st.subheader("4. Numeric Security Metrics")
                
                # Format metrics for nice display
                m_df = pd.DataFrame({
                    "Metric": ["Entropy", "Correlation Coefficient", "NPCR", "UACI"],
                    "Plain Image": [
                        f"{analysis_metrics.get('Entropy (Plain)', 0):.5f}",
                        f"{analysis_metrics.get('Correlation Coeff (Plain)', 0):.5f}",
                        "-", "-"
                    ],
                    "Cipher Image": [
                        f"{analysis_metrics.get('Entropy (Cipher)', 0):.5f}",
                        f"{analysis_metrics.get('Correlation Coeff (Cipher)', 0):.5f}",
                        f"{analysis_metrics.get('NPCR', 0):.4f} %" if run_analysis else "-",
                        f"{analysis_metrics.get('UACI', 0):.4f} %" if run_analysis else "-"
                    ],
                    "Ideal Value": [
                        "~7.999", "~0.0", ">99.60%", "~33.46%"
                    ]
                })
                st.table(m_df)

                # Download
                metadata = f"{mode}|{size[0]}|{size[1]}|".encode('utf-8')
                full_encrypted = metadata + c1_bytes
                st.download_button("⬇️ Download Encrypted File (.enc)", full_encrypted, "encrypted_image.enc", "application/octet-stream")

with tab_img2:
    st.subheader("Dekripsi Gambar")
    col1, col2 = st.columns(2)
    with col1:
        encrypted_file = st.file_uploader("Upload File .enc:", type=['enc'], key="img_decrypt")
        img_decrypt_key = st.text_input("Decryption Key:", "kuncirahasia1234", key="img_decrypt_key")

    if st.button("🔓 Decrypt Image", key="btn_decrypt_img"):
        if not img_decrypt_key or encrypted_file is None:
            st.error("Data tidak lengkap!")
        else:
            try:
                with st.spinner("Mendekripsi..."):
                    encrypted_data = encrypted_file.read()
                    
                    # Parsing Metadata
                    metadata_end = encrypted_data.index(b'|', encrypted_data.index(b'|', encrypted_data.index(b'|') + 1) + 1) + 1
                    metadata = encrypted_data[:metadata_end].decode('utf-8')
                    mode, width, height = metadata.rstrip('|').split('|')
                    width, height = int(width), int(height)
                    
                    cipher_bytes_only = encrypted_data[metadata_end:]
                    
                    key_bytes = list(img_decrypt_key.encode('utf-8').ljust(16, b'\0')[:16])
                    round_keys = aes.simple_key_expansion(key_bytes, selected_sbox)
                    inv_sbox = INV_SBOXES[selected_sbox_name]
                    
                    decrypted_bytes = []
                    for i in range(0, len(cipher_bytes_only), 16):
                        block = list(cipher_bytes_only[i:i + 16])
                        decrypted_block = aes.aes_decrypt_block(block, 10, round_keys, inv_sbox)
                        decrypted_bytes.extend(decrypted_block)
                    
                    decrypted_bytes = bytes(decrypted_bytes)
                    padding_len = decrypted_bytes[-1]
                    img_bytes = decrypted_bytes[:-padding_len]
                    
                    decrypted_img = Image.frombytes(mode, (width, height), img_bytes)
                    
                    st.image(decrypted_img, caption="Hasil Dekripsi", use_column_width=True)
                    
                    buf = io.BytesIO()
                    decrypted_img.save(buf, format='PNG')
                    buf.seek(0)
                    st.download_button("⬇️ Download Gambar PNG", buf, "decrypted.png", "image/png")
                    st.success("✅ Berhasil!")
            except Exception as e:
                st.error(f"Gagal: {str(e)}")
