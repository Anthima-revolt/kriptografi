import streamlit as st
import pandas as pd
from PIL import Image
import io

# Import modul custom
from sbox_generator import SBOXES, INV_SBOXES, AFFINE_128_DICT
from sbox_test import get_single_sbox_metrics
import aes_engine as aes

# --- Streamlit app layout ---
st.set_page_config(layout="wide")
st.title('S-Box Cryptographic Properties Analysis')

# 1. Select S-box
selected_sbox_name = st.selectbox("Select an S-box:", list(SBOXES.keys()))

st.header("Affine Transformation Matrix (8×8)")
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
st.header(f"16x16 Hexadecimal Grid for S-box: {selected_sbox_name}")
if selected_sbox_name:
    selected_sbox = SBOXES[selected_sbox_name]
    st.code(format_sbox_grid(selected_sbox))

# 4 & 5. Display cryptographic properties
st.header(f"Cryptographic Properties for S-box: {selected_sbox_name}")

# Wrapper agar caching streamlit bekerja dengan baik jika diinginkan
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

# Tabs untuk Enkripsi dan Dekripsi
tab1, tab2 = st.tabs(["🔒 Encryption", "🔓 Decryption"])

with tab1:
    st.subheader("Enkripsi Plaintext")
    st.info(f"S-Box yang digunakan: {selected_sbox_name}")

    col1, col2 = st.columns(2)
    with col1:
        user_text = st.text_area("Input Plaintext (Teks Biasa):", "Halo Dunia", key="encrypt_text")
    with col2:
        user_key = st.text_input("Encryption Key (Max 16 chars):", "kuncirahasia1234", key="encrypt_key")

    if st.button("🔒 Encrypt Now", key="btn_encrypt"):
        if not user_key:
            st.error("Key tidak boleh kosong!")
        else:
            key_bytes = list(user_key.encode('utf-8').ljust(16, b'\0')[:16])
            round_keys = aes.simple_key_expansion(key_bytes, selected_sbox)

            with st.spinner("Mengenkripsi..."):
                cipher_output = aes.encrypt_full_text(user_text, round_keys, selected_sbox)

            # Output hanya Hexadecimal
            hex_output = "".join(f"{b:02x}" for b in cipher_output)
            st.success("✅ Enkripsi Berhasil!")
            st.text_input("Ciphertext (Hexadecimal Output):", value=hex_output)

with tab2:
    st.subheader("Dekripsi Ciphertext")
    st.info(f"S-Box yang akan digunakan: {selected_sbox_name} (Pastikan sama dengan enkripsi)")

    col1, col2 = st.columns(2)
    with col1:
        cipher_input = st.text_area("Input Ciphertext (Hexadecimal Only):", key="decrypt_cipher_hex", height=100)
    with col2:
        decrypt_key = st.text_input("Decryption Key (Max 16 chars):", "kuncirahasia1234", key="decrypt_key")

    if st.button("🔓 Decrypt Now", key="btn_decrypt"):
        if not decrypt_key:
            st.error("Key tidak boleh kosong!")
        elif not cipher_input:
            st.error("Ciphertext tidak boleh kosong!")
        else:
            try:
                hex_clean = cipher_input.replace(" ", "").replace("\n", "").strip()
                cipher_bytes = bytes.fromhex(hex_clean)

                if len(cipher_bytes) % 16 != 0:
                    st.error(f"❌ Panjang Hex tidak valid (harus kelipatan 32 karakter hex).")
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
                    plaintext = unpadded.decode('utf-8', errors='replace')

                    st.success("✅ Dekripsi Berhasil!")
                    st.text_area("Plaintext Output:", value=plaintext, height=100)

            except ValueError:
                st.error("❌ Format Hex tidak valid. Pastikan hanya berisi karakter 0-9 dan a-f.")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {str(e)}")

st.divider()
st.header("🖼️ Image Encryption & Decryption")
st.write(f"Menggunakan S-Box: **{selected_sbox_name}**")

tab_img1, tab_img2 = st.tabs(["🔒 Encrypt Image", "🔓 Decrypt Image"])

with tab_img1:
    st.subheader("Enkripsi Gambar")
    st.info(f"S-Box yang digunakan: {selected_sbox_name}")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_img = st.file_uploader("Upload Gambar:", type=['png', 'jpg', 'jpeg', 'bmp'], key="img_encrypt")
        img_key = st.text_input("Encryption Key (Max 16 chars):", "kuncirahasia1234", key="img_encrypt_key")

    with col2:
        if uploaded_img is not None:
            img = Image.open(uploaded_img)
            st.image(img, caption="Original Image", use_column_width=True)

    if st.button("🔒 Encrypt Image", key="btn_encrypt_img"):
        if not img_key:
            st.error("Key tidak boleh kosong!")
        elif uploaded_img is None:
            st.error("Silakan upload gambar terlebih dahulu!")
        else:
            with st.spinner("Mengenkripsi gambar..."):
                img = Image.open(uploaded_img)
                img_bytes = img.tobytes()
                mode = img.mode
                size = img.size

                key_bytes = list(img_key.encode('utf-8').ljust(16, b'\0')[:16])
                round_keys = aes.simple_key_expansion(key_bytes, selected_sbox)

                padding_len = 16 - (len(img_bytes) % 16)
                padded_img = img_bytes + bytes([padding_len] * padding_len)

                encrypted_bytes = []
                for i in range(0, len(padded_img), 16):
                    block = list(padded_img[i:i + 16])
                    encrypted_block = aes.aes_encrypt_block(block, 10, round_keys, selected_sbox)
                    encrypted_bytes.extend(encrypted_block)

                metadata = f"{mode}|{size[0]}|{size[1]}|".encode('utf-8')
                full_encrypted = metadata + bytes(encrypted_bytes)

                encrypted_img = Image.frombytes('RGB', size, bytes(encrypted_bytes[:size[0] * size[1] * 3]), 'raw')

                col1, col2 = st.columns(2)
                with col1:
                    st.image(encrypted_img, caption="Encrypted Image (Visualization)", use_column_width=True)
                with col2:
                    st.download_button(
                        label="⬇️ Download Encrypted Image",
                        data=full_encrypted,
                        file_name="encrypted_image.enc",
                        mime="application/octet-stream"
                    )
                st.success("✅ Gambar berhasil dienkripsi!")

with tab_img2:
    st.subheader("Dekripsi Gambar")
    st.info(f"S-Box yang akan digunakan: {selected_sbox_name}")

    col1, col2 = st.columns(2)
    with col1:
        encrypted_file = st.file_uploader("Upload Encrypted File (.enc):", type=['enc'], key="img_decrypt")
        img_decrypt_key = st.text_input("Decryption Key (Max 16 chars):", "kuncirahasia1234", key="img_decrypt_key")

    if st.button("🔓 Decrypt Image", key="btn_decrypt_img"):
        if not img_decrypt_key:
            st.error("Key tidak boleh kosong!")
        elif encrypted_file is None:
            st.error("Silakan upload file terenkripsi terlebih dahulu!")
        else:
            try:
                with st.spinner("Mendekripsi gambar..."):
                    encrypted_data = encrypted_file.read()

                    metadata_end = encrypted_data.index(b'|', encrypted_data.index(b'|', encrypted_data.index(b'|') + 1) + 1) + 1
                    metadata = encrypted_data[:metadata_end].decode('utf-8')
                    mode, width, height = metadata.rstrip('|').split('|')
                    width, height = int(width), int(height)

                    encrypted_bytes = encrypted_data[metadata_end:]

                    key_bytes = list(img_decrypt_key.encode('utf-8').ljust(16, b'\0')[:16])
                    round_keys = aes.simple_key_expansion(key_bytes, selected_sbox)
                    inv_sbox = INV_SBOXES[selected_sbox_name]

                    decrypted_bytes = []
                    for i in range(0, len(encrypted_bytes), 16):
                        block = list(encrypted_bytes[i:i + 16])
                        decrypted_block = aes.aes_decrypt_block(block, 10, round_keys, inv_sbox)
                        decrypted_bytes.extend(decrypted_block)

                    decrypted_bytes = bytes(decrypted_bytes)
                    padding_len = decrypted_bytes[-1]
                    img_bytes = decrypted_bytes[:-padding_len]

                    decrypted_img = Image.frombytes(mode, (width, height), img_bytes)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(decrypted_img, caption="Original Image", use_column_width=True)
                    with col2:
                        buf = io.BytesIO()
                        decrypted_img.save(buf, format='PNG')
                        buf.seek(0)
                        st.download_button(
                            label="⬇️ Download Decrypted Image",
                            data=buf,
                            file_name="decrypted_image.png",
                            mime="image/png"
                        )
                    st.success("✅ Gambar berhasil didekripsi!")

            except Exception as e:
                st.error(f"❌ Gagal mendekripsi: {str(e)}")