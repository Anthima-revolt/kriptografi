import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image
import io


# --- Re-define Matrix and S-box generation functions and data ---
def mat(*rows):
    return np.array([[int(b) for b in row] for row in rows], dtype=np.uint8)


KAES = mat(
    "10001111", "11000111", "11100011", "11110001",
    "11111000", "01111100", "00111110", "00011111"
)

# K1 - K128 (Defining a subset for brevity in logic, using the original data)
K1 = mat("00000001", "10000000", "01000000", "00100000", "00010000", "00001000", "00000100", "00000001")
K2 = mat("00000010", "00000001", "10000000", "01000000", "00100000", "00010000", "00001000", "00000100")
K3 = mat("00000100", "00000010", "00000001", "10000000", "01000000", "00100000", "00010000", "00001000")
K4 = mat("00000111", "10000011", "11000001", "11100000", "01110000", "00111000", "00011100", "00001111")
K5 = mat("00001000", "00000100", "00000010", "00000001", "10000000", "01000000", "00100000", "00010000")
K6 = mat("00001011", "10000101", "11000010", "01100001", "10110000", "01011000", "00101100", "00010110")
K7 = mat("00001101", "10000110", "01000011", "10100001", "11010000", "01101000", "00110100", "00011010")
K8 = mat("00001110", "00000111", "10000011", "11110001", "11111000", "01111100", "00111110", "00011111")
K9 = mat("00010000", "00001000", "00000100", "00000010", "00000001", "10000000", "01000000", "00100000")
K10 = mat("00010011", "10001001", "11000100", "01100010", "00110001", "10011000", "01001100", "00100110")
K11 = mat("00010101", "10001010", "01000101", "10100010", "01010001", "10101000", "01010100", "00101010")
K12 = mat("00011001", "10001100", "01000110", "00100011", "10010001", "11001000", "01100100", "00110010")
K13 = mat("10001111", "11000111", "11100011", "11110001", "11111000", "01111100", "00111110", "00011111")
K14 = mat("00011010", "00001101", "10000110", "01000011", "10100001", "11010000", "01101000", "00110100")
K15 = mat("00011100", "00001110", "00000111", "10000011", "11000001", "11100000", "01110000", "00111000")
K16 = mat("00011111", "10001111", "11000111", "11100011", "11110001", "11111000", "01111100", "00111110")
K17 = mat("00100000", "00010000", "00001000", "00000100", "00000010", "00000001", "10000000", "01000000")
K18 = mat("00100011", "10010001", "11001000", "11100100", "01110010", "00111001", "10011100", "01001110")
K19 = mat("00100101", "10010010", "01001001", "10100100", "01010010", "00101001", "10010100", "01001010")
K20 = mat("00100110", "00010011", "10001001", "11000100", "01100010", "00110001", "10011000", "01001100")
K21 = mat("00101001", "10010100", "01001010", "00100101", "10010010", "01001001", "10100100", "01010010")
K22 = mat("00101010", "00010101", "10001010", "01000101", "10100010", "01010001", "10101000", "01010100")
K44 = mat("01010111", "10101011", "11010101", "11101010", "01110101", "10111010", "01011101", "10101110")
K81 = mat("10100001", "11010000", "01101000", "00110100", "00011010", "00001101", "10000110", "01000011")
K111 = mat("11011100", "01101110", "00110111", "10011011", "11001101", "11100110", "01110011", "10111001")
K127 = mat("11111101", "11111110", "01111111", "10111111", "11011111", "11101111", "11110111", "11111011")
K128 = mat("11111110", "01111111", "10111111", "11011111", "11101111", "11110111", "11111011", "11111101")

AFFINE_MATRICES = {
    "AES": KAES,
    "K1": K1, "K2": K2, "K3": K3, "K4": K4, "K5": K5, "K6": K6, "K7": K7, "K8": K8,
    "K9": K9, "K10": K10, "K11": K11, "K12": K12, "K13": K13, "K14": K14, "K15": K15, "K16": K16,
    "K17": K17, "K18": K18, "K19": K19, "K20": K20, "K21": K21, "K22": K22,
    "K44": K44, "K81": K81, "K111": K111, "K127": K127, "K128": K128
}


def is_invertible_gf2(M):
    M = M.copy()
    n = 8
    rank = 0
    for col in range(n):
        pivot = None
        for row in range(rank, n):
            if M[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for r in range(n):
            if r != rank and M[r, col] == 1:
                M[r] ^= M[rank]
        rank += 1
    return rank == n


initial_affines = {name: M for name, M in AFFINE_MATRICES.items()}


def generate_remaining_affines(existing_dict, target_total=128):
    matrices = list(existing_dict.values())
    while len(matrices) < target_total:
        M = np.random.randint(0, 2, (8, 8), dtype=np.uint8)
        if not is_invertible_gf2(M): continue
        if np.array_equal(M, KAES): continue
        if any(np.array_equal(M, X) for X in matrices): continue
        matrices.append(M)
    return matrices


affine_128 = generate_remaining_affines(initial_affines, target_total=128)
AFFINE_128_DICT = {}
for name, M in initial_affines.items(): AFFINE_128_DICT[name] = M
idx = 1
for M in affine_128:
    if any(np.array_equal(M, X) for X in initial_affines.values()): continue
    AFFINE_128_DICT[f"G{idx}"] = M
    idx += 1

AES_CONSTANT = 0x63
IRREDUCIBLE_POLY = 0x11B


def gf_multiply(a, b):
    p = 0
    for i in range(8):
        if b & 1: p ^= a
        hi_bit_set = a & 0x80
        a <<= 1
        if hi_bit_set: a ^= IRREDUCIBLE_POLY
        b >>= 1
    return p & 0xFF


def generate_inverse_table():
    inv = [0] * 256
    for i in range(1, 256):
        for j in range(1, 256):
            if gf_multiply(i, j) == 1:
                inv[i] = j
                break
    return inv


INVERSE_TABLE = generate_inverse_table()


def byte_to_bits(byte_val):
    return np.array([(byte_val >> i) & 1 for i in range(8)], dtype=int)


def bits_to_byte(bits):
    val = 0
    for i in range(8):
        if bits[i]: val |= (1 << i)
    return val


def create_sbox(matrix, constant):
    sbox = []
    matrix_np = np.array(matrix)
    for x in range(256):
        inv_x = INVERSE_TABLE[x]
        inv_bits = byte_to_bits(inv_x)
        trans_bits = matrix_np.dot(inv_bits) % 2
        final_val = bits_to_byte(trans_bits) ^ constant
        sbox.append(final_val)
    return sbox


def create_inverse_sbox(sbox):
    inv_sbox = [0] * 256
    for i in range(256):
        inv_sbox[sbox[i]] = i
    return inv_sbox


SBOXES = {}
SBOXES["AES"] = create_sbox(KAES, AES_CONSTANT)
for name, M in AFFINE_128_DICT.items():
    SBOXES[name] = create_sbox(M, AES_CONSTANT)

INV_SBOXES = {}
for name, sbox in SBOXES.items():
    INV_SBOXES[name] = create_inverse_sbox(sbox)


# --- Cryptographic property calculation functions ---
def walsh_hadamard_transform(func):
    n = 8
    wf = [(-1) ** func[x] for x in range(256)]
    h = 1
    while h < 256:
        for i in range(0, 256, h * 2):
            for j in range(i, i + h):
                x = wf[j]
                y = wf[j + h]
                wf[j] = x + y
                wf[j + h] = x - y
        h *= 2
    return wf


def calculate_nl(sbox):
    min_nl = 256
    for i in range(8):
        func = [(sbox[x] >> i) & 1 for x in range(256)]
        spectrum = walsh_hadamard_transform(func)
        max_abs_wh = max(abs(val) for val in spectrum)
        nl = (2 ** 8 - max_abs_wh) // 2
        if nl < min_nl: min_nl = nl
    return int(min_nl)


def calculate_sac(sbox):
    total_avg_hamming_weight = 0
    for i in range(8):
        diff_hamming_weights = []
        for x in range(256):
            flipped_input = x ^ (1 << i)
            output_diff = sbox[x] ^ sbox[flipped_input]
            diff_hamming_weights.append(bin(output_diff).count('1'))
        total_avg_hamming_weight += sum(diff_hamming_weights) / 256.0
    return total_avg_hamming_weight / 8.0


def calculate_bic_nl(sbox):
    min_bic_nl = 256
    for i in range(8):
        for j in range(i + 1, 8):
            func_xor = [((sbox[x] >> i) & 1) ^ ((sbox[x] >> j) & 1) for x in range(256)]
            spectrum = walsh_hadamard_transform(func_xor)
            max_abs_wh = max(abs(val) for val in spectrum)
            nl = (2 ** 8 - max_abs_wh) // 2
            if nl < min_bic_nl: min_bic_nl = nl
    return int(min_bic_nl)


def calculate_bic_sac(sbox):
    total_sac = 0
    count = 0
    for i in range(8):
        for j in range(i + 1, 8):
            for inp in range(256):
                val_orig = ((sbox[inp] >> i) & 1) ^ ((sbox[inp] >> j) & 1)
                for bit in range(8):
                    flipped = inp ^ (1 << bit)
                    val_flip = ((sbox[flipped] >> i) & 1) ^ ((sbox[flipped] >> j) & 1)
                    if val_orig != val_flip: total_sac += 1
                    count += 1
    if count == 0: return 0
    return total_sac / count


def calculate_lap(sbox):
    max_spectrum = 0
    for i in range(1, 256):
        func = [0] * 256
        for x in range(256):
            val = sbox[x] & i
            func[x] = bin(val).count('1') % 2
        spectrum = walsh_hadamard_transform(func)
        current_max = max(abs(val) for val in spectrum if val != 256)
        if current_max > max_spectrum: max_spectrum = current_max
    return max_spectrum / 256.0


def calculate_dap(sbox):
    max_diff = 0
    for dx in range(1, 256):
        counts = Counter()
        for x in range(256):
            dy = sbox[x] ^ sbox[x ^ dx]
            counts[dy] += 1
        current_max = max(counts.values())
        if current_max > max_diff: max_diff = current_max
    return max_diff / 256.0


def calculate_du(sbox):
    max_diff = 0
    for dx in range(1, 256):
        counts = {}
        for x in range(256):
            dy = sbox[x] ^ sbox[x ^ dx]
            counts[dy] = counts.get(dy, 0) + 1
        current_max = max(counts.values())
        if current_max > max_diff: max_diff = current_max
    return int(max_diff)


def calculate_ad(sbox):
    max_deg = 0
    for i in range(8):
        func = [(sbox[x] >> i) & 1 for x in range(256)]
        anf = list(func)
        for step in range(1, 256):
            if step & (step - 1) == 0:
                for j in range(0, 256, step * 2):
                    for k in range(j, j + step):
                        anf[k + step] ^= anf[k]
        deg = 0
        for x in range(256):
            if anf[x] == 1:
                hw = bin(x).count('1')
                if hw > deg: deg = hw
        if deg > max_deg: max_deg = deg
    return int(max_deg)


def calculate_to(sbox):
    N = 256
    all_autocorr = []
    for a in range(1, N):
        current_sum = 0
        for v in range(N):
            correlation = 0
            for x in range(N):
                if bin((sbox[x] ^ sbox[x ^ a]) & v).count('1') % 2 == 0:
                    correlation += 1
                else:
                    correlation -= 1
            current_sum += abs(correlation)
        all_autocorr.append(current_sum)
    to_score = max(all_autocorr) / (N ** 2)
    return round(to_score, 6)


def calculate_ci(sbox):
    N = 256
    n = 8
    possible_t = []
    for i in range(8):
        func = [(sbox[x] >> i) & 1 for x in range(N)]
        wf = walsh_hadamard_transform(func)
        current_t = 0
        for t in range(1, n + 1):
            is_imun = True
            for mask in range(1, N):
                if bin(mask).count('1') == t:
                    if wf[mask] != 0:
                        is_imun = False
                        break
            if is_imun:
                current_t = t
            else:
                break
        possible_t.append(current_t)
    return int(min(possible_t))


@st.cache_data(show_spinner=True)
def get_single_sbox_metrics(sbox):
    return {
        "NL": calculate_nl(sbox),
        "SAC": calculate_sac(sbox),
        "BIC_NL": calculate_bic_nl(sbox),
        "BIC_SAC": calculate_bic_sac(sbox),
        "LAP": calculate_lap(sbox),
        "DAP": calculate_dap(sbox),
        "DU": calculate_du(sbox),
        "AD": calculate_ad(sbox),
        "TO": calculate_to(sbox),
        "CI": calculate_ci(sbox)
    }


# --- AES Implementation Functions ---

def sub_bytes(state, sbox):
    for i in range(len(state)): state[i] = sbox[state[i]]


def inv_sub_bytes(state, inv_sbox):
    for i in range(len(state)): state[i] = inv_sbox[state[i]]


def shift_rows(state):
    res = [0] * 16
    res[0], res[4], res[8], res[12] = state[0], state[4], state[8], state[12]
    res[1], res[5], res[9], res[13] = state[5], state[9], state[13], state[1]
    res[2], res[6], res[10], res[14] = state[10], state[14], state[2], state[6]
    res[3], res[7], res[11], res[15] = state[15], state[3], state[7], state[11]
    return res


def inv_shift_rows(state):
    res = [0] * 16
    res[0], res[4], res[8], res[12] = state[0], state[4], state[8], state[12]
    res[1], res[5], res[9], res[13] = state[13], state[1], state[5], state[9]
    res[2], res[6], res[10], res[14] = state[10], state[14], state[2], state[6]
    res[3], res[7], res[11], res[15] = state[7], state[11], state[15], state[3]
    return res


def gmul(a, b):
    p = 0
    for _ in range(8):
        if b & 1: p ^= a
        hi_bit_set = a & 0x80
        a <<= 1
        if hi_bit_set: a ^= 0x1B
        b >>= 1
    return p & 0xFF


def mix_columns(state):
    new_state = [0] * 16
    for i in range(4):
        col = state[i * 4: i * 4 + 4]
        new_state[i * 4] = gmul(0x02, col[0]) ^ gmul(0x03, col[1]) ^ col[2] ^ col[3]
        new_state[i * 4 + 1] = col[0] ^ gmul(0x02, col[1]) ^ gmul(0x03, col[2]) ^ col[3]
        new_state[i * 4 + 2] = col[0] ^ col[1] ^ gmul(0x02, col[2]) ^ gmul(0x03, col[3])
        new_state[i * 4 + 3] = gmul(0x03, col[0]) ^ col[1] ^ col[2] ^ gmul(0x02, col[3])
    return new_state


def inv_mix_columns(state):
    new_state = [0] * 16
    for i in range(4):
        col = state[i * 4: i * 4 + 4]
        new_state[i * 4] = gmul(0x0e, col[0]) ^ gmul(0x0b, col[1]) ^ gmul(0x0d, col[2]) ^ gmul(0x09, col[3])
        new_state[i * 4 + 1] = gmul(0x09, col[0]) ^ gmul(0x0e, col[1]) ^ gmul(0x0b, col[2]) ^ gmul(0x0d, col[3])
        new_state[i * 4 + 2] = gmul(0x0d, col[0]) ^ gmul(0x09, col[1]) ^ gmul(0x0e, col[2]) ^ gmul(0x0b, col[3])
        new_state[i * 4 + 3] = gmul(0x0b, col[0]) ^ gmul(0x0d, col[1]) ^ gmul(0x09, col[2]) ^ gmul(0x0e, col[3])
    return new_state


def aes_encrypt_block(plaintext_block, rounds, round_keys, sbox):
    state = [plaintext_block[i] ^ round_keys[0][i] for i in range(16)]
    for r in range(1, rounds):
        sub_bytes(state, sbox)
        state = shift_rows(state)
        state = mix_columns(state)
        rk = round_keys[r]
        state = [state[i] ^ rk[i] for i in range(16)]
    sub_bytes(state, sbox)
    state = shift_rows(state)
    rk = round_keys[rounds]
    state = [state[i] ^ rk[i] for i in range(16)]
    return state


def aes_decrypt_block(ciphertext_block, rounds, round_keys, inv_sbox):
    state = [ciphertext_block[i] ^ round_keys[rounds][i] for i in range(16)]
    state = inv_shift_rows(state)
    inv_sub_bytes(state, inv_sbox)
    for r in range(rounds - 1, 0, -1):
        rk = round_keys[r]
        state = [state[i] ^ rk[i] for i in range(16)]
        state = inv_mix_columns(state)
        state = inv_shift_rows(state)
        inv_sub_bytes(state, inv_sbox)
    state = [state[i] ^ round_keys[0][i] for i in range(16)]
    return state


def simple_key_expansion(key_16_bytes, sbox):
    keys = []
    for i in range(11):
        keys.append([(b + i) % 256 for b in key_16_bytes])
    return keys


def pad_text(text):
    padding_len = 16 - (len(text) % 16)
    return text.encode('utf-8') + bytes([padding_len] * padding_len)


def unpad_text(padded_bytes):
    if len(padded_bytes) == 0: return b''
    padding_len = padded_bytes[-1]
    if padding_len > 16 or padding_len == 0: return padded_bytes
    if len(padded_bytes) >= padding_len:
        padding_bytes = padded_bytes[-padding_len:]
        if all(b == padding_len for b in padding_bytes):
            return padded_bytes[:-padding_len]
    return padded_bytes


def encrypt_full_text(plaintext, round_keys, sbox):
    padded_data = pad_text(plaintext)
    ciphertext_full = []
    for i in range(0, len(padded_data), 16):
        block = list(padded_data[i:i + 16])
        encrypted_block = aes_encrypt_block(block, 10, round_keys, sbox)
        ciphertext_full.extend(encrypted_block)
    return ciphertext_full


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
current_metrics = get_single_sbox_metrics(selected_sbox)
metrics_df = pd.DataFrame([current_metrics])
st.dataframe(metrics_df, use_container_width=True)

# --- Section 6: Encryption & Decryption ---
st.divider()
st.header(f"AES Text Encryption & Decryption Simulation")
st.write(f"Menggunakan S-Box: **{selected_sbox_name}**")

# Tabs untuk Enkripsi dan Dekripsi (Tanpa Test/Round-Trip)
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
            round_keys = simple_key_expansion(key_bytes, selected_sbox)

            with st.spinner("Mengenkripsi..."):
                cipher_output = encrypt_full_text(user_text, round_keys, selected_sbox)

            # Output hanya Hexadecimal
            hex_output = "".join(f"{b:02x}" for b in cipher_output)
            st.success("✅ Enkripsi Berhasil!")
            st.text_input("Ciphertext (Hexadecimal Output):", value=hex_output)

with tab2:
    st.subheader("Dekripsi Ciphertext")
    st.info(f"S-Box yang akan digunakan: {selected_sbox_name} (Pastikan sama dengan enkripsi)")

    col1, col2 = st.columns(2)
    with col1:
        # Input hanya Hexadecimal
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
                # Membersihkan input hex dan convert ke bytes
                hex_clean = cipher_input.replace(" ", "").replace("\n", "").strip()
                cipher_bytes = bytes.fromhex(hex_clean)

                if len(cipher_bytes) % 16 != 0:
                    st.error(f"❌ Panjang Hex tidak valid (harus kelipatan 32 karakter hex).")
                else:
                    key_bytes = list(decrypt_key.encode('utf-8').ljust(16, b'\0')[:16])
                    round_keys = simple_key_expansion(key_bytes, selected_sbox)
                    inv_sbox = INV_SBOXES[selected_sbox_name]

                    plaintext_full = []
                    for i in range(0, len(cipher_bytes), 16):
                        block = list(cipher_bytes[i:i + 16])
                        decrypted_block = aes_decrypt_block(block, 10, round_keys, inv_sbox)
                        plaintext_full.extend(decrypted_block)

                    plaintext_bytes = bytes(plaintext_full)
                    unpadded = unpad_text(plaintext_bytes)
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
                # Convert image to bytes
                img = Image.open(uploaded_img)
                img_bytes = img.tobytes()
                mode = img.mode
                size = img.size

                # Prepare key
                key_bytes = list(img_key.encode('utf-8').ljust(16, b'\0')[:16])
                round_keys = simple_key_expansion(key_bytes, selected_sbox)

                # Pad image bytes
                padding_len = 16 - (len(img_bytes) % 16)
                padded_img = img_bytes + bytes([padding_len] * padding_len)

                # Encrypt
                encrypted_bytes = []
                for i in range(0, len(padded_img), 16):
                    block = list(padded_img[i:i + 16])
                    encrypted_block = aes_encrypt_block(block, 10, round_keys, selected_sbox)
                    encrypted_bytes.extend(encrypted_block)

                # Save metadata + encrypted data
                metadata = f"{mode}|{size[0]}|{size[1]}|".encode('utf-8')
                full_encrypted = metadata + bytes(encrypted_bytes)

                # Create encrypted image visualization (noise)
                encrypted_img = Image.frombytes('RGB', size, bytes(encrypted_bytes[:size[0] * size[1] * 3]), 'raw')

                col1, col2 = st.columns(2)
                with col1:
                    st.image(encrypted_img, caption="Encrypted Image (Visualization)", use_column_width=True)
                with col2:
                    # Download button
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
                    # Read encrypted file
                    encrypted_data = encrypted_file.read()

                    # Extract metadata
                    metadata_end = encrypted_data.index(b'|', encrypted_data.index(b'|', encrypted_data.index(
                        b'|') + 1) + 1) + 1
                    metadata = encrypted_data[:metadata_end].decode('utf-8')
                    mode, width, height = metadata.rstrip('|').split('|')
                    width, height = int(width), int(height)

                    encrypted_bytes = encrypted_data[metadata_end:]

                    # Prepare key
                    key_bytes = list(img_decrypt_key.encode('utf-8').ljust(16, b'\0')[:16])
                    round_keys = simple_key_expansion(key_bytes, selected_sbox)
                    inv_sbox = INV_SBOXES[selected_sbox_name]

                    # Decrypt
                    decrypted_bytes = []
                    for i in range(0, len(encrypted_bytes), 16):
                        block = list(encrypted_bytes[i:i + 16])
                        decrypted_block = aes_decrypt_block(block, 10, round_keys, inv_sbox)
                        decrypted_bytes.extend(decrypted_block)

                    # Unpad
                    decrypted_bytes = bytes(decrypted_bytes)
                    padding_len = decrypted_bytes[-1]
                    img_bytes = decrypted_bytes[:-padding_len]

                    # Reconstruct image
                    decrypted_img = Image.frombytes(mode, (width, height), img_bytes)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(decrypted_img, caption="Original Image", use_column_width=True)
                    with col2:
                        # Convert to downloadable format
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
                st.error("Pastikan key dan S-box yang digunakan sama dengan saat enkripsi!")