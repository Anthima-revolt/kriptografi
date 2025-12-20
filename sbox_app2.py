import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter

# --- Re-define Matrix and S-box generation functions and data ---
def mat(*rows):
    return np.array([[int(b) for b in row] for row in rows], dtype=np.uint8)

KAES = mat(
    "10001111",
    "11000111",
    "11100011",
    "11110001",
    "11111000",
    "01111100",
    "00111110",
    "00011111"
)

K1 = mat("00000001","10000000","01000000","00100000","00010000","00001000","00000100","00000001")
K2 = mat("00000010","00000001","10000000","01000000","00100000","00010000","00001000","00000100")
K3 = mat("00000100","00000010","00000001","10000000","01000000","00100000","00010000","00001000")
K4 = mat("00000111","10000011","11000001","11100000","01110000","00111000","00011100","00001111")
K5 = mat("00001000","00000100","00000010","00000001","10000000","01000000","00100000","00010000")
K6 = mat("00001011","10000101","11000010","01100001","10110000","01011000","00101100","00010110")
K7 = mat("00001101","10000110","01000011","10100001","11010000","01101000","00110100","00011010")
K8 = mat("00001110","00000111","10000011","11110001","11111000","01111100","00111110","00011111")
K9 = mat("00010000","00001000","00000100","00000010","00000001","10000000","01000000","00100000")
K10 = mat("00010011","10001001","11000100","01100010","00110001","10011000","01001100","00100110")
K11 = mat("00010101","10001010","01000101","10100010","01010001","10101000","01010100","00101010")
K12 = mat("00011001","10001100","01000110","00100011","10010001","11001000","01100100","00110010")
K13 = mat("10001111","11000111","11100011","11110001","11111000","01111100","00111110","00011111")
K14 = mat("00011010","00001101","10000110","01000011","10100001","11010000","01101000","00110100")
K15 = mat("00011100","00001110","00000111","10000011","11000001","11100000","01110000","00111000")
K16 = mat("00011111","10001111","11000111","11100011","11110001","11111000","01111100","00111110")
K17 = mat("00100000","00010000","00001000","00000100","00000010","00000001","10000000","01000000")
K18 = mat("00100011","10010001","11001000","11100100","01110010","00111001","10011100","01001110")
K19 = mat("00100101","10010010","01001001","10100100","01010010","00101001","10010100","01001010")
K20 = mat("00100110","00010011","10001001","11000100","01100010","00110001","10011000","01001100")
K21 = mat("00101001","10010100","01001010","00100101","10010010","01001001","10100100","01010010")
K22 = mat("00101010","00010101","10001010","01000101","10100010","01010001","10101000","01010100")
K44 = mat("01010111","10101011","11010101","11101010","01110101","10111010","01011101","10101110")
K81 = mat("10100001","11010000","01101000","00110100","00011010","00001101","10000110","01000011")
K111 = mat("11011100","01101110","00110111","10011011","11001101","11100110","01110011","10111001")
K127 = mat("11111101","11111110","01111111","10111111","11011111","11101111","11110111","11111011")
K128 = mat("11111110","01111111","10111111","11011111","11101111","11110111","11111011","11111101")

AFFINE_MATRICES = {
    "AES": KAES,
    "K1": K1, "K2": K2, "K3": K3,
    "K4": K4, "K5": K5, "K6": K6, "K7": K7, "K8": K8,
    "K9": K9, "K10": K10, "K11": K11, "K12": K12,
    "K13": K13, "K14": K14, "K15": K15, "K16": K16,
    "K17": K17, "K18": K18, "K19": K19, "K20": K20,
    "K21": K21, "K22": K22,
    "K44": K44, "K81": K81, "K111": K111,
    "K127": K127, "K128": K128
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

initial_affines = {
    name: M for name, M in AFFINE_MATRICES.items()
    # if name != "AES"
}

def generate_remaining_affines(existing_dict, target_total=128):
    matrices = list(existing_dict.values())

    while len(matrices) < target_total:
        M = np.random.randint(0, 2, (8, 8), dtype=np.uint8)

        if not is_invertible_gf2(M):
            continue

        if np.array_equal(M, KAES):
            continue

        if any(np.array_equal(M, X) for X in matrices):
            continue

        matrices.append(M)

    return matrices

affine_128 = generate_remaining_affines(initial_affines, target_total=128)

AFFINE_128_DICT = {}

for name, M in initial_affines.items():
    AFFINE_128_DICT[name] = M

idx = 1
for M in affine_128:
    if any(np.array_equal(M, X) for X in initial_affines.values()):
        continue
    AFFINE_128_DICT[f"G{idx}"] = M
    idx += 1

AES_CONSTANT = 0x63
IRREDUCIBLE_POLY = 0x11B

def gf_multiply(a, b):
    p = 0
    for i in range(8):
        if b & 1:
          p ^= a
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

SBOXES = {}
SBOXES["AES"] = create_sbox(KAES, AES_CONSTANT)

for name, M in AFFINE_128_DICT.items():
    SBOXES[name] = create_sbox(M, AES_CONSTANT)

# --- Cryptographic property calculation functions ---
def walsh_hadamard_transform(func):
    n = 8
    wf = [(-1)**func[x] for x in range(256)]
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
        nl = (2**8 - max_abs_wh) // 2
        if nl < min_nl:
            min_nl = nl
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
            nl = (2**8 - max_abs_wh) // 2
            if nl < min_bic_nl:
                min_bic_nl = nl
    return int(min_bic_nl)
def calculate_bic_sac(sbox):
    total_sac = 0
    count = 0
    for i in range(8):
        for j in range(i+1, 8):
            for inp in range(256):
                val_orig = ((sbox[inp] >> i) & 1) ^ ((sbox[inp] >> j) & 1)
                for bit in range(8):
                    flipped = inp ^ (1 << bit)
                    val_flip = ((sbox[flipped] >> i) & 1) ^ ((sbox[flipped] >> j) & 1)
                    if val_orig != val_flip:
                        total_sac += 1
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
        if current_max > max_spectrum:
            max_spectrum = current_max
    return max_spectrum / 256.0
def calculate_dap(sbox):
    max_diff = 0
    for dx in range(1, 256):
        counts = Counter()
        for x in range(256):
            dy = sbox[x] ^ sbox[x ^ dx]
            counts[dy] += 1
        current_max = max(counts.values())
        if current_max > max_diff:
            max_diff = current_max
    return max_diff / 256.0
def calculate_du(sbox):
    max_diff = 0
    for dx in range(1, 256):
        counts = {}
        for x in range(256):
            dy = sbox[x] ^ sbox[x ^ dx]
            counts[dy] = counts.get(dy, 0) + 1
        current_max = max(counts.values())
        if current_max > max_diff:
            max_diff = current_max
    return int(max_diff)
def calculate_ad(sbox):
    max_deg = 0
    for i in range(8):
        func = [(sbox[x] >> i) & 1 for x in range(256)]
        anf = list(func)
        for step in range(1, 256):
             if step & (step-1) == 0:
                 for j in range(0, 256, step*2):
                     for k in range(j, j+step):
                         anf[k+step] ^= anf[k]

        deg = 0
        for x in range(256):
            if anf[x] == 1:
                hw = bin(x).count('1')
                if hw > deg: deg = hw
        if deg > max_deg: max_deg = deg
    return int(max_deg)


#Transparency Order
def calculate_to(sbox):
    """
    Transparency Order (TO): Mengukur ketahanan terhadap Side-Channel Attacks (DPA).
    Berdasarkan definisi Prouff. Semakin RENDAH semakin baik.
    """
    N = 256
    n = 8
    
    # Menghitung autokorelasi untuk setiap bit output
    max_to = 0
    # Kita ambil rata-rata kebocoran dari komponen fungsi boolean bitwise
    all_autocorr = []
    
    for a in range(1, N):
        current_sum = 0
        for v in range(N):
            # HW = Hamming Weight dari v
            hw_v = bin(v).count('1')
            
            # Menghitung korelasi derivatif
            # D_a f(x) = f(x) ^ f(x ^ a)
            correlation = 0
            for x in range(N):
                # Memeriksa parity dari S(x) ^ S(x ^ a) terhadap v
                if bin((sbox[x] ^ sbox[x ^ a]) & v).count('1') % 2 == 0:
                    correlation += 1
                else:
                    correlation -= 1
            
            current_sum += abs(correlation)
        all_autocorr.append(current_sum)
    
    # Normalisasi skala (umumnya dalam range 0.1 - 0.9 untuk S-Box 8-bit)
    to_score = max(all_autocorr) / (N**2)
    return round(to_score, 6)

# CI
def calculate_ci(sbox):
    """
    Correlation Immunity (CI): Mengukur apakah bit output independen dari bit input.
    Menggunakan Walsh-Hadamard Transform.
    Nilai yang dikembalikan adalah 'Orde' (t). Semakin TINGGI semakin baik.
    """
    N = 256
    n = 8
    
    # Mencari nilai t tertinggi
    # S-Box memiliki CI orde t jika WHT(f) = 0 untuk 1 <= HW(mask) <= t
    max_t = 0
    
    # Cek untuk setiap bit fungsi (8 fungsi boolean)
    possible_t = []
    
    for i in range(8):
        # Ambil fungsi bit ke-i
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
        
    # CI S-Box adalah nilai minimum dari CI semua fungsi komponennya
    return int(min(possible_t))

# --- Calculate S-box metrics ---
sbox_metrics = {}

# for name, sbox in SBOXES.items():
#     metrics = {
#         "NL": calculate_nl(sbox),
#         "SAC": calculate_sac(sbox),
#         "BIC_NL": calculate_bic_nl(sbox),
#         "BIC_SAC": calculate_bic_sac(sbox),
#         "LAP": calculate_lap(sbox),
#         "DAP": calculate_dap(sbox),
#         "DU": calculate_du(sbox),
#         "AD": calculate_ad(sbox),
#         "TO" : calculate_to(sbox),
#         "CI" : calculate_ci(sbox)
#     }
#     sbox_metrics[name] = metrics

@st.cache_data(show_spinner=True)
def get_single_sbox_metrics(sbox):
    """Fungsi ini hanya menghitung metrik untuk S-Box yang dipilih user"""
    return {
        "NL": calculate_nl(sbox),
        "SAC": calculate_sac(sbox),
        "BIC_NL": calculate_bic_nl(sbox),
        "BIC_SAC": calculate_bic_sac(sbox),
        "LAP": calculate_lap(sbox),
        "DAP": calculate_dap(sbox),
        "DU": calculate_du(sbox),
        "AD": calculate_ad(sbox),
        "TO" : calculate_to(sbox),
        "CI" : calculate_ci(sbox)
    }

# sbox_df = pd.DataFrame.from_dict(sbox_metrics, orient='index')

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
        row = sbox[i*16:(i+1)*16]
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
# Tampilkan hasil dalam DataFrame
metrics_df = pd.DataFrame([current_metrics])
st.dataframe(metrics_df, use_container_width=True)
# if selected_sbox_name:
#     properties = sbox_df.loc[selected_sbox_name]
#     st.dataframe(properties.to_frame().T)



#AES Encryption
# --- AES Implementation Functions ---

def sub_bytes(state, sbox):
    for i in range(len(state)):
        state[i] = sbox[state[i]]

def shift_rows(state):
    # state diwakili sebagai list 16 byte (kolom per kolom)
    # [0 4 8  12]
    # [1 5 9  13]
    # [2 6 10 14]
    # [3 7 11 15]
    res = [0] * 16
    res[0], res[4], res[8], res[12] = state[0], state[4], state[8], state[12]
    res[5], res[9], res[13], res[1] = state[1], state[5], state[9], state[13]
    res[10], res[14], res[2], res[6] = state[2], state[6], state[10], state[14]
    res[15], res[3], res[7], res[11] = state[3], state[7], state[11], state[15]
    return res

def mix_columns(state):
    def gmul(a, b): # Perkalian Galois Field sederhana
        p = 0
        for _ in range(8):
            if b & 1: p ^= a
            hi_bit_set = a & 0x80
            a <<= 1
            if hi_bit_set: a ^= 0x1B
            b >>= 1
        return p & 0xFF

    new_state = [0] * 16
    for i in range(4): # per kolom
        col = state[i*4 : i*4+4]
        new_state[i*4]     = gmul(0x02, col[0]) ^ gmul(0x03, col[1]) ^ col[2] ^ col[3]
        new_state[i*4 + 1] = col[0] ^ gmul(0x02, col[1]) ^ gmul(0x03, col[2]) ^ col[3]
        new_state[i*4 + 2] = col[0] ^ col[1] ^ gmul(0x02, col[2]) ^ gmul(0x03, col[3])
        new_state[i*4 + 3] = gmul(0x03, col[0]) ^ col[1] ^ col[2] ^ gmul(0x02, col[3])
    return new_state

def aes_encrypt_block(plaintext_block, rounds, round_keys, sbox):
    # plaintext_block: list of 16 integers
    state = [plaintext_block[i] ^ round_keys[0][i] for i in range(16)]
    
    for r in range(1, rounds):
        sub_bytes(state, sbox)
        state = shift_rows(state)
        state = mix_columns(state)
        rk = round_keys[r]
        state = [state[i] ^ rk[i] for i in range(16)]
        
    # Round terakhir (tanpa MixColumns)
    sub_bytes(state, sbox)
    state = shift_rows(state)
    rk = round_keys[rounds]
    state = [state[i] ^ rk[i] for i in range(16)]
    return state

def simple_key_expansion(key_16_bytes, sbox):
    # Versi sangat sederhana untuk simulasi (hanya mengulang kunci)
    # Di dunia nyata, ini menggunakan Rcon dan S-Box
    keys = []
    for i in range(11):
        keys.append([(b + i) % 256 for b in key_16_bytes])
    return keys

def pad_text(text):
    # Menggunakan PKCS7 padding sederhana
    padding_len = 16 - (len(text) % 16)
    padding = bytes([padding_len] * padding_len)
    return text.encode('utf-8') + padding

def encrypt_full_text(plaintext, round_keys, sbox):
    # 1. Padding teks
    padded_data = pad_text(plaintext)
    
    # 2. Pecah menjadi blok-blok 16 byte
    ciphertext_full = []
    for i in range(0, len(padded_data), 16):
        block = list(padded_data[i:i+16])
        # Enkripsi per blok menggunakan fungsi aes_encrypt_block yang sudah dibuat
        encrypted_block = aes_encrypt_block(block, 10, round_keys, sbox)
        ciphertext_full.extend(encrypted_block)
    
    return ciphertext_full

# --- Section 6: Try Encryption ---
st.divider()
st.header(f"AES Text Encryption Simulation")
st.write(f"Menggunakan S-Box: **{selected_sbox_name}**")

col1, col2 = st.columns(2)
with col1:
    user_text = st.text_area("Input Plaintext (Teks Biasa):", "Halo, ini adalah pesan rahasia yang panjangnya lebih dari enam belas karakter.")
with col2:
    user_key = st.text_input("Encryption Key (Max 16 chars):", "kuncirahasia1234")

if st.button("Encrypt Now"):
    if not user_key:
        st.error("Key tidak boleh kosong!")
    else:
        # Menyiapkan kunci 16 byte
        key_bytes = list(user_key.encode('utf-8').ljust(16, b'\0')[:16])
        round_keys = simple_key_expansion(key_bytes, selected_sbox)
        
        # Proses Enkripsi
        with st.spinner("Memproses enkripsi blok demi blok..."):
            cipher_output = encrypt_full_text(user_text, round_keys, selected_sbox)
        
        # Menampilkan hasil
        st.subheader("Hasil Enkripsi")
        
        # Tampilan Hex
        hex_output = "".join(f"{b:02x}" for b in cipher_output)
        st.write("**Ciphertext (Hexadecimal):**")
        st.code(hex_output)
        
        # Tampilan Base64 (lebih umum untuk transmisi data)
        import base64
        b64_output = base64.b64encode(bytes(cipher_output)).decode('utf-8')
        st.write("**Ciphertext (Base64):**")
        st.code(b64_output)

        st.info(f"Teks asli panjangnya {len(user_text)} karakter, setelah padding menjadi {len(cipher_output)} byte ({len(cipher_output)//16} blok AES).")