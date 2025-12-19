
import streamlit as st
import pandas as pd
import json

# --- Re-define all necessary functions and data for the Streamlit app ---
# This is a condensed version of the code that generated sbox_metrics
# to make the Streamlit app self-contained. In a real scenario, this might
# be refactored into a separate module or loaded from a pre-calculated file.

import numpy as np
import collections

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
    "AES": KAES, "K1": K1, "K2": K2, "K3": K3, "K4": K4, "K5": K5, "K6": K6, "K7": K7, "K8": K8,
    "K9": K9, "K10": K10, "K11": K11, "K12": K12, "K13": K13, "K14": K14, "K15": K15, "K16": K16,
    "K17": K17, "K18": K18, "K19": K19, "K20": K20, "K21": K21, "K22": K22,
    "K44": K44, "K81": K81, "K111": K111, "K127": K127, "K128": K128
}

def is_invertible_gf2(M):
    M_copy = M.copy()
    n = 8
    rank = 0

    for col in range(n):
        pivot = None
        for row in range(rank, n):
            if M_copy[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            continue

        M_copy[[rank, pivot]] = M_copy[[pivot, rank]]
        for r in range(n):
            if r != rank and M_copy[r, col] == 1:
                M_copy[r] ^= M_copy[rank]
        rank += 1

    return rank == n

initial_affines = {name: M for name, M in AFFINE_MATRICES.items() if name != "AES"}

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

def is_bijective_sbox(sbox):
    return len(set(sbox)) == 256

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
    total_flipped_bits = 0
    num_tests = 0

    for x in range(256):
        original_out = sbox[x]
        for bit_pos in range(8):
            flipped_input = x ^ (1 << bit_pos)
            flipped_out = sbox[flipped_input]
            diff = original_out ^ flipped_out
            total_flipped_bits += bin(diff).count('1')
            num_tests += 1

    if num_tests == 0:
        return 0

    average_flipped_bits = total_flipped_bits / num_tests
    return average_flipped_bits

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
    bic_sac_scores = []

    for i in range(8):
        for j in range(i + 1, 8):
            combined_sbox_output = [( ((sbox[x] >> i) & 1) ^ ((sbox[x] >> j) & 1) ) for x in range(256)]

            total_flipped_bits_combined = 0
            num_tests_combined = 0

            for x_input in range(256):
                original_combined_out = combined_sbox_output[x_input]
                for bit_pos in range(8):
                    flipped_input_combined = x_input ^ (1 << bit_pos)
                    flipped_combined_out = combined_sbox_output[flipped_input_combined]

                    if original_combined_out != flipped_combined_out:
                        total_flipped_bits_combined += 1
                    num_tests_combined += 1

            if num_tests_combined > 0:
                bic_sac_scores.append(total_flipped_bits_combined / num_tests_combined)

    if not bic_sac_scores:
        return 0.0

    return sum(bic_sac_scores) / len(bic_sac_scores)

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
    max_diff_count = 0
    for dx in range(1, 256):
        counts = collections.Counter()
        for x in range(256):
            dy = sbox[x] ^ sbox[x ^ dx]
            counts[dy] += 1

        if counts:
            current_max = max(counts.values())
            if current_max > max_diff_count:
                max_diff_count = current_max

    return max_diff_count / 256.0

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

# --- Calculate S-box Metrics (as in the previous step) ---
sbox_metrics = {}
for name, sbox in SBOXES.items():
    if not is_bijective_sbox(sbox):
        continue # Skip non-bijective S-boxes

    sbox_metrics[name] = {
        "Non-Linearity": calculate_nl(sbox),
        "SAC": calculate_sac(sbox),
        "BIC-NL": calculate_bic_nl(sbox),
        "BIC-SAC": calculate_bic_sac(sbox),
        "LAP": calculate_lap(sbox),
        "DAP": calculate_dap(sbox),
        "DU": calculate_du(sbox),
        "AD": calculate_ad(sbox)
    }

# --- Streamlit Application Layout ---
st.set_page_config(layout="wide", page_title="S-box Cryptographic Properties")

st.title("S-box Cryptographic Properties Analysis")
st.write("This application displays the cryptographic properties of generated S-boxes. Only bijective S-boxes are included in the analysis.")

# Convert sbox_metrics to DataFrame for better display
df_metrics = pd.DataFrame.from_dict(sbox_metrics, orient='index')
df_metrics.index.name = 'S-box Name'

st.header("Cryptographic Properties Table")
st.dataframe(df_metrics)

st.header("How to Run This Application Locally")
st.markdown(
    """
    To run this Streamlit application on your local machine, follow these steps:

    1.  **Save the code**: Save the Python code above into a file named `sbox_app.py`.
    2.  **Install Streamlit**: If you haven't already, install Streamlit using pip:
        ```bash
        pip install streamlit pandas numpy
        ```
    3.  **Run the app**: Open your terminal or command prompt, navigate to the directory where you saved `sbox_app.py`, and run:
        ```bash
        streamlit run sbox_app.py
        ```

    Your web browser will automatically open a new tab displaying the application.
    """
)
