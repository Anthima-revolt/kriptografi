import numpy as np
from config import AFFINE_MATRICES, AES_CONSTANT, IRREDUCIBLE_POLY, KAES

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

def generate_remaining_affines(existing_dict, target_total=128):
    matrices = list(existing_dict.values())
    while len(matrices) < target_total:
        M = np.random.randint(0, 2, (8, 8), dtype=np.uint8)
        if not is_invertible_gf2(M): continue
        if np.array_equal(M, KAES): continue
        if any(np.array_equal(M, X) for X in matrices): continue
        matrices.append(M)
    return matrices

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

def byte_to_bits(byte_val):
    return np.array([(byte_val >> i) & 1 for i in range(8)], dtype=int)

def bits_to_byte(bits):
    val = 0
    for i in range(8):
        if bits[i]: val |= (1 << i)
    return val

def create_sbox(matrix, constant, inverse_table):
    sbox = []
    matrix_np = np.array(matrix)
    for x in range(256):
        inv_x = inverse_table[x]
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

# --- Execution to build data structures ---
INVERSE_TABLE = generate_inverse_table()
initial_affines = {name: M for name, M in AFFINE_MATRICES.items()}
affine_128 = generate_remaining_affines(initial_affines, target_total=128)

AFFINE_128_DICT = {}
for name, M in initial_affines.items(): AFFINE_128_DICT[name] = M
idx = 1
for M in affine_128:
    if any(np.array_equal(M, X) for X in initial_affines.values()): continue
    AFFINE_128_DICT[f"G{idx}"] = M
    idx += 1

SBOXES = {}
SBOXES["AES"] = create_sbox(KAES, AES_CONSTANT, INVERSE_TABLE)
for name, M in AFFINE_128_DICT.items():
    SBOXES[name] = create_sbox(M, AES_CONSTANT, INVERSE_TABLE)

INV_SBOXES = {}
for name, sbox in SBOXES.items():
    INV_SBOXES[name] = create_inverse_sbox(sbox)