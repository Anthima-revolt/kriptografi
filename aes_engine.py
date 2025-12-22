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