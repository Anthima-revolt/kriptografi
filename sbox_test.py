from collections import Counter

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