"""
BIL216 - İşaretler ve Sistemler — Ödev 2
DTMF Türk Alfabesi Uygulaması
Flask backend + DSP (NumPy/SciPy)
Premium Siyah & Parlak Turuncu Tema (30 Karakter Optimize Edilmiş)
"""

from flask import Flask, request, jsonify, send_file, render_template_string
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import get_window
import io, base64, os

app = Flask(__name__)

# ─── DSP SABITLERI ───────────────────────────────────────────
SAMPLE_RATE   = 44100
CHAR_DURATION = 0.040   # 40 ms
GAP_DURATION  = 0.010   # 10 ms
THRESHOLD     = 0.01
WINDOW_TYPE   = 'hann'

# Tam 30 kombinasyon için optimize edilmiş 5x6 frekans matrisi
LOW_FREQS  = [697, 770, 852, 941, 1209]
HIGH_FREQS = [1336, 1477, 1633, 1776, 1933, 2089]
CHARS = list("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ ") # 29 Harf + 1 Boşluk

# Frekans haritaları
FREQ_MAP = {}
REVERSE_MAP = {}

def _build_maps():
    used = set()
    for idx, ch in enumerate(CHARS):
        li = idx % len(LOW_FREQS)
        hi = (idx + idx // len(HIGH_FREQS)) % len(HIGH_FREQS)
        fl, fh = LOW_FREQS[li], HIGH_FREQS[hi]
        attempts = 0
        while (fl, fh) in used and attempts < 60:
            hi = (hi + 1) % len(HIGH_FREQS)
            fh = HIGH_FREQS[hi]
            attempts += 1
        used.add((fl, fh))
        FREQ_MAP[ch] = (fl, fh)
        REVERSE_MAP[(fl, fh)] = ch

_build_maps()

# ─── DSP FONKSİYONLARI ───────────────────────────────────────
def synthesize_tone(fl, fh, duration=CHAR_DURATION, sr=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    s = np.sin(2 * np.pi * fl * t) + np.sin(2 * np.pi * fh * t)
    s /= np.max(np.abs(s) + 1e-9)
    return s.astype(np.float32)

def text_to_signal(text):
    text = text.upper()
    segments = []
    gap = np.zeros(int(SAMPLE_RATE * GAP_DURATION), dtype=np.float32)
    for ch in text:
        if ch not in FREQ_MAP:
            ch = ' '
        fl, fh = FREQ_MAP[ch]
        segments.extend([synthesize_tone(fl, fh), gap])
    return np.concatenate(segments) if segments else np.zeros(1, dtype=np.float32)

def goertzel(samples, target_freq, sr=SAMPLE_RATE):
    N = len(samples)
    k = round(N * target_freq / sr)
    omega = 2 * np.pi * k / N
    coeff = 2 * np.cos(omega)
    s1 = s2 = 0.0
    for x in samples:
        s0 = x + coeff * s1 - s2
        s2, s1 = s1, s0
    return s2**2 + s1**2 - coeff * s1 * s2

def detect_char(segment, sr=SAMPLE_RATE):
    win = get_window(WINDOW_TYPE, len(segment))
    w = segment * win
    rms = np.sqrt(np.mean(w**2))
    if rms < THRESHOLD:
        return None, {}
    all_freqs = list(set(LOW_FREQS + HIGH_FREQS))
    energy = {f: float(goertzel(w, f, sr)) for f in all_freqs}
    best_ch, best_score = None, -1
    for ch, (fl, fh) in FREQ_MAP.items():
        score = energy[fl] + energy[fh]
        if score > best_score:
            best_score, best_ch = score, ch
    return best_ch, energy

def decode_signal(data, sr=SAMPLE_RATE):
    char_n = int(sr * CHAR_DURATION)
    step_n = int(sr * (CHAR_DURATION + GAP_DURATION))
    result, last_ch = [], None
    energy_log = []
    pos = 0
    while pos + char_n <= len(data):
        seg = data[pos:pos + char_n]
        ch, energy = detect_char(seg, sr)
        if ch and ch != last_ch:
            result.append(ch)
            last_ch = ch
            energy_log.append({'char': ch, 'energy': energy,
                                'fl': FREQ_MAP[ch][0], 'fh': FREQ_MAP[ch][1]})
        elif ch is None:
            last_ch = None
        pos += step_n
    return ''.join(result), energy_log

def signal_to_wav_bytes(signal):
    buf = io.BytesIO()
    wav.write(buf, SAMPLE_RATE, (signal * 32767).astype(np.int16))
    return buf.getvalue()

# Siyah & Parlak Turuncu Temalı Matplotlib Fonksiyonları
BG_COLOR = '#141414'      # Card Background
AX_BG_COLOR = '#1f1f1f'   # Plot Background
TEXT_COLOR = '#e5e5e5'    # Light Text
TEXT_DARK = '#888888'     # Dim Text
GRID_COLOR = '#333333'    # Grid/Borders
PLOT_COLOR = '#ff5e00'    # Bright Orange
PLOT_COLOR_LIGHT = '#ff8800' # Lighter Orange

def waveform_png(signal, width=800, height=100):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width/100, height/100), facecolor=BG_COLOR)
    ax.set_facecolor(AX_BG_COLOR)
    t = np.linspace(0, len(signal)/SAMPLE_RATE, len(signal))
    step = max(1, len(signal)//2000)
    ax.plot(t[::step], signal[::step], color=PLOT_COLOR_LIGHT, linewidth=1, alpha=0.9)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-1.1, 1.1)
    ax.tick_params(colors=TEXT_DARK, labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.set_xlabel('Zaman (s)', color=TEXT_DARK, fontsize=9)
    ax.set_ylabel('Genlik', color=TEXT_DARK, fontsize=9)
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def spectrum_png(signal, width=800, height=160):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(width/100, height/100), facecolor=BG_COLOR)
    ax.set_facecolor(AX_BG_COLOR)
    N = min(len(signal), 8192)
    chunk = signal[:N] * get_window('hann', N)
    fft_mag = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(N, 1/SAMPLE_RATE)
    mask = freqs < 4000
    ax.fill_between(freqs[mask], fft_mag[mask], alpha=0.3, color=PLOT_COLOR)
    ax.plot(freqs[mask], fft_mag[mask], color=PLOT_COLOR_LIGHT, linewidth=1.2)
    all_f = list(set(LOW_FREQS + HIGH_FREQS))
    for f in all_f:
        ax.axvline(f, color='#00e5ff', alpha=0.4, linewidth=1, linestyle='--') 
    ax.set_xlim(0, 4000)
    ax.tick_params(colors=TEXT_DARK, labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.set_xlabel('Frekans (Hz)', color=TEXT_DARK, fontsize=9)
    ax.set_ylabel('Genlik', color=TEXT_DARK, fontsize=9)
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def goertzel_png(energy_log, width=800, height=180):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not energy_log:
        return ''
    last = energy_log[-1]
    all_freqs = sorted(set(LOW_FREQS + HIGH_FREQS))
    energies = [last['energy'].get(f, 0) for f in all_freqs]
    max_e = max(energies) or 1
    energies_norm = [e/max_e for e in energies]
    colors = [PLOT_COLOR_LIGHT if f in (last['fl'], last['fh']) else '#333333' for f in all_freqs]

    fig, ax = plt.subplots(figsize=(width/100, height/100), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    bars = ax.bar(range(len(all_freqs)), energies_norm, color=colors, edgecolor='none', width=0.6)
    ax.set_xticks(range(len(all_freqs)))
    ax.set_xticklabels([str(f) for f in all_freqs], rotation=45, ha='right', fontsize=8, color=TEXT_DARK)
    ax.tick_params(axis='y', colors=TEXT_DARK, labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.set_ylabel('Normalize Enerji', color=TEXT_DARK, fontsize=9)
    ax.set_title(f"Goertzel Analizi — Tespit Edilen: '{last['char']}' ({last['fl']}Hz + {last['fh']}Hz)",
                 color=PLOT_COLOR_LIGHT, fontsize=10, pad=10, fontweight='bold')
    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ─── FLASK ROUTES ─────────────────────────────────────────────
@app.route('/')
def index():
    freq_table = [{'char': ch if ch != ' ' else 'BOŞLUK',
                   'fl': FREQ_MAP[ch][0], 'fh': FREQ_MAP[ch][1]}
                  for ch in CHARS]
    return render_template_string(HTML_TEMPLATE, freq_table=freq_table)

@app.route('/encode', methods=['POST'])
def encode():
    data = request.json
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'Metin boş'}), 400
    signal = text_to_signal(text)
    wav_b64 = base64.b64encode(signal_to_wav_bytes(signal)).decode()
    wave_img = waveform_png(signal)
    spec_img = spectrum_png(signal)
    chars = [{'ch': c if c != ' ' else '⎵',
              'fl': FREQ_MAP.get(c.upper(), FREQ_MAP[' '])[0],
              'fh': FREQ_MAP.get(c.upper(), FREQ_MAP[' '])[1]}
             for c in text.upper()]
    return jsonify({
        'wav_b64': wav_b64,
        'wave_img': wave_img,
        'spec_img': spec_img,
        'chars': chars,
        'duration_ms': round(len(signal) / SAMPLE_RATE * 1000),
        'num_chars': len(text)
    })

@app.route('/decode', methods=['POST'])
def decode():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yok'}), 400
    f = request.files['file']
    buf = io.BytesIO(f.read())
    sr, data = wav.read(buf)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    text, energy_log = decode_signal(data, sr)
    goertzel_img = goertzel_png(energy_log)
    return jsonify({
        'text': text,
        'energy_log': energy_log,
        'goertzel_img': goertzel_img,
        'num_detected': len(text)
    })

@app.route('/freq_table')
def freq_table_api():
    return jsonify([{'char': ch, 'fl': FREQ_MAP[ch][0], 'fh': FREQ_MAP[ch][1]} for ch in CHARS])


# ─── HTML TEMPLATe ────────────────────────────────────────────
_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_dir, "index.html"), "r", encoding="utf-8") as f:
    HTML_TEMPLATE = f.read()

application = app

if __name__ == '__main__':
    print("\n" + "="*63)
    print("  BIL216 DTMF Uygulaması — Siyah/Turuncu Flask Sunucusu")
    print("="*63)
    print("  Tarayıcıda açın: http://127.0.0.1:5000")
    print("="*63 + "\n")
    app.run(debug=True, port=5000)