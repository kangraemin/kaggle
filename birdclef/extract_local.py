"""로컬 soundscape Perch 임베딩 추출
Python 3.11 + TF 2.21 + Perch v2_cpu
"""
import os, re, gc, time
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
from pathlib import Path
from tqdm.auto import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC  # 160000
FILE_SAMPLES = 60 * SR  # 1920000
N_WINDOWS = 12
BATCH_FILES = 8

DATA = Path('data')
MODEL_DIR = DATA / 'perch_model'
SS_DIR = DATA / 'train_soundscapes'
OUT_DIR = DATA / 'soundscape_extracted'

FNAME_RE = re.compile(r'BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg')

def parse_soundscape_filename(name):
    m = FNAME_RE.match(name)
    if not m:
        return {'site': None, 'hour_utc': -1}
    _, site, _, hms = m.groups()
    return {'site': site, 'hour_utc': int(hms[:2])}

def read_soundscape_60s(path):
    y, sr = sf.read(path, dtype='float32', always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    return y[:FILE_SAMPLES]

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print('Loading Perch model...')
    t0 = time.time()
    model = tf.saved_model.load(str(MODEL_DIR))
    infer_fn = model.signatures['serving_default']
    print(f'Perch loaded in {time.time()-t0:.0f}s')

    # Load taxonomy for species mapping
    taxonomy = pd.read_csv(DATA / 'taxonomy.csv')
    sample_sub = pd.read_csv(DATA / 'sample_submission.csv')
    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    N_CLASSES = len(PRIMARY_LABELS)

    bc_labels = pd.read_csv(MODEL_DIR / 'assets' / 'labels.csv').reset_index().rename(
        columns={'index': 'bc_index', 'inat2024_fsd50k': 'scientific_name'})
    NO_LABEL_INDEX = len(bc_labels)

    taxonomy_ = taxonomy.copy()
    taxonomy_['scientific_name'] = taxonomy_['scientific_name'].astype(str)
    mapping = taxonomy_.merge(bc_labels[['scientific_name', 'bc_index']], on='scientific_name', how='left')
    mapping['bc_index'] = mapping['bc_index'].fillna(NO_LABEL_INDEX).astype(int)
    label_to_bc = mapping.set_index('primary_label')['bc_index']
    BC_INDICES = np.array([int(label_to_bc.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)
    MAPPED_MASK = BC_INDICES != NO_LABEL_INDEX
    MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
    MAPPED_BC_INDICES = BC_INDICES[MAPPED_MASK].astype(np.int32)

    print(f'Species: {N_CLASSES}, Mapped: {MAPPED_MASK.sum()}')

    # List all soundscape files
    all_files = sorted(SS_DIR.glob('*.ogg'))
    print(f'Total soundscape files: {len(all_files)}')

    n_files = len(all_files)
    n_rows = n_files * N_WINDOWS

    row_ids = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites = np.empty(n_rows, dtype=object)
    hours = np.empty(n_rows, dtype=np.int16)
    scores = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embeddings = np.zeros((n_rows, 1536), dtype=np.float32)

    write_row = 0
    t_start = time.time()

    for start in tqdm(range(0, n_files, BATCH_FILES), desc='Extracting'):
        batch = all_files[start:start + BATCH_FILES]
        bn = len(batch)
        x = np.empty((bn * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
        bstart = write_row

        for bi, path in enumerate(batch):
            audio = read_soundscape_60s(path)
            x[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = audio.reshape(N_WINDOWS, WINDOW_SAMPLES)
            meta = parse_soundscape_filename(path.name)
            row_ids[write_row:write_row + N_WINDOWS] = [f'{path.stem}_{t}' for t in range(5, 65, 5)]
            filenames[write_row:write_row + N_WINDOWS] = path.name
            sites[write_row:write_row + N_WINDOWS] = meta['site']
            hours[write_row:write_row + N_WINDOWS] = meta['hour_utc']
            write_row += N_WINDOWS

        out = infer_fn(inputs=tf.convert_to_tensor(x))
        logits = out['label'].numpy().astype(np.float32)
        emb = out['embedding'].numpy().astype(np.float32)

        scores[bstart:write_row, MAPPED_POS] = logits[:write_row - bstart, MAPPED_BC_INDICES]
        embeddings[bstart:write_row] = emb

        del x, out, logits, emb
        gc.collect()

        if (start // BATCH_FILES + 1) % 50 == 0:
            elapsed = time.time() - t_start
            done = start + BATCH_FILES
            eta = elapsed / done * (n_files - done)
            print(f'  {done}/{n_files} files ({elapsed:.0f}s, ETA {eta:.0f}s)')

    total_time = time.time() - t_start

    # Save
    meta_df = pd.DataFrame({
        'row_id': row_ids[:write_row],
        'filename': filenames[:write_row],
        'site': sites[:write_row],
        'hour_utc': hours[:write_row],
    })
    meta_df.to_parquet(OUT_DIR / 'all_ss_meta.parquet', index=False)
    np.save(OUT_DIR / 'all_ss_scores.npy', scores[:write_row])
    np.save(OUT_DIR / 'all_ss_embeddings.npy', embeddings[:write_row])

    print(f'\n=== 완료 ===')
    print(f'Files: {n_files}')
    print(f'Windows: {write_row}')
    print(f'Scores: {scores[:write_row].shape}')
    print(f'Embeddings: {embeddings[:write_row].shape}')
    print(f'Time: {total_time:.0f}s ({total_time/60:.1f}min)')
    print(f'Saved to: {OUT_DIR}')

if __name__ == '__main__':
    main()
