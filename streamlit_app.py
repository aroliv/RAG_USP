# app.py
import re
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageFilter# app.py
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
from unidecode import unidecode
from rapidfuzz import process, fuzz

# ========= OCR backend (Tesseract) =========
EASYOCR_OK = False   # manter False para evitar PyTorch no Streamlit Cloud
PYTESS_OK = False
try:
    import pytesseract
    PYTESS_OK = True
except Exception:
    PYTESS_OK = False

# ========= Constantes do seu XLSX =========
COL_NOME = "Nome"
COL_USP  = "NUSP"

# ========= Helpers gerais =========
def normalize_name(s: str) -> str:
    s = unidecode(str(s or ""))
    s = re.sub(r"[^a-zA-Z\s]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s

@st.cache_data
def load_default_inscritos():
    """Carrega inscritos.xlsx do repo (colunas: Nome, NUSP)."""
    df = pd.read_excel("inscritos.xlsx")
    return prepare_inscritos_df(df)

def prepare_inscritos_df(df: pd.DataFrame) -> pd.DataFrame:
    """Garante colunas Nome/NUSP e campos normalizados."""
    # tenta mapear títulos alternativos:
    title_map = {}
    for c in df.columns:
        k = normalize_name(c)
        if k in {"nome", "nome completo", "aluno", "participante"} and COL_NOME not in df.columns:
            title_map[c] = COL_NOME
        if k in {"usp", "nusp", "numero usp", "n usp", "matricula", "ra"} and COL_USP not in df.columns:
            title_map[c] = COL_USP
    df = df.rename(columns=title_map).copy()

    if COL_NOME not in df.columns or COL_USP not in df.columns:
        st.error(f"Planilha precisa ter colunas '{COL_NOME}' e '{COL_USP}'.")
        st.stop()

    df[COL_NOME] = df[COL_NOME].astype(str).str.strip()
    df[COL_USP]  = df[COL_USP].astype(str).str.replace(r"\D","", regex=True).str.strip()
    df["__nome_norm"] = df[COL_NOME].map(normalize_name)
    df["__usp_norm"]  = df[COL_USP]
    return df

def read_xlsx_from_upload_or_drive(uploaded_file, drive_link_or_id: str) -> pd.DataFrame | None:
    """Retorna DF se usuário mandou arquivo/link; senão None (usamos default)."""
    if uploaded_file is not None:
        return prepare_inscritos_df(pd.read_excel(uploaded_file))

    if drive_link_or_id:
        try:
            import gdown
        except ImportError:
            st.error("gdown não instalado. Adicione 'gdown' no requirements.txt.")
            st.stop()

        file_id = None
        m = re.search(r"/d/([A-Za-z0-9_-]+)", drive_link_or_id)
        if m:
            file_id = m.group(1)
        elif re.fullmatch(r"[A-Za-z0-9_-]{20,}", drive_link_or_id):
            file_id = drive_link_or_id

        if not file_id:
            st.error("Cole um link/ID do Drive válido (Compartilhar → Qualquer pessoa com o link).")
            st.stop()

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "inscritos.xlsx"
            import gdown
            gdown.download(id=file_id, output=str(out), quiet=True)
            return prepare_inscritos_df(pd.read_excel(out))

    return None

def best_match(name_norm: str, candidates_norm: list[str]):
    if not candidates_norm:
        return None, 0, None
    match, score, idx = process.extractOne(
        name_norm, candidates_norm, scorer=fuzz.token_set_ratio
    )
    return match, score, idx

def build_download(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv", use_container_width=True)

# ========= OCR: utilitários focados em pt-BR =========
# correções comuns em NUSP manuscrito
TRANS_DIGITS = str.maketrans({"O":"0","o":"0","Q":"0","S":"5","s":"5","B":"8","b":"8","I":"1","l":"1"})
def clean_usp_token(t: str) -> str:
    t = (t or "").translate(TRANS_DIGITS)
    return re.sub(r"\D","", t)

def is_valid_usp(t: str) -> bool:
    return t.isdigit() and 6 <= len(t) <= 9

def preprocess_for_ocr(img: "Image.Image") -> "Image.Image":
    # escala + contraste + nitidez + binarização leve
    up = img.convert("L").resize((int(img.width*1.6), int(img.height*1.6)), resample=Image.BICUBIC)
    up = ImageOps.autocontrast(up)
    up = ImageEnhance.Sharpness(up).enhance(1.8)
    arr = np.array(up)
    arr = (arr > 165) * 255
    return Image.fromarray(arr.astype(np.uint8))

def build_user_words_from_roster(df: pd.DataFrame, col_nome: str) -> str:
    """Gera arquivo temporário com tokens (com acentos) dos nomes para sesgar o Tesseract."""
    tokens = []
    for name in df[col_nome].astype(str).fillna(""):
        parts = re.split(r"[^\wÁáÂâÃãÀàÉéÊêÍíÓóÔôÕõÚúÜüÇç-]+", name)
        for p in parts:
            p = p.strip()
            if len(p) >= 2:
                tokens.append(p)
    seen, vocab = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t); vocab.append(t)
    words_path = str(Path(tempfile.gettempdir()) / "user_words_pt.txt")
    with open(words_path, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab))
    return words_path

def extract_records_from_image(img: "Image.Image", ocr_lang: str, user_words_path: str, conf_min_value: int = 60):
    """
    Retorna [{'raw','nome_detectado','usp_detectado'}].
    Duas passadas com Tesseract:
      1) dígitos (whitelist 0-9) para NUSP
      2) letras pt-BR (whitelist com acentos) + --user-words (nomes do XLSX)
    Agrupa por linha (y) e junta tokens por x.
    """
    if not PYTESS_OK:
        st.error("pytesseract não disponível. Verifique o requirements e o packages.txt.")
        st.stop()

    prep = preprocess_for_ocr(img)

    # ----- Passo 1: dígitos (NUSP) -----
    usp_df = pytesseract.image_to_data(
        prep,
        lang="por",
        config="--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789",
        output_type=pytesseract.Output.DATAFRAME
    )
    usp_df = usp_df.dropna(subset=["text"])
    usp_df["text"] = usp_df["text"].astype(str).str.strip()
    usp_df = usp_df[(usp_df["text"]!="") & (usp_df["conf"].fillna(0).astype(float) >= conf_min_value)]
    if not usp_df.empty:
        usp_df["y_center"] = usp_df["top"] + usp_df["height"]/2

    # ----- Passo 2: letras (Nome pt-BR) -----
    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÁáÂâÃãÀàÉéÊêÍíÓóÔôÕõÚúÜüÇç-"
    cfg = f"--oem 3 --psm 6 -c tessedit_char_whitelist={whitelist}"
    if user_words_path:
        cfg += f" --user-words {user_words_path}"

    name_df = pytesseract.image_to_data(
        prep,
        lang="por",
        config=cfg,
        output_type=pytesseract.Output.DATAFRAME
    )
    name_df = name_df.dropna(subset=["text"])
    name_df["text"] = name_df["text"].astype(str).str.strip()
    name_df = name_df[(name_df["text"]!="") & (name_df["conf"].fillna(0).astype(float) >= conf_min_value)]
    if not name_df.empty:
        name_df["y_center"] = name_df["top"] + name_df["height"]/2

    if (usp_df is None or usp_df.empty) and (name_df is None or name_df.empty):
        return []

    # ----- Agrupar por linha (y) -----
    def group_by_line(df):
        rows = []
        if df is None or df.empty: return rows
        df = df.sort_values("y_center")
        tol = 14
        cur, last_y = [], None
        for _, r in df.iterrows():
            y = float(r["y_center"])
            if last_y is None or abs(y - last_y) <= tol:
                cur.append(r); last_y = y if last_y is None else (last_y + y)/2
            else:
                rows.append(cur); cur=[r]; last_y=y
        if cur: rows.append(cur)
        return rows

    usp_lines  = group_by_line(usp_df)
    name_lines = group_by_line(name_df)

    def compress_line(tokens):
        toks = sorted(tokens, key=lambda r: r["left"])
        txt = " ".join(str(r["text"]) for r in toks)
        return re.sub(r"\s+"," ", txt).strip(), toks

    # aprox nome↔usp por proximidade vertical
    out, used_usp = [], set()
    for n_line in name_lines:
        n_txt, n_toks = compress_line(n_line)
        if len(n_txt) < 2:
            continue
        n_y = np.mean([t["y_center"] for t in n_toks])
        best_j, best_d = None, 9999
        for j, u_line in enumerate(usp_lines):
            if j in used_usp: continue
            _, u_toks = compress_line(u_line)
            u_y = np.mean([t["y_center"] for t in u_toks])
            d = abs(n_y - u_y)
            if d < best_d:
                best_d, best_j = d, j

        usp_val, usp_raw = "", ""
        if best_j is not None and best_d <= 18:
            u_txt, _ = compress_line(usp_lines[best_j])
            cand = clean_usp_token(u_txt)
            if is_valid_usp(cand):
                usp_val = cand; usp_raw = u_txt; used_usp.add(best_j)

        out.append({"raw": (n_txt + " " + usp_raw).strip(),
                    "nome_detectado": n_txt, "usp_detectado": usp_val})

    # qualquer usp sozinho que sobrou → registra (nome vazio)
    for j, u_line in enumerate(usp_lines):
        if j in used_usp: continue
        u_txt, _ = compress_line(u_line)
        cand = clean_usp_token(u_txt)
        if is_valid_usp(cand):
            out.append({"raw": u_txt, "nome_detectado": "", "usp_detectado": cand})

    # dedup
    final, seen = [], set()
    for r in out:
        key = (r["usp_detectado"], normalize_name(r["nome_detectado"]))
        if key not in seen:
            seen.add(key); final.append(r)
    return final

# ========= UI =========
st.set_page_config(page_title="Presenças (OCR pt-BR + USP + Fuzzy)", page_icon="✅")
st.title("✅ Conferência de Presenças (OCR pt-BR + USP + Fuzzy)")
st.caption("Usa `inscritos.xlsx` (Nome, NUSP) embutido por padrão. Você pode substituir por upload ou link do Drive.")

with st.sidebar:
    st.header("Planilha de inscritos")
    uploaded_xlsx = st.file_uploader("Substituir via upload (.xlsx)", type=["xlsx"])
    drive_link = st.text_input("Ou substituir via link/ID público do Google Drive")
    st.markdown("---")
    imgs = st.file_uploader("Imagens da lista (PNG/JPEG)", type=["png","jpg","jpeg"], accept_multiple_files=True)
    st.markdown("---")
    ocr_lang = st.selectbox("Idioma base do Tesseract", ["por", "por+eng"], index=0)
    threshold = st.slider("Similaridade mínima p/ Nome (0–100)", 60, 100, 85, 1)
    conf_min_ui = st.slider("Confiança mínima do OCR (0–100)", 50, 90, 60, 1)  # valor padrão 60
    show_debug = st.checkbox("Mostrar OCR bruto (debug)", value=False)

if st.button("🔎 Processar", type="primary", use_container_width=True):
    # 1) Inscritos: usa override se enviado; senão embutido
    inscritos_override = read_xlsx_from_upload_or_drive(uploaded_xlsx, drive_link)
    if inscritos_override is None:
        st.info("Usando planilha embutida: inscritos.xlsx (colunas Nome, NUSP).")
        inscritos = load_default_inscritos()
    else:
        inscritos = inscritos_override

    # usa a confiança mínima escolhida na UI
    conf_min_value = int(conf_min_ui)

    if not imgs:
        st.error("Envie ao menos uma imagem da lista manuscrita (nome + NUSP na mesma linha).")
        st.stop()

    # 2) Dicionário de palavras (pt-BR) com base nos inscritos
    user_words_path = build_user_words_from_roster(inscritos, col_nome=COL_NOME)

    # 3) OCR → registros
    ocr_recs = []
    for f in imgs:
        try:
            img = Image.open(f).convert("RGB")
            recs = extract_records_from_image(img, ocr_lang, user_words_path, conf_min_value)
            for r in recs:
                r["arquivo"] = getattr(f, "name", "imagem")
            ocr_recs.extend(recs)
        except Exception as e:
            st.warning(f"Falha ao ler {getattr(f,'name','imagem')}: {e}")

    if show_debug:
        st.subheader("OCR bruto")
        st.dataframe(pd.DataFrame(ocr_recs), use_container_width=True)

    if len(ocr_recs) == 0:
        st.error("Não consegui extrair texto das imagens. Tente fotos mais nítidas/retilíneas e ajuste a 'Confiança mínima'.")
        st.stop()

    ocr_df = pd.DataFrame(ocr_recs)
    ocr_df["__nome_norm"] = ocr_df["nome_detectado"].map(normalize_name)
    ocr_df["__usp_norm"]  = ocr_df["usp_detectado"].astype(str).str.replace(r"\D","", regex=True)

    # 4) MATCH — prioridade USP (exato), depois Nome (fuzzy)
    presentes_rows, faltantes_rows = [], []
    usados_idx = set()

    usp_to_idx = {}
    for i, row in ocr_df.iterrows():
        usp = row["__usp_norm"]
        if usp:
            usp_to_idx.setdefault(usp, []).append(i)

    for _, row in inscritos.iterrows():
        nome_ins = row[COL_NOME]
        usp_ins  = row[COL_USP]
        nome_norm = row["__nome_norm"]
        usp_norm  = row["__usp_norm"]

        matched = False
        # 4a) USP exato
        if usp_norm and usp_norm in usp_to_idx:
            cand_list = usp_to_idx[usp_norm]
            idx = next((j for j in cand_list if j not in usados_idx), None)
            if idx is not None:
                usados_idx.add(idx)
                presentes_rows.append({
                    COL_NOME: nome_ins, COL_USP: usp_ins,
                    "presente": True, "criterio": "USP",
                    "linha_ocr": ocr_df.loc[idx, "raw"], "similaridade_nome": ""
                })
                matched = True

        # 4b) Nome (fuzzy)
        if not matched:
            mask = ~ocr_df.index.isin(usados_idx)
            cands = ocr_df[mask]
            match, score, idx_local = best_match(nome_norm, cands["__nome_norm"].tolist())
            if idx_local is not None and score >= threshold:
                idx_global = cands.index.tolist()[idx_local]
                usados_idx.add(idx_global)
                presentes_rows.append({
                    COL_NOME: nome_ins, COL_USP: usp_ins,
                    "presente": True, "criterio": "NOME",
                    "linha_ocr": ocr_df.loc[idx_global, "raw"], "similaridade_nome": int(score)
                })
                matched = True

        if not matched:
            faltantes_rows.append({
                COL_NOME: nome_ins, COL_USP: usp_ins,
                "presente": False, "criterio": "", "linha_ocr": "", "similaridade_nome": 0
            })

    # 5) DataFrames com schema fixo
    presentes_df = pd.DataFrame(
        presentes_rows,
        columns=[COL_NOME, COL_USP, "presente", "criterio", "linha_ocr", "similaridade_nome"]
    )
    faltantes_df = pd.DataFrame(
        faltantes_rows,
        columns=[COL_NOME, COL_USP, "presente", "criterio", "linha_ocr", "similaridade_nome"]
    )

    # 6) Detectados que não estão na planilha
    nao_usados = ocr_df[~ocr_df.index.isin(usados_idx)].copy()
    nao_inscritos_df = nao_usados[
        (nao_usados["__nome_norm"]!="") | (nao_usados["__usp_norm"]!="")
    ][["nome_detectado","usp_detectado","raw","arquivo"]].rename(columns={
        "nome_detectado":"nome_detectado_ocr",
        "usp_detectado":"usp_detectado_ocr",
        "raw":"linha_ocr"
    })

    # 7) KPIs e resultados
    c1, c2, c3 = st.columns(3)
    c1.metric("Inscritos", len(inscritos))
    c2.metric("Presentes", len(presentes_df))
    c3.metric("Faltantes", len(faltantes_df))

    st.subheader("✅ Presentes (nome da planilha + NUSP)")
    st.dataframe(presentes_df[[COL_NOME, COL_USP, "criterio", "linha_ocr", "similaridade_nome"]], use_container_width=True)
    build_download(presentes_df, "Baixar Presentes (CSV)", "presentes.csv")

    st.subheader("❌ Faltantes")
    st.dataframe(faltantes_df[[COL_NOME, COL_USP, "criterio", "linha_ocr", "similaridade_nome"]], use_container_width=True)
    build_download(faltantes_df, "Baixar Faltantes (CSV)", "faltantes.csv")

    st.subheader("⚠️ Detectados mas não-inscritos (OCR)")
    st.dataframe(nao_inscritos_df, use_container_width=True)
    build_download(nao_inscritos_df, "Baixar Não-inscritos (CSV)", "nao_inscritos.csv")

    st.info("Casamento: USP exato → Nome (fuzzy). Ajuste 'Confiança mínima' e use fotos retas/nítidas se o OCR trouxer ruído.")
