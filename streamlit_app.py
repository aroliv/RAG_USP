import re
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from unidecode import unidecode
from rapidfuzz import process, fuzz

# ---------------- OCR backends (EasyOCR preferido; pytesseract como fallback) ----------------
EASYOCR_OK = False
PYTESS_OK = True
easy_reader = None

try:
    import easyocr
    easy_reader = easyocr.Reader(['pt', 'pt_br', 'en'], gpu=False, verbose=False)
    EASYOCR_OK = True
except Exception:
    EASYOCR_OK = False

try:
    import pytesseract
    PYTESS_OK = True
except Exception:
    PYTESS_OK = False

# ---------------- Constantes e utils ----------------
USP_RX = re.compile(r"\b(\d{6,9})\b")  # USP com 6–9 dígitos

COL_NOME = "Nome"   # colunas EXATAS do seu xlsx
COL_USP  = "NUSP"

def normalize_name(s: str) -> str:
    s = unidecode(str(s or ""))
    s = re.sub(r"[^a-zA-Z\s]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s

@st.cache_data
def load_default_inscritos():
    """Carrega o inscritos.xlsx do repositório (mesmo diretório do app)."""
    df = pd.read_excel("inscritos.xlsx")
    # força tipos corretos
    df[COL_NOME] = df[COL_NOME].astype(str).str.strip()
    df[COL_USP]  = df[COL_USP].astype(str).str.replace(r"\D", "", regex=True).str.strip()
    df["__nome_norm"] = df[COL_NOME].map(normalize_name)
    df["__usp_norm"]  = df[COL_USP]
    return df

def read_xlsx_from_upload_or_drive(uploaded_file, drive_link_or_id: str) -> pd.DataFrame | None:
    """Se o usuário enviar algo, lê; caso contrário, retorna None para usarmos o default embutido."""
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        return prepare_inscritos_df(df)

    if drive_link_or_id:
        try:
            import gdown
        except ImportError:
            st.error("gdown não instalado. Verifique o requirements.txt")
            st.stop()

        file_id = None
        m = re.search(r"/d/([A-Za-z0-9_-]+)", drive_link_or_id)
        if m:
            file_id = m.group(1)
        elif re.fullmatch(r"[A-Za-z0-9_-]{20,}", drive_link_or_id):
            file_id = drive_link_or_id

        if not file_id:
            st.error("Cole um link/ID público do Google Drive válido (Compartilhar → Qualquer pessoa com o link).")
            st.stop()

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "inscritos.xlsx"
            import gdown
            gdown.download(id=file_id, output=str(out), quiet=True)
            df = pd.read_excel(out)
            return prepare_inscritos_df(df)

    return None  # não veio nada → usamos default embutido

def prepare_inscritos_df(df: pd.DataFrame) -> pd.DataFrame:
    """Adapta qualquer DF recebido para as colunas exatas COL_NOME/COL_USP."""
    # mapeia títulos possíveis para os nomes exatos:
    title_map = {}
    for c in df.columns:
        k = normalize_name(c)
        if k in {"nome", "nome completo", "aluno", "participante"} and COL_NOME not in title_map.values():
            title_map[c] = COL_NOME
        if k in {"usp", "nusp", "numero usp", "n usp", "matricula", "ra"} and COL_USP not in title_map.values():
            title_map[c] = COL_USP

    # renomeia; se já estão corretas, nada muda
    df = df.rename(columns=title_map).copy()

    # valida existência das colunas
    if COL_NOME not in df.columns or COL_USP not in df.columns:
        st.error(f"Planilha enviada precisa ter colunas '{COL_NOME}' e '{COL_USP}'.")
        st.stop()

    df[COL_NOME] = df[COL_NOME].astype(str).str.strip()
    df[COL_USP]  = df[COL_USP].astype(str).str.replace(r"\D", "", regex=True).str.strip()
    df["__nome_norm"] = df[COL_NOME].map(normalize_name)
    df["__usp_norm"]  = df[COL_USP]
    return df

def ocr_easyocr_lines(img: Image.Image) -> list[str]:
    arr = np.array(img.convert("RGB"))
    chunks = easy_reader.readtext(arr, detail=0, paragraph=True)
    lines = []
    for ch in chunks:
        for ln in str(ch).split("\n"):
            ln = ln.strip()
            if ln:
                lines.append(ln)
    return lines

def ocr_tesseract_lines(img: Image.Image, lang: str) -> list[str]:
    gray = ImageOps.grayscale(img)
    arr = np.array(gray)
    arr = (arr > 180) * 255
    bin_img = Image.fromarray(arr.astype(np.uint8))
    text = pytesseract.image_to_string(bin_img, lang=lang)
    return [ln.strip() for ln in text.split("\n") if ln.strip()]

def extract_records_from_image(img: "Image.Image", ocr_lang: str):
    """Retorna lista de dicts {raw, nome_detectado, usp_detectado}."""
    if EASYOCR_OK:
        raw_lines = ocr_easyocr_lines(img)
    elif PYTESS_OK:
        raw_lines = ocr_tesseract_lines(img, ocr_lang)
    else:
        st.error("Nenhum backend de OCR disponível. Verifique o requirements.txt (easyocr/pytesseract).")
        st.stop()

    cleaned = []
    for ln in raw_lines:
        if len(ln) < 3 or not re.search(r"[A-Za-zÀ-ÿ0-9]", ln):
            continue
        ln = re.sub(r"^\s*\d+[\).:-]?\s*", "", ln)  # remove numeração "35) Fulano"
        cleaned.append(ln.strip())

    recs = []
    for ln in cleaned:
        # capturar USP mesmo se vier colado no nome
        compact = ln.replace(" ", "")
        usp_match = USP_RX.search(compact) or USP_RX.search(ln)
        usp = usp_match.group(1) if usp_match else ""
        name_part = ln
        if usp:
            name_part = re.sub(USP_RX, " ", ln)
        name_part = re.sub(r"[-_/.,;:]+", " ", name_part).strip()
        if len(name_part) < 2 and not usp:
            continue
        recs.append({"raw": ln, "nome_detectado": name_part, "usp_detectado": usp})
    return recs

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

# ---------------- UI ----------------
st.set_page_config(page_title="Presenças (OCR + USP + Fuzzy)", page_icon="✅")
st.title("✅ Conferência de Presenças (OCR + USP + Fuzzy)")
st.caption("Usa planilha embutida `inscritos.xlsx` (colunas Nome, NUSP) por padrão. Você pode substituir por upload ou link do Drive.")

with st.sidebar:
    st.header("Planilha de inscritos")
    uploaded_xlsx = st.file_uploader("Substituir via upload (.xlsx)", type=["xlsx"])
    drive_link = st.text_input("Ou substituir via link/ID público do Google Drive")
    st.markdown("---")
    imgs = st.file_uploader("Imagens da lista (PNG/JPEG)", type=["png","jpg","jpeg"], accept_multiple_files=True)
    st.markdown("---")
    ocr_lang = st.selectbox("Idioma (fallback Tesseract)", ["por", "por+eng", "eng"], index=1)
    threshold = st.slider("Similaridade mínima para nome (0–100)", 60, 100, 85, 1)
    show_debug = st.checkbox("Mostrar OCR bruto (debug)", value=False)

if st.button("🔎 Processar", type="primary", use_container_width=True):
    # 1) Inscritos: substitui se o usuário enviar; senão usa embutido
    inscritos_override = read_xlsx_from_upload_or_drive(uploaded_xlsx, drive_link)
    if inscritos_override is None:
        st.info("Usando planilha embutida: inscritos.xlsx (colunas Nome, NUSP).")
        inscritos = load_default_inscritos()
    else:
        inscritos = inscritos_override

    if not imgs:
        st.error("Envie ao menos uma imagem da lista manuscrita (nome + USP por linha).")
        st.stop()

    # 2) OCR → registros
    ocr_recs = []
    for f in imgs:
        try:
            from PIL import Image
            img = Image.open(f).convert("RGB")
            recs = extract_records_from_image(img, ocr_lang)
            for r in recs:
                r["arquivo"] = getattr(f, "name", "imagem")
            ocr_recs.extend(recs)
        except Exception as e:
            st.warning(f"Falha ao ler {getattr(f,'name','imagem')}: {e}")

    if show_debug:
        st.subheader("OCR bruto")
        st.dataframe(pd.DataFrame(ocr_recs), use_container_width=True)

    if len(ocr_recs) == 0:
        st.error("Não consegui extrair texto das imagens. Tente fotos mais nítidas/retilíneas.")
        st.stop()

    ocr_df = pd.DataFrame(ocr_recs)
    ocr_df["__nome_norm"] = ocr_df["nome_detectado"].map(normalize_name)
    ocr_df["__usp_norm"]  = ocr_df["usp_detectado"].astype(str).str.replace(r"\D", "", regex=True)

    # 3) MATCH — prioridade USP (exato), depois nome (fuzzy)
    presentes_rows, faltantes_rows = [], []
    usados_idx = set()

    # index por USP lido
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
        # 3a) USP exato
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

        # 3b) Nome (fuzzy)
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

    presentes_df = pd.DataFrame(presentes_rows)
    faltantes_df = pd.DataFrame(faltantes_rows)

    # 4) Detectados que não estão na planilha
    nao_usados = ocr_df[~ocr_df.index.isin(usados_idx)].copy()
    nao_inscritos_df = nao_usados[
        (nao_usados["__nome_norm"]!="") | (nao_usados["__usp_norm"]!="")
    ][["nome_detectado","usp_detectado","raw","arquivo"]].rename(columns={
        "nome_detectado":"nome_detectado_ocr",
        "usp_detectado":"usp_detectado_ocr",
        "raw":"linha_ocr"
    })

    # 5) KPIs e resultados
    c1, c2, c3 = st.columns(3)
    c1.metric("Inscritos", len(inscritos))
    c2.metric("Presentes", (presentes_df["presente"]==True).sum())
    c3.metric("Faltantes", (faltantes_df["presente"]==False).sum())

    st.subheader("✅ Presentes")
    st.dataframe(presentes_df, use_container_width=True)
    build_download(presentes_df, "Baixar Presentes (CSV)", "presentes.csv")

    st.subheader("❌ Faltantes")
    st.dataframe(faltantes_df, use_container_width=True)
    build_download(faltantes_df, "Baixar Faltantes (CSV)", "faltantes.csv")

    st.subheader("⚠️ Detectados mas não-inscritos")
    st.dataframe(nao_inscritos_df, use_container_width=True)
    build_download(nao_inscritos_df, "Baixar Não-inscritos (CSV)", "nao_inscritos.csv")

    st.info("Ordem de casamento: USP exato → Nome (fuzzy). Aumente/diminua a similaridade conforme a qualidade do OCR.")
