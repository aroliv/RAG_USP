# ===============================================================
#  app.py
# ===============================================================

import os, re, json, base64, tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from unidecode import unidecode
from rapidfuzz import process, fuzz

# ========= CONFIG BÁSICA =========
COL_NOME = "Nome"
COL_USP  = "NUSP"

st.set_page_config(page_title="Presenças (LLM de Visão + Match)", page_icon="✅")

# ============================================
# Normalização básica
# ============================================

def normalize_name(s: str) -> str:
    s = unidecode(str(s or ""))
    s = re.sub(r"[^a-zA-Z\s]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def prepare_inscritos_df(df: pd.DataFrame) -> pd.DataFrame:
    title_map = {}
    for c in df.columns:
        k = normalize_name(c)
        if k in {"nome", "nome completo", "aluno", "participante"} and COL_NOME not in df.columns:
            title_map[c] = COL_NOME
        if k in {"usp", "nusp", "numero usp", "n usp", "matricula", "ra"} and COL_USP not in df.columns:
            title_map[c] = COL_USP

    df = df.rename(columns=title_map).copy()

    if COL_NOME not in df.columns or COL_USP not in df.columns:
        st.error(f"A planilha precisa ter as colunas '{COL_NOME}' e '{COL_USP}'.")
        st.stop()

    df[COL_NOME] = df[COL_NOME].astype(str).str.strip()
    df[COL_USP]  = df[COL_USP].astype(str).str.replace(r"\D","", regex=True).str.strip()

    df["__nome_norm"] = df[COL_NOME].map(normalize_name)
    df["__usp_norm"]  = df[COL_USP]

    return df

@st.cache_data
def load_default_inscritos():
    df = pd.read_excel("inscritos.xlsx")
    return prepare_inscritos_df(df)


def img_to_base64(img: Image.Image) -> str:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.convert("RGB").save(tmp.name, format="JPEG", quality=92)
        with open(tmp.name, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


# ============================================
#  NOVA VERSÃO DA FUNÇÃO — compatível com 0.8.x
# ============================================

USE_GEMINI = True

def call_vision_llm(images: list[Image.Image], lingua: str = "pt-BR") -> list[dict]:

    if not USE_GEMINI:
        raise RuntimeError("Nenhum provedor configurado.")

    # Import dinâmico
    try:
        import google.generativeai as genai
    except ImportError:
        st.error("Instale o pacote 'google-generativeai>=0.8.0'.")
        st.stop()

    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Defina GEMINI_API_KEY.")
        st.stop()

    genai.configure(api_key=api_key)

    # MODELO: agora exige prefixo "models/"
    model = genai.GenerativeModel(
        model_name="models/gemini-2.5-flash",
        generation_config={
            "temperature": 0.1,
            "response_mime_type": "application/json",
        }
    )

    # NOVO FORMATO DO PROMPT
    parts = []

    # Texto inicial como string normal
    parts.append(
        "Você receberá 1..N fotos de listas de presença manuscritas em português do Brasil. "
        "Extraia cada LINHA como um objeto JSON com os campos:\n"
        "  - nome: string\n"
        "  - nusp: string com 6 a 9 dígitos (ou \"\" se ausente)\n"
        "Regras:\n"
        "  - Cada objeto representa uma única pessoa por linha.\n"
        "  - Se houver vários números, escolha o que parecer USP.\n"
        "  - Se nome ilegível mas houver número, use nome=\"\".\n"
        "  - Remova índices como '12)'.\n"
        "  - Ignore cabeçalhos.\n"
        "Responda APENAS com JSON: [{\"nome\":\"...\",\"nusp\":\"...\"}, ...]."
    )

    # Cada imagem vira um item com mime_type + data
    for img in images:
        parts.append({
            "mime_type": "image/jpeg",
            "data": img_to_base64(img)
        })

    try:
        resp = model.generate_content(parts)
        raw = resp.text or "[]"
        data = json.loads(raw)
    except Exception as e:
        st.error(f"Falha ao chamar LLM de visão: {e}")
        st.stop()

    # Limpeza
    out = []
    if isinstance(data, list):
        for r in data:
            nome = str(r.get("nome", "")).strip()
            nusp = re.sub(r"\D", "", str(r.get("nusp", "")))
            if nome == "" and nusp == "":
                continue
            out.append({"nome": nome, "nusp": nusp})

    return out


# ============================================
# Match + Interface
# ============================================

def build_download(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv", use_container_width=True)

def best_match(name_norm: str, candidates_norm: list[str]):
    if not candidates_norm:
        return None, 0, None
    match, score, idx = process.extractOne(
        name_norm, candidates_norm, scorer=fuzz.token_set_ratio
    )
    return match, score, idx


# =====================================================
# UI Streamlit
# =====================================================

st.title("✅ Presenças por LLM de Visão (Gemini 2.5)")
st.caption("Suba listas manuscritas → LLM extrai dados → match com inscritos.xlsx.")

with st.sidebar:
    st.header("Planilha de inscritos")
    uploaded_xlsx = st.file_uploader("Substituir via upload (.xlsx)", type=["xlsx"])
    drive_link = st.text_input("Ou ID/link público do Google Drive")
    st.markdown("---")
    imgs = st.file_uploader("Imagens da lista (PNG/JPEG)", type=["png","jpg","jpeg"], accept_multiple_files=True)
    st.markdown("---")
    threshold = st.slider("Similaridade mínima p/ nome (fallback)", 60, 100, 85, 1)
    min_name_len = st.slider("Filtro: tamanho mínimo do nome OCR", 0, 6, 3, 1)
    show_extracted = st.checkbox("Mostrar extração do LLM (debug)", value=False)


# ==== Upload da planilha (manter intacto) ====

def read_xlsx_from_upload_or_drive(uploaded_file, drive_link_or_id: str):
    if uploaded_file is not None:
        return prepare_inscritos_df(pd.read_excel(uploaded_file))
    if drive_link_or_id:
        try:
            import gdown
        except ImportError:
            st.error("Faltou 'gdown'.")
            st.stop()

        file_id = None
        m = re.search(r"/d/([A-Za-z0-9_-]+)", drive_link_or_id)
        if m:
            file_id = m.group(1)
        elif re.fullmatch(r"[A-Za-z0-9_-]{20,}", drive_link_or_id):
            file_id = drive_link_or_id

        if not file_id:
            st.error("Cole um link/ID do Drive com acesso público.")
            st.stop()

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "inscritos.xlsx"
            import gdown
            gdown.download(id=file_id, output=str(out), quiet=True)
            return prepare_inscritos_df(pd.read_excel(out))

    return None


# =====================================================
# Botão principal
# =====================================================

if st.button("🔎 Processar", type="primary", use_container_width=True):

    inscritos_override = read_xlsx_from_upload_or_drive(uploaded_xlsx, drive_link)

    if inscritos_override is None:
        st.info("Usando planilha embutida: inscritos.xlsx.")
        inscritos = load_default_inscritos()
    else:
        inscritos = inscritos_override

    if not imgs:
        st.error("Envie pelo menos uma imagem da lista manuscrita.")
        st.stop()

    # Abrir imagens
    images = []
    for f in imgs:
        try:
            images.append(Image.open(f).convert("RGB"))
        except Exception as e:
            st.warning(f"Falha ao abrir {getattr(f,'name','imagem')}: {e}")

    # Extrair via LLM
    extracted = call_vision_llm(images)

    rows = []
    for r in extracted:
        nome = str(r.get("nome","")).strip()
        nusp = re.sub(r"\D","", str(r.get("nusp","")))
        if nome == "" and nusp == "":
            continue
        if nome and len(normalize_name(nome)) < min_name_len and not nusp:
            continue
        rows.append({"nome_detectado": nome, "nusp_detectado": nusp})

    ocr_df = pd.DataFrame(rows)

    if show_extracted:
        st.subheader("Extraído pelo LLM (debug)")
        st.dataframe(ocr_df, use_container_width=True)

    if ocr_df.empty:
        st.error("O LLM não conseguiu extrair nada útil.")
        st.stop()

    # Normalização
    ocr_df["__nome_norm"] = ocr_df["nome_detectado"].map(normalize_name)
    ocr_df["__usp_norm"]  = ocr_df["nusp_detectado"].astype(str).str.replace(r"\D","", regex=True)

    presentes_rows, faltantes_rows = [], []
    usados_idx = set()

    usp_to_idx = {}
    for i, row in ocr_df.iterrows():
        usp = row["__usp_norm"]
        if usp:
            usp_to_idx.setdefault(usp, []).append(i)

    # MATCH
    for _, row in inscritos.iterrows():
        nome_ins = row[COL_NOME]
        usp_ins  = row[COL_USP]
        nome_norm = row["__nome_norm"]
        usp_norm  = row["__usp_norm"]

        matched = False

        # USP exato
        if usp_norm and usp_norm in usp_to_idx:
            cand_list = usp_to_idx[usp_norm]
            idx = next((j for j in cand_list if j not in usados_idx), None)
            if idx is not None:
                usados_idx.add(idx)
                presentes_rows.append({
                    COL_NOME: nome_ins, COL_USP: usp_ins,
                    "presente": True, "criterio": "USP",
                    "linha_origem": ocr_df.loc[idx, "nome_detectado"],
                    "similaridade_nome": ""
                })
                matched = True

        # Nome fuzzy
        if not matched:
            mask = ~ocr_df.index.isin(usados_idx)
            cands = ocr_df[mask]
            match, score, idx_local = best_match(nome_norm, cands["__nome_norm"].tolist())
            if idx_local is not None and score >= threshold:
                idx_global = cands.index.tolist()[idx_local]
                usados_idx.add(idx_global)
                presentes_rows.append({
                    COL_NOME: nome_ins, COL_USP: usp_ins,
                    "presente": True, "criterio": "NOME(LLM)",
                    "linha_origem": ocr_df.loc[idx_global, "nome_detectado"],
                    "similaridade_nome": int(score)
                })
                matched = True

        if not matched:
            faltantes_rows.append({
                COL_NOME: nome_ins, COL_USP: usp_ins,
                "presente": False, "criterio": "",
                "linha_origem": "", "similaridade_nome": 0
            })

    presentes_df = pd.DataFrame(presentes_rows)
    faltantes_df = pd.DataFrame(faltantes_rows)

    # NÃO INSCRITOS
    nao_usados = ocr_df[~ocr_df.index.isin(usados_idx)].copy()
    nao_inscritos_df = nao_usados[
        (nao_usados["__nome_norm"]!="") | (nao_usados["__usp_norm"]!="")
    ][["nome_detectado","nusp_detectado"]].rename(columns={
        "nome_detectado":"nome_detectado_llm",
        "nusp_detectado":"nusp_detectado_llm",
    })

    def is_noise(name, usp):
        if usp and usp.strip():
            return False
        return len(normalize_name(name)) < min_name_len

    nao_inscritos_df = nao_inscritos_df[
        ~nao_inscritos_df.apply(lambda r: is_noise(r["nome_detectado_llm"], r["nusp_detectado_llm"]), axis=1)
    ].reset_index(drop=True)

    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Inscritos", len(inscritos))
    c2.metric("Presentes", len(presentes_df))
    c3.metric("Faltantes", len(faltantes_df))

    st.subheader("✅ Presentes")
    st.dataframe(presentes_df, use_container_width=True)
    build_download(presentes_df, "Baixar Presentes (CSV)", "presentes.csv")

    st.subheader("❌ Faltantes")
    st.dataframe(faltantes_df, use_container_width=True)
    build_download(faltantes_df, "Baixar Faltantes (CSV)", "faltantes.csv")

    st.subheader("⚠️ Detectados mas não-inscritos")
    st.dataframe(nao_inscritos_df, use_container_width=True)
    build_download(nao_inscritos_df, "Baixar Não-inscritos (CSV)", "nao_inscritos.csv")

