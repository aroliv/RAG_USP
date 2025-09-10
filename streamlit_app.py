# app.py
import os, re, json, base64, tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from unidecode import unidecode
from rapidfuzz import process, fuzz

# ========= CONFIG BÁSICA =========
COL_NOME = "Nome"  # colunas do seu XLSX
COL_USP  = "NUSP"

st.set_page_config(page_title="Presenças (LLM de Visão + Match)", page_icon="✅")

# ========= UTIL =========
def normalize_name(s: str) -> str:
    s = unidecode(str(s or ""))
    s = re.sub(r"[^a-zA-Z\s]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def prepare_inscritos_df(df: pd.DataFrame) -> pd.DataFrame:
    # tenta renomear variações para Nome/NUSP
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

# ========= CLIENTE LLM (Gemini 1.5 por padrão) =========
# Requer variável de ambiente GEMINI_API_KEY
USE_GEMINI = True  # se quiser plugar outro provedor, veja stub abaixo

def img_to_base64(img: Image.Image) -> str:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.convert("RGB").save(tmp.name, format="JPEG", quality=92)
        with open(tmp.name, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

def call_vision_llm(images: list[Image.Image], lingua: str = "pt-BR") -> list[dict]:
    """
    Retorna uma lista de dicts: [{"nome": "...", "nusp": "########"}, ...]
    O LLM deve extrair por LINHA (nome + NUSP). Aceitamos nome vazio se só tiver NUSP.
    """
    if USE_GEMINI:
        try:
            import google.generativeai as genai
        except Exception:
            st.error("Faltou instalar 'google-generativeai' no requirements.")
            st.stop()

        api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            st.error("Defina GEMINI_API_KEY (em Secrets ou variável de ambiente).")
            st.stop()
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",  # rápido e mais barato; troque por pro se quiser
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            }
        )

        # prompt com instruções rígidas + JSON schema textual
        system_prompt = {
            "role": "user",
            "parts": [(
                "Você receberá 1..N fotos de listas de presença manuscritas em português do Brasil. "
                "Extraia cada LINHA como um objeto JSON com os campos:\n"
                "  - nome: string (pode estar abreviado; mantenha acentos).\n"
                "  - nusp: string com 6 a 9 dígitos (se ausente, devolva \"\").\n"
                "Regra:\n"
                "  - Cada objeto representa uma ÚNICA PESSOA por linha.\n"
                "  - Se houver vários números na mesma linha, escolha o que parecer USP (6–9 dígitos).\n"
                "  - Se não houver nome legível mas houver número, devolva nome=\"\" e o nusp.\n"
                "  - Remova números/índices iniciais (ex.: '12) ...').\n"
                "  - NÃO devolva comentários ou texto fora de linhas (cabeçalhos/rodapés).\n"
                "Responda APENAS com JSON como uma lista: [{\"nome\":\"...\",\"nusp\":\"...\"}, ...]."
            )]
        }

        # monta a entrada com todas as imagens
        contents = [system_prompt]
        for img in images:
            contents.append(
                {"role": "user",
                 "parts": [{"inline_data": {"mime_type": "image/jpeg", "data": img_to_base64(img)}}]}
            )

        try:
            resp = model.generate_content(contents)
            text = resp.text or "[]"
            data = json.loads(text)
        except Exception as e:
            st.error(f"Falha ao chamar LLM de visão: {e}")
            st.stop()

        # saneamento básico
        out = []
        for r in data if isinstance(data, list) else []:
            nome = str(r.get("nome","")).strip()
            nusp = re.sub(r"\D","", str(r.get("nusp","")))
            if nome == "" and nusp == "":
                continue
            out.append({"nome": nome, "nusp": nusp})
        return out

    # ===== Stub para outros provedores (ex.: OpenAI/Claude) =====
    # Implemente chamada e retorne [{"nome":"...", "nusp":"..."}]
    raise RuntimeError("Nenhum provedor de LLM de visão configurado.")

# ========= UI =========
st.title("✅ Presenças por LLM de Visão (sem OCR)")
st.caption("Sobe imagens manuscritas → LLM extrai {nome, nusp} → cruzamos com inscritos.xlsx (Nome, NUSP).")

with st.sidebar:
    st.header("Planilha de inscritos")
    uploaded_xlsx = st.file_uploader("Substituir via upload (.xlsx) — opcional", type=["xlsx"])
    drive_link = st.text_input("Ou substituir via link/ID público do Google Drive (opcional)")
    st.markdown("---")
    imgs = st.file_uploader("Imagens da lista (PNG/JPEG)", type=["png","jpg","jpeg"], accept_multiple_files=True)
    st.markdown("---")
    threshold = st.slider("Similaridade mínima p/ nome (fallback)", 60, 100, 85, 1)
    min_name_len = st.slider("Filtro: tamanho mínimo do nome OCR", 0, 6, 3, 1)
    show_extracted = st.checkbox("Mostrar extração do LLM (debug)", value=False)

# leitura da planilha (default embutida + override)
def read_xlsx_from_upload_or_drive(uploaded_file, drive_link_or_id: str) -> pd.DataFrame | None:
    if uploaded_file is not None:
        return prepare_inscritos_df(pd.read_excel(uploaded_file))
    if drive_link_or_id:
        try:
            import gdown
        except ImportError:
            st.error("Faltou 'gdown' no requirements.")
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

if st.button("🔎 Processar", type="primary", use_container_width=True):
    # 1) Carrega inscritos
    inscritos_override = read_xlsx_from_upload_or_drive(uploaded_xlsx, drive_link)
    if inscritos_override is None:
        st.info("Usando planilha embutida: inscritos.xlsx (colunas Nome, NUSP).")
        inscritos = load_default_inscritos()
    else:
        inscritos = inscritos_override

    if not imgs:
        st.error("Envie pelo menos uma imagem da lista manuscrita.")
        st.stop()

    # 2) Chama LLM de visão para extrair {nome, nusp}
    images = []
    for f in imgs:
        try:
            images.append(Image.open(f).convert("RGB"))
        except Exception as e:
            st.warning(f"Falha ao abrir {getattr(f,'name','imagem')}: {e}")
    extracted = call_vision_llm(images)

    # 2a) Sanitização + filtros anti-ruído
    rows = []
    for r in extracted:
        nome = str(r.get("nome","")).strip()
        nusp = re.sub(r"\D","", str(r.get("nusp","")))
        if nome == "" and nusp == "":
            continue
        if nome and len(normalize_name(nome)) < min_name_len and not nusp:
            # descarta nomes muito curtinhos sem NUSP
            continue
        rows.append({"nome_detectado": nome, "nusp_detectado": nusp})
    ocr_df = pd.DataFrame(rows)
    if show_extracted:
        st.subheader("Extraído pelo LLM (debug)")
        st.dataframe(ocr_df, use_container_width=True)
    if ocr_df.empty:
        st.error("O LLM não conseguiu extrair nada útil dessas imagens.")
        st.stop()

    # 3) Normalização p/ match
    ocr_df["__nome_norm"] = ocr_df["nome_detectado"].map(normalize_name)
    ocr_df["__usp_norm"]  = ocr_df["nusp_detectado"].astype(str).str.replace(r"\D","", regex=True)

    # 4) MATCH — USP exato, depois nome (fuzzy) como fallback
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
                    "presente": True, "criterio": "USP", "linha_origem": ocr_df.loc[idx, "nome_detectado"],
                    "similaridade_nome": ""
                })
                matched = True

        # 4b) Nome (fuzzy) — se LLM não trouxe NUSP
        if not matched:
            mask = ~ocr_df.index.isin(usados_idx)
            cands = ocr_df[mask]
            match, score, idx_local = best_match(nome_norm, cands["__nome_norm"].tolist())
            if idx_local is not None and score >= threshold:
                idx_global = cands.index.tolist()[idx_local]
                usados_idx.add(idx_global)
                presentes_rows.append({
                    COL_NOME: nome_ins, COL_USP: usp_ins,
                    "presente": True, "criterio": "NOME(LLM)", "linha_origem": ocr_df.loc[idx_global, "nome_detectado"],
                    "similaridade_nome": int(score)
                })
                matched = True

        if not matched:
            faltantes_rows.append({
                COL_NOME: nome_ins, COL_USP: usp_ins,
                "presente": False, "criterio": "", "linha_origem": "", "similaridade_nome": 0
            })

    presentes_df = pd.DataFrame(
        presentes_rows, columns=[COL_NOME, COL_USP, "presente", "criterio", "linha_origem", "similaridade_nome"]
    )
    faltantes_df = pd.DataFrame(
        faltantes_rows, columns=[COL_NOME, COL_USP, "presente", "criterio", "linha_origem", "similaridade_nome"]
    )

    # 5) Detectados mas não-inscritos
    nao_usados = ocr_df[~ocr_df.index.isin(usados_idx)].copy()
    nao_inscritos_df = nao_usados[
        (nao_usados["__nome_norm"]!="") | (nao_usados["__usp_norm"]!="")
    ][["nome_detectado","nusp_detectado"]].rename(columns={
        "nome_detectado":"nome_detectado_llm",
        "nusp_detectado":"nusp_detectado_llm",
    })

    # Filtro anti-ruído: remove nomes < min_name_len sem NUSP
    def is_noise(name, usp):
        if usp and usp.strip() != "":
            return False
        return len(normalize_name(name)) < min_name_len
    nao_inscritos_df = nao_inscritos_df[~nao_inscritos_df.apply(
        lambda r: is_noise(r["nome_detectado_llm"], r["nusp_detectado_llm"]), axis=1
    )].reset_index(drop=True)

    # 6) KPIs e saída
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

    st.subheader("⚠️ Detectados mas não-inscritos (extraídos pelo LLM)")
    st.dataframe(nao_inscritos_df, use_container_width=True)
    build_download(nao_inscritos_df, "Baixar Não-inscritos (CSV)", "nao_inscritos.csv")
