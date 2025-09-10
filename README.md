# Presenças por LLM de Visão (sem OCR)

- Sobe **imagens** da lista manuscrita.
- Um **LLM de visão** (Gemini 1.5 por padrão) extrai `{nome, nusp}` em **JSON**.
- Cruzamos com `inscritos.xlsx` (colunas **Nome**, **NUSP**) e geramos:
  - ✅ Presentes
  - ❌ Faltantes
  - ⚠️ Detectados não-inscritos

## Rodar local
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
export GEMINI_API_KEY=SEU_TOKEN_AQUI
streamlit run app.py
