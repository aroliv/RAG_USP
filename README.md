# ✅ Conferência de Presenças (OCR + USP + Fuzzy)

- Usa **planilha embutida** `inscritos.xlsx` (colunas **Nome**, **NUSP**) por padrão.
- Opcionalmente, você pode **substituir** por upload do `.xlsx` ou por link/ID público do Google Drive.

## Como rodar local
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
