# ✅ Conferência de Presenças (OCR + USP + Fuzzy)

App Streamlit gratuito para conferir presença a partir de **fotos/prints** de listas manuscritas.  
Ele cruza com um `.xlsx` de **inscritos** (colunas **Nome** e **USP**) e retorna:

- ✅ **Presentes** (match por **USP** exato; se USP faltou, tenta **fuzzy** no **Nome**)
- ❌ **Faltantes** (inscritos sem match)
- ⚠️ **Detectados não-inscritos** (aparecem na lista, mas não estão no `.xlsx`)

## 📦 Como usar (local)
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
