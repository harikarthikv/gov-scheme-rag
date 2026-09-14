# Government scheme PDF assistant

This project searches text extracted from PDF files in `data/text_data` and sends the most relevant chunks to a DeepSeek-compatible chat model. It provides a command-line interface and a Streamlit interface.

It is a learning project, not an authoritative source of government-scheme information. Check important information against the original document and an official source.

## Setup

Use Python 3.10 or later. Create a virtual environment, activate it, and install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set a valid key:

```text
DEEPSEEK_API_KEY=your_key_here
```

The moved `shrijayan/gov_myscheme` dataset is stored locally in `data/text_data`. The application uses this relative project path; it does not read from the Hugging Face cache at runtime.

## Use

Build the index explicitly after adding, removing, or changing PDFs:

```powershell
python build_vectordb.py
```

Then use either interface:

```powershell
python app.py
streamlit run app_streamlit.py
```

The CLI and Streamlit app load the configured collection from `chroma_db`. If it is missing, they build it from the local PDFs. The build script does not replace an existing collection.

## Project structure

```text
app.py                         Command-line interface
app_streamlit.py               Streamlit chat interface
build_vectordb.py              Explicit index builder
src/                           Application modules
test/gold_standard_dataset.py  Evaluation fixture
run_evaluation.py              API-backed evaluation script
```

## Workflow

```text
Local PDFs -> page text -> overlapping chunks -> local embeddings -> Chroma index
Question -> Chroma similarity search -> top matching chunks -> chat model -> answer and source metadata
```

The chunk size is 1,000 characters with 200-character overlap. The retriever asks Chroma for up to three results and keeps scores at or above 0.3. These are configuration values, not quality guarantees.

## Evaluation

`run_evaluation.py` runs ten manually defined cases from `test/gold_standard_dataset.py`. It makes model API calls and uses another model call to grade answers, so it requires `DEEPSEEK_API_KEY`, network access, and can incur provider charges. Run it with `python run_evaluation.py`.

No evaluation results are committed as project claims. See [limitations](docs/limitations.md) before interpreting output.

## Limitations

- PDF extraction skips files that raise an extraction error; it does not use OCR.
- Retrieval and generated answers can be incomplete, irrelevant, or incorrect.
- Source citations are requested from the model but are not verified after generation.
- The relevance scores come from Chroma's search API; they are not calibrated confidence values.
- The current code has no automated unit-test suite or lint configuration.

More detail is in [docs/architecture.md](docs/architecture.md) and [docs/limitations.md](docs/limitations.md).
