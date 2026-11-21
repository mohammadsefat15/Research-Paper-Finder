# AI-Powered Research Paper Finder and Summarizer

A research-support tool that finds relevant arXiv AI/ML papers from a natural-language query and summarizes their abstracts.

- **Semantic search:** Sentence-BERT embeddings (`all-MiniLM-L6-v2`) with a FAISS index
- **Baseline:** TF-IDF keyword retrieval (scikit-learn), compared with semantic search
- **Summarization:** pretrained `sshleifer/distilbart-cnn-12-6`, evaluated with ROUGE on SciTLDR and by manual checks
- **App:** Streamlit interface with a switch between semantic and keyword search

> Summaries and search results are generated automatically and need human verification.

## Datasets

| Dataset | Source | Used for |
|---|---|---|
| arXiv metadata snapshot (about 25.6 thousand unique papers from cs.CL, cs.IR, cs.LG, cs.HC, cs.CV, up to April 2020) | https://www.kaggle.com/datasets/Cornell-University/arxiv (also `gs://arxiv-dataset/metadata-v5/arxiv-metadata-oai.json`) | Paper search |
| SciTLDR (SciTLDR-A test set, 618 papers) | https://github.com/allenai/SciTLDR and https://huggingface.co/datasets/allenai/scitldr | Summarization evaluation |
| Semantic Scholar Academic Graph API | https://api.semanticscholar.org/api-docs/ | Planned source of current paper records (not used in the final system) |

The full processed files are large, so they are not stored in this repository. `data/papers_sample_1000.csv` is a small sample of the processed paper table. Full processed table: https://drive.google.com/file/d/1-HJWNME4-sV6g6aRI6N_aXgK6djmOs5T/view?usp=drive_link

## Results

Retrieval: 500 random papers, query = the paper title, correct answer = that paper's abstract.

| Method | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| TF-IDF | 0.554 | 0.754 | 0.806 | 0.642 |
| DistilBERT NLI (model of the original project) | 0.184 | 0.314 | 0.376 | 0.247 |
| MiniLM (this project) | 0.704 | 0.876 | 0.934 | 0.784 |

Summarization: 100 SciTLDR-A test papers, compared with the first reference TLDR.

| ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|
| 0.267 | 0.098 | 0.195 |

## Screenshots

Add your screenshots to the `results/` folder and show them here.

## How to run

1. Open `research_paper_finder.ipynb` in Google Colab and run the cells from top to bottom.
2. For the app, put `app.py`, `papers_clean.csv` and `embeddings_minilm.npy` in one folder, then run:

```
pip install -r requirements.txt
set PAPER_DATA_DIR=path\to\that\folder\   (Windows)  or  export PAPER_DATA_DIR=path/to/that/folder/   (Linux/Mac)
streamlit run app.py
```

## Limitations

- The paper collection ends around April 2020.
- The retrieval test uses paper titles as queries, which share many words with their own abstracts. This helps TF-IDF.
- The summarizer was trained on news articles. It often copies sentences from the abstract and can end mid-sentence.
- ROUGE is low because SciTLDR references are one sentence and the model writes longer text. ROUGE was computed on 100 papers against one reference each.

## Credits

- Based on **Arxiv-Neural-Search** by karndeb (GPL-3.0): https://github.com/karndeb/Arxiv-Neural-Search . This project reuses its idea (arXiv metadata, sentence embeddings and FAISS search with a Streamlit app) and its data preparation approach. New in this project: TF-IDF baseline, comparison of three embedding setups, a retrieval evaluation, summarization with ROUGE evaluation and manual checks, and the new Streamlit interface.
- SciTLDR (Allen Institute for AI), Sentence-BERT, FAISS, Hugging Face Transformers, Streamlit.

## License

GPL-3.0, because this project builds on a GPL-3.0 project.
