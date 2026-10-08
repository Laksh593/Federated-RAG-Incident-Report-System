# Federated RAG Incident Report Generator

An automated intelligence system combining **Clustered Federated Learning (CFL)** for video anomaly detection with a **Hybrid Retrieval-Augmented Generation (RAG)** pipeline to generate formal, policy-compliant security incident reports.

---

## Architectural Overview

```
React 18 + Vite Dashboard (Frontend)
        ↓  (HTTP REST / Multipart Form Data)
FastAPI Backend (api.py)
        ↓
PyTorch C3D + MIL Anomaly Classifier (CFL-Trained)
        ↓  [Binary: Normal vs Anomaly]
Hybrid Retrieval Engine (rag/retriever.py)
        ├── Dense Vector Search (ChromaDB + all-MiniLM-L6-v2)
        └── Sparse Keyword Search (BM25Okapi)
        ↓
Reciprocal Rank Fusion (RRF, k=60)
        ↓
Cross-Encoder Re-Ranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
        ↓  [Top-k Facility Standard Operating Procedures]
Local Ollama Generation Engine
        ├── LLaVA (Multimodal Vision: Keyframe Analysis)
        └── Llama 3 (Text Drafting: Incident Report Formulation)
        ↓
Formal Security Incident Report (Markdown)
```

> **Important Architectural Clarification:**
> - **Federated Learning:** Used exclusively during the *offline training* phase to train the 3D CNN (C3D) anomaly classification model across simulated camera clusters using the UCF-Crime dataset.
> - **RAG Layer:** The current retrieval and generation pipeline is *centralized* (local ChromaDB vector store + local BM25 index + local Ollama LLM server).
> - **Model Semantics:** The computer vision model performs binary anomaly classification (Normal vs Anomaly). When an anomaly is confirmed, the system retains the user's selected incident category (e.g., Fighting, Vandalism, Trespassing) for policy retrieval and report generation.

---

## Features

- **Video Evidence Ingestion:** Accepts video footage (`.mp4`, `.webm`, `.mov`), with automatic H.264 transcoding via FFmpeg.
- **Deep Learning Anomaly Detection:** Extracts uniform video clips into a C3D convolutional neural network with temporal attention to verify anomalies.
- **SOP Policy Grounding:** Retrieves facility Standard Operating Procedures (SOPs) using dense semantic search (ChromaDB) and sparse keyword search (BM25).
- **High-Precision Re-Ranking:** Uses a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to re-rank candidate policies before LLM prompting.
- **Multimodal Visual Verification:** Uses Ollama LLaVA to inspect keyframes directly from video evidence to minimize hallucinations.
- **Structured Incident Reporting:** Produces structured reports with Incident Summaries, Verified Locations & Timestamps, Policy Citations, and Recommended Actions.
- **Non-Blocking Architecture:** Offloads deep learning inference, frame extraction, and video transcoding to worker threads via `asyncio.to_thread()`.

---

## Tech Stack

- **Backend:** Python 3.10+, FastAPI, Uvicorn, Pydantic, OpenCV (`opencv-python`), imageio-ffmpeg
- **Machine Learning:** PyTorch, Sentence-Transformers, HuggingFace Transformers
- **Retrieval & Storage:** ChromaDB (HNSW / Cosine), rank-bm25 (BM25Okapi), LangChain Text Splitters
- **LLM / Vision-Language Engine:** Ollama (LLaVA for vision, Llama 3 for text)
- **Frontend:** React 18, Vite, Tailwind CSS v4, React-Markdown

---

## Project Structure

```
FL_RAG/
├── .env.example                     # Environment template for local Ollama settings
├── api.py                           # FastAPI application & REST endpoints
├── requirements.txt                 # Backend Python dependencies
├── README.md                        # Documentation
│
├── assets/                          # Sample media files
│   └── sample_videos/               # Sample crime & test video clips
│
├── data/
│   ├── knowledge_base/              # Standard Operating Procedure (SOP) text documents
│   └── vector_store/                # Local ChromaDB and BM25 persistent index files
│
├── frontend/                        # React + Vite web dashboard
│   ├── .env.example                 # Frontend API configuration template
│   ├── package.json                 # Node dependencies
│   ├── index.html                   # HTML entry point
│   ├── vite.config.js               # Vite build configuration
│   └── src/
│       ├── App.jsx                  # Main dashboard component
│       ├── index.css                # Dark-mode glassmorphic styling
│       └── main.jsx                 # React root
│
├── model/                           # Neural network & federated learning codebase
│   ├── config.py                    # Model & training hyperparameter dataclasses
│   ├── model.py                     # C3D Feature Extractor + MIL Classifier architecture
│   ├── inference.py                 # Runtime frame extraction & anomaly detection
│   ├── ucf_dataset.py               # UCF-Crime dataset loader & non-IID client partitioner
│   ├── cfl_train.py                 # Clustered Federated Learning training pipeline
│   ├── run_cfl.py                   # Self-contained training script
│   ├── visualize_cfl.py             # Performance visualization utilities
│   ├── checkpoints/                 # Model checkpoints directory (see Note below)
│   ├── metrics/                     # CFL training metric logs (JSON)
│   └── visualizations/              # Genuine CFL training curves & metric plots
│
├── scripts/
│   └── generate_policies.py         # Utility script to generate synthetic SOP policies
│
└── temp_uploads/                    # Temporary staging directory for video uploads
```

---

## Model Checkpoint Notice

The trained PyTorch model checkpoints (`~112.5 MB` each) exceed GitHub's 100 MB single-file limit and are excluded from version control via `.gitignore`:
- `model/checkpoints/cfl_latest.pt`
- `model/checkpoints/cfl_round_1.pt`
- `model/checkpoints/cfl_round_5.pt`

### Setting Up the Checkpoint for Local Inference:
To run inference with the trained model, ensure `cfl_latest.pt` is placed in the checkpoints directory:
```
model/checkpoints/cfl_latest.pt
```
If this file is missing, the backend will halt startup with an informative `FileNotFoundError` rather than silently executing with random, untrained weights.

---

## Prerequisites

- **Python:** 3.10 or 3.11
- **Node.js:** v18.0.0 or higher
- **Ollama:** Installed and accessible locally ([Download Ollama](https://ollama.ai/))

---

## Getting Started

### 1. Ollama Setup
Start the local Ollama daemon and pull the required vision and text models:
```bash
# Start Ollama service
ollama serve

# Pull multimodal vision model (for video keyframe understanding)
ollama pull llava

# Pull text model (for standard text reports)
ollama pull llama3
```

### 2. Backend Setup
Create and activate a virtual environment, then install dependencies:

```bash
# In the project root:
python -m venv .venv

# Activate on Windows PowerShell:
Set-ExecutionPolicy Unrestricted -Scope Process
.\.venv\Scripts\Activate

# Activate on Linux / macOS:
source .venv/bin/activate

# Install requirements:
pip install -r requirements.txt
```

Set up your environment configuration:
```bash
cp .env.example .env
```

Start the FastAPI backend server:
```bash
python api.py
```
*The server will start on `http://localhost:8000`. You should see `[OK] Successfully loaded trained CFL model weights`.*

### 3. Frontend Setup
In a new terminal window, navigate to the `frontend/` directory:

```bash
cd frontend

# Set up local environment config:
cp .env.example .env

# Install dependencies:
npm install

# Start Vite development server:
npm run dev
```
*Open `http://localhost:5173` in your browser.*

---

## Environment Variables

### Backend (`.env`)
| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL for the local Ollama REST API |
| `OLLAMA_TEXT_MODEL` | `llama3` | Model used for text-only incident reporting |
| `OLLAMA_VISION_MODEL` | `llava` | Multimodal model used for video keyframe analysis |
| `UCF_CRIME_DATASET_PATH`| *Empty* | Optional path to local UCF-Crime dataset for offline CFL training |

### Frontend (`frontend/.env`)
| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Target FastAPI backend URL |

---

## API Endpoints

| Method | Endpoint | Payload | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | None | API health status probe |
| `POST` | `/api/generate` | JSON (`AnomalyEvent`) | Generates a text-based incident report from metadata |
| `POST` | `/api/generate-video` | `multipart/form-data` | Uploads video evidence, runs C3D inference, extracts keyframes, and generates multimodal report |
| `POST` | `/api/refresh-kb` | None | Re-indexes documents in `data/knowledge_base/` idempotently |

---

## Offline Model Training (Clustered Federated Learning)

To reproduce or run the Clustered Federated Learning (CFL) training pipeline:
1. Download the UCF-Crime dataset with extracted PNG frames.
2. Set the path in your `.env`:
   ```bash
   UCF_CRIME_DATASET_PATH=/path/to/UCF-Crime
   ```
3. Run the training script:
   ```bash
   python model/run_cfl.py
   ```
4. Generate training metric plots:
   ```bash
   python model/visualize_cfl.py
   ```
Checkpoints will be saved automatically to `model/checkpoints/`.

---

## Known Limitations

- **Binary Anomaly Classification:** The C3D neural network distinguishes between Normal activity and Anomalies. Fine-grained incident categorization relies on user selection and LLM visual interpretation of keyframes.
- **Hardware Requirement for Video LLM:** Running `llava` locally via Ollama requires at least 8 GB of VRAM (or adequate unified system RAM on Apple Silicon / CPU).
- **Centralized RAG Retrieval:** Knowledge base queries are executed against a local ChromaDB instance rather than a decentralized federated vector store.

---

## Future Improvements

- Fully decentralized Federated RAG with privacy-preserving retrieval (PIR) across edge camera nodes.
- Temporal bounding box annotations for detected anomalies directly in video playback.
- Dynamic keyframe selection based on peak attention weights from the MIL classifier.
