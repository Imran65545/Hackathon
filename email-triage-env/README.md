# 📬 Email Triage Environment (OpenEnv)

A real-world AI training and evaluation environment designed to test frontier language models (and autonomous agents) on their ability to act as corporate triage operators. 

This repository strictly complies with the **OpenEnv specification**. It operates via a robust Pydantic framework internally and is fully wrapped in FastAPI and Docker for massive scalability or continuous benchmarking.

---

## 🚀 The 3 Evaluation Constraints

Our `emails.json` data operates under three distinct OpenEnv evaluation contexts:

1. **Task 1 (Easy): Priority Sorting**  
   The agent must label an incoming email's urgency exactly as `urgent`, `normal`, or `low`. Correct strings award `1.0`.
2. **Task 2 (Medium): Full Categorization**  
   The agent must evaluate the priority AND correctly route the email to `HR`, `Sales`, `Tech`, `Billing`, or `Other`.
3. **Task 3 (Hard): Advanced Drafting**  
   The ultimate test: Assigning priority, categorizing precisely, AND drafting a single-line, highly professional, contextual reply back to the sender. The environment natively scores this text generation probabilistically based on length bounds, sentiment, and ground-truth knowledge matching.

*(Note: Agents are actively penalized dynamically via `tasks.py` for empty arrays, ignoring dependencies, or looping repeatedly with identical payloads).*

---

## 🛠️ Repository Architecture
- `env/models.py`: Immutable state definitions governing Actions, Observations, and Rewards.
- `env/environment.py`: Natively loads the synthetic datasets, advances the episodes sequentially, and manages context bounds.
- `env/tasks.py`: Advanced point-allocation matrices executing strict evaluation metrics.
- `app.py`: High-speed API server orchestrating `/reset`, `/step`, and `/state` workflows globally.
- `tests/test_env.py`: Broad-coverage validation suite mocking exact data edge-case scenarios.
- `baseline/inference.py`: Production-level benchmark script utilizing temperature `0.0` over GPT-3.5 to document reference metrics!

---

## 💻 Local Developer Usage

1. **Install Virtual Env**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/macOS
   .\venv\Scripts\Activate   # For Windows
   pip install -r requirements.txt
   ```

2. **Run Pytest Validations**:
   ```bash
   pytest tests/test_env.py
   ```

3. **Serve Environment APIs via Uvicorn**:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 7860
   ```

---

## 📊 Invoking the Agent Baseline

If you want to view the baseline metrics natively on your hardware:
```bash
# Export your token
export OPENAI_API_KEY="sk-..."

# Execute all identical tasks fully
python baseline/inference.py
```

---

## 🐳 Hugging Face Spaces (Deploying Core)

This codebase is natively written to support a 1-click **Hugging Face Docker Space**.

### Local Build Simulation
```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

### Hugging Face Deployment Sequence
1. Navigate to Hugging Face -> **Create New Space**
2. Input your nomenclature and choose **Docker** as your Space SDK -> **Blank**.
3. Obtain your new repository clone URL.
4. Push these files directly to main:
   ```bash
   git init
   git add .
   git commit -m "Initialize OpenEnv Environment"
   git push origin main
   ```
5. Hugging Face will immediately intercept `Dockerfile`, spawn your FastAPI server, and expose the testing endpoints!
