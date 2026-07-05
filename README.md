# AI-Driven Emotion Detection & Personalized Learning Support Platform

## Project structure
```
EmotionDetectionProject/
├── app.py                     # Streamlit app (run this)
├── train_bilstm.py            # Trains the BiLSTM model
├── train_bert.py              # Fine-tunes DistilBERT
├── requirements.txt
├── dataset/
│   ├── generate_dataset.py    # Creates emotions.csv (synthetic data)
│   └── emotions.csv           # Created after running generate_dataset.py
├── models/
│   ├── bilstm_model.h5        # Created after train_bilstm.py
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   ├── bert_model/            # Created after train_bert.py
│   └── bert_label_encoder.pkl
├── utils/
│   ├── preprocess.py
│   ├── predict.py
│   └── gemini.py
├── logs/
│   └── interactions.csv       # Auto-created by the app
└── images/
```

## Setup steps (run in order, from the project root in VS Code's terminal)

```bash
# 1. Install dependencies (you already did this)
pip install streamlit pandas numpy matplotlib scikit-learn tensorflow transformers torch google-generativeai

# 2. Generate the training dataset (synthetic — replace dataset/emotions.csv
#    with your own real CSV of columns text,emotion if you have one, and
#    skip this step)
python dataset/generate_dataset.py

# 3. Train the BiLSTM model
python train_bilstm.py

# 4. Fine-tune BERT (requires internet to download distilbert-base-uncased
#    the first time; this step can take several minutes on CPU)
python train_bert.py

# 5. Run the app
streamlit run app.py
```

The app works even if you skip step 3 or 4 — it will show a warning for
whichever model isn't trained yet and still work using the other model
plus keyword-based rule enhancement.

## Gemini API key
1. Get a free key at https://aistudio.google.com/apikey
2. Paste it into the "Gemini API Key" field in the app's sidebar, **or**
   set it as an environment variable before launching:
   ```bash
   # Windows (PowerShell)
   $env:GEMINI_API_KEY="your_key_here"
   # macOS/Linux
   export GEMINI_API_KEY="your_key_here"
   ```

## Note on the Gemini package
`google-generativeai` (the package you installed) is now in Google's
"limited maintenance" mode — it still works, but Google recommends new
projects use the newer `google-genai` package going forward
(`pip install google-genai`). `utils/gemini.py` is written for the
package you already have installed so nothing breaks; if you want to
migrate later, the new usage pattern is:
```python
from google import genai
client = genai.Client(api_key="YOUR_KEY")
response = client.models.generate_content(model="gemini-2.5-flash", contents="Hello")
print(response.text)
```

## How it works
- **preprocess.py** — cleans text and computes keyword-based emotion scores.
- **train_bilstm.py / train_bert.py** — train and save the two classifiers.
- **predict.py** — loads both models (if present), averages their
  probabilities, blends in the keyword score, and detects "mixed
  emotions" (any emotion within 0.15 of the top probability).
- **gemini.py** — sends the input text + detected emotion to Gemini and
  returns a short, empathetic, tip-filled response. Fails gracefully
  with a fallback tip if the API is unreachable.
- **app.py** — the Streamlit UI: an Assistant tab (enter text → see
  emotion breakdown, model comparison charts, AI guidance) and an
  Analytics Dashboard tab (trends from logs/interactions.csv).

## Troubleshooting
- **`ModuleNotFoundError`** → re-run the pip install command above inside
  the same virtual environment / interpreter VS Code is using.
- **BERT training is slow** → normal on CPU; reduce `num_train_epochs`
  in `train_bert.py` or reduce dataset size in `generate_dataset.py`.
- **Gemini errors in the app** → check your API key and internet
  connection; the app will still function with a fallback message.
