# 🍼 Baby Food RAG Assistant

A Retrieval-Augmented Generation (RAG) system for baby food recommendations using FastAPI, hybrid search, and natural language processing.

## 🚀 Features

- **Hybrid Search**: Combines BM25 keyword search with vector similarity (TF-IDF or Sentence Transformers)
- **Natural Language Q&A**: Ask questions in plain English about baby foods
- **Evidence-Based Answers**: Provides confidence scores and USDA citations
- **Clean Web UI**: Simple interface for querying the system
- **Modular Architecture**: Easy to extend with new features like RAGAS evaluation

## 📊 Dataset

The system includes nutritional data for 10 baby foods with:
- Nutritional information (calories, protein, fiber, iron, vitamins)
- Food categories and descriptions  
- USDA reference URLs
- Pediatric feeding notes

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd babyfood-rag
```

2. **Create virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Quick Start

1. **Run the application**:
```bash
python main.py
```

2. **Open your browser** and go to: `http://localhost:8000`

3. **Try example questions**:
   - "Which foods are high in iron?"
   - "What foods help with immunity?"
   - "Best first foods for babies"
   - "High protein baby foods"

## 📡 API Usage

### Ask Endpoint

**POST** `/api/ask`

```json
{
  "question": "Which foods are high in iron?",
  "top_k": 3
}
```

**Response**:
```json
{
  "answer": "For iron content, Iron-Fortified Rice Cereal contains 45.0mg iron per 100g...",
  "citations": [
    {
      "food_name": "Iron-Fortified Rice Cereal",
      "usda_url": "https://fdc.nal.usda.gov/fdc-app.html#/food-details/174852",
      "relevance_score": 0.892
    }
  ],
  "confidence": "High",
  "retrieved_foods": [...]
}
```

## 🏗️ Architecture

```
├── main.py                 # FastAPI application entry point
├── app/
│   ├── models.py          # Pydantic data models
│   ├── data_loader.py     # CSV data loading utilities
│   ├── retriever.py       # Advanced retriever (Sentence Transformers)
│   ├── simple_retriever.py # Fallback retriever (TF-IDF)
│   └── routers/
│       └── ask.py         # Q&A API endpoint
├── data/
│   └── foods.csv          # Baby food dataset
├── static/
│   ├── index.html         # Web interface
│   └── style.css          # UI styling
└── requirements.txt       # Python dependencies
```

## 🔧 Configuration

The system automatically falls back to a simpler TF-IDF based retriever if Sentence Transformers is not available. To use the advanced retriever:

```bash
pip install sentence-transformers==2.7.0 transformers==4.36.0 torch==2.1.0
```

## 🧪 Development

### Adding New Foods

1. Edit `data/foods.csv` with new entries
2. Restart the application to reload the data

### Extending Retrieval

The retriever classes are modular and can be extended with:
- New embedding models
- Additional search algorithms  
- Custom scoring functions

### Future Enhancements

- RAGAS evaluation framework
- More sophisticated answer generation
- Multi-language support
- Additional data sources

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This system is for informational purposes only. Always consult with a pediatrician before making feeding decisions for your baby.

## 📞 Support

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].
