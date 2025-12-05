# Streamlit Interface Guide

## Quick Start with Streamlit

Streamlit provides a simpler, pure-Python alternative to the FastAPI + HTML/JS frontend. It's ideal for academic projects and rapid prototyping.

### Why Streamlit?

- ✅ **Pure Python**: No HTML/CSS/JavaScript needed
- ✅ **Built-in UI**: Pre-made components for data science
- ✅ **Interactive**: Automatic reactivity and state management
- ✅ **Fast Development**: Build UI in minutes, not hours
- ✅ **Academic Friendly**: Perfect for presentations and demos

### Installation

The Streamlit dependency is already added to `requirements.txt`. Install all dependencies:

```bash
pip install -r requirements.txt
```

Or install Streamlit separately:

```bash
pip install streamlit
```

### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### Features

The Streamlit interface includes all the same functionality as the FastAPI frontend:

1. **📤 Data Upload**
   - Upload CSV files
   - Load demo data
   - View data preview and statistics

2. **🧹 Preprocessing**
   - One-click data cleaning
   - Before/after comparison
   - Download cleaned data

3. **📊 Clustering**
   - K-means and DBSCAN algorithms
   - Interactive parameter tuning
   - Visual cluster maps
   - Quality metrics (silhouette score)

4. **🔮 Prediction**
   - **Gradient Boosting** for maximum accuracy
   - **Poisson Regression** for interpretable results
   - **Compare Both Models** side-by-side
   - Feature importance visualization
   - Interpretable coefficients for Poisson

5. **🗺️ Visualization**
   - Interactive heatmaps
   - Statistical charts
   - Embedded map viewer

### Comparison: FastAPI vs Streamlit

| Feature | FastAPI + Frontend | Streamlit |
|---------|-------------------|-----------|
| **Setup** | Complex (2 components) | Simple (1 file) |
| **Language** | Python + HTML/JS/CSS | Pure Python |
| **Development Time** | Hours | Minutes |
| **Customization** | Full control | Limited by components |
| **Production** | Better for APIs | Better for demos |
| **Academic Use** | Good | **Excellent** |
| **Learning Curve** | Steep | Gentle |

### For Your Project

**Recommendation**: Use **Streamlit** for:
- Academic presentations
- Adviser demonstrations
- Quick iterations and testing
- Thesis defense demos

Use **FastAPI** for:
- Production deployment
- API services
- Custom UI requirements
- External integrations

### Key Advantages for Academic Projects

1. **Model Comparison Built-in**: Easily compare Gradient Boosting vs Poisson Regression
2. **Interpretable Results**: Display Poisson coefficients with explanations
3. **Interactive Exploration**: Advisers can try different parameters live
4. **Clean Presentation**: Professional look without CSS/HTML knowledge
5. **Easy Screenshots**: Perfect for thesis documentation

### Tips

- State is preserved in `st.session_state` - no database needed
- All parameters are interactive - change and see results immediately
- Built-in caching with `@st.cache_data` for performance
- Automatic error handling with nice error messages
- Download buttons for exporting results

### Next Steps

1. Run `streamlit run streamlit_app.py`
2. Upload your data or use demo data
3. Walk through the pipeline step-by-step
4. Compare both prediction models
5. Generate visualizations
6. Take screenshots for your thesis!

---

**Both interfaces work with the same backend code**, so you can use whichever fits your needs better. For thesis defense and academic presentations, Streamlit is highly recommended.
