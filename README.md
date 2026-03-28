# GEOSPATIAL ANALYSIS OF SOIL TYPES USING GENERATIVE AI AND DEEP LEARNING

A Flask web application that classifies soil type from images using a pre-trained VGG-based model and recommends suitable crops. Built for academic/prototype usage in soil classification and agriculture recommendations.

## Project Structure

- `app.py` – Flask app with upload + prediction endpoints, model loading, and soil-to-crop mapping.
- `model_vgg.json` + `model_vgg.weights.h5` – saved model architecture and weights.
- `templates/index.html` – upload and UI page.
- `templates/result.html` – prediction outcome page.
- `static/uploads` – uploaded and sample soil images.
- `static/videos` – background video asset.
- `train/` and `test/` folders – image dataset split by soil type.

## Soil Classes
- Black Soil
- Cinder Soil
- Laterite Soil
- Peat Soil
- Yellow Soil

## Crop Recommendations
Each soil type maps to a set of vegetables and fruits in `SOIL_CROPS` (see `app.py`).

## Quick Start

1. Install dependencies:
   ```bash
   pip install flask tensorflow numpy pillow werkzeug
   ```

2. Run app:
   ```bash
   python app.py
   ```

3. Visit:
   - `http://127.0.0.1:5000`

4. Upload soil image and view predicted soil type, confidence, and recommended crops.

## Notes
- The model expects images resized to `150x150` and normalized to `[0,1]`.
- Ensure `'static/uploads'` exists (app creates it automatically).
- Replace or retrain model files if you need improved accuracy.

## GitHub
Repo: https://github.com/gulammuzammilkhan-del/GEOSPATIAL-ANALYSIS-OF-SOIL-TYPES-UTILIZING-GENERATIVE-AI-AND-DEEP-LEARNING-FOR-HIG
