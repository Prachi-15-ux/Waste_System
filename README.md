# AI Waste Segregation System

A lightweight, visually appealing, Deep Learning-based web application built with Flask and TensorFlow (MobileNetV2) to classify waste images and provide recycling suggestions. 

## Features
- **Instant Classification**: Upload an image to detect if it's Organic, Recyclable, Hazardous, or Non-recyclable.
- **Pre-trained Model**: Uses `MobileNetV2` with ImageNet weights, requiring ZERO manual training or dataset management.
- **Modern UI**: Clean, responsive, glassmorphism design.
- **Recycling Suggestions**: Instant advice on how to properly dispose of the detected waste.

## How to Run Locally

1. **Install Dependencies**
   Make sure you are in the `Waste_System` folder, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python app.py
   ```
   *Note: On the very first run, it might take a few seconds to download the lightweight MobileNetV2 weights (approx 14MB).*
   
   The browser will automatically open to `http://127.0.0.1:5000/`.

## Deployment Guide (Render / Railway)

This app is ready to be deployed to free hosting platforms.

### Deploying on Render.com
1. Create a new GitHub repository and push this `Waste_System` folder (or push it as the root of the repo).
2. Go to [Render](https://render.com) and create a new **Web Service**.
3. Connect your GitHub repository.
4. Set the following details:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Click **Create Web Service**. Render will install the dependencies and launch the app.

*(Note: Free tiers on Render may put the app to sleep after inactivity, leading to a 50s delay on the next initial load).*

## Project Structure
```text
Waste_System/
│── app.py                 # Main Flask server and prediction logic
│── requirements.txt       # Project dependencies
│── README.md              # Documentation
│── static/                # Static assets
│   │── style.css          # UI styling
│   │── script.js          # Frontend logic
│── templates/             # HTML templates
│   │── index.html         # Main web interface
```
