SafeZone AI: Wildfire Risk and Forecast Dashboard

SafeZone AI is an educational geospatial machine learning project designed for wildfire situational awareness. It integrates satellite fire detections, weather data, and spatial analysis to estimate wildfire risk and forecast potential next-day fire activity.

The system combines:
- NASA FIRMS wildfire detections  
- NOAA weather enrichment (wind, alerts)  
- Spatial binning for regional aggregation  
- Rule-based risk scoring  
- A machine learning model for next-day fire prediction  

It also generates decision-support outputs such as advisory levels and review flags to highlight elevated-risk areas.

Disclaimer:
This project is intended for educational and demonstration purposes only. It is not designed for operational emergency response or real-world evacuation decision-making.

------------------------------------------------------------

Project Overview

SafeZone AI answers three key questions:

1. Where are fires currently occurring?  
   Using NASA FIRMS detections.

2. Where is wildfire risk elevated?  
   Based on spatial density, fire intensity, and weather conditions.

3. Where might fire activity occur next?  
   Predicted using a machine learning model.

The outputs are visualized in an ArcGIS Dashboard combining:
- Real fire detections  
- Risk levels (low, medium, high)  
- Predicted next-day activity  
- Spatial clustering and risk zones  
- Wind conditions  

------------------------------------------------------------

System Architecture

Data Sources:
- NASA FIRMS: Satellite fire detections  
- NOAA Weather API: Wind speed and alerts  
- ArcGIS Historical Data: Extended training dataset  

Feature Engineering:
- fire_count: Number of nearby fire detections (spatial density)  
- avg_brightness, max_brightness: Fire intensity  
- wind_speed: Environmental spread factor  

Lag features:
- prev_day_fire_count  
- rolling_2day_fire_count  
- rolling_3day_fire_count  

Machine Learning Model:
- Model: RandomForestClassifier  
- Task: Predict next-day fire activity (binary classification)

Outputs:
- predicted_fire_next_day (0 or 1)  
- probability (likelihood of fire activity)  

------------------------------------------------------------

Risk and Advisory Logic

Risk Levels:
- Low  
- Medium  
- High  

Advisory Levels:
- Monitor  
- Prepare  
- Urgent Review  

Review Flag:
- Indicates areas requiring closer attention  

------------------------------------------------------------

Visualization (ArcGIS Dashboard)

The system outputs are visualized in an interactive dashboard.

Map Layers:
- Actual fire detections (ground truth)  
- Risk zones (model and density)  
- Spatial clustering (fire density)  
- Wind direction and speed  

Dashboard Components:
- Key indicators (risk counts, predictions)  
- Interactive map  
- Priority risk list  
- Advisory distribution chart  

Users can click on high-risk areas to zoom and inspect conditions.

------------------------------------------------------------

Project Structure

src/
  data_fetch.py
  data_fetch_arcgis.py
  clean.py
  features.py
  weather.py
  model.py
  dashboard.py
  main.py

output/
  predictions.csv
  dashboard_points.csv
  model_metrics.json

------------------------------------------------------------

How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Add API key to .env:
   NASA_API_KEY=your_key_here

3. Run the pipeline:
   python src/main.py

4. Generate dashboard dataset:
   python src/dashboard.py

5. Upload dashboard_points.csv to ArcGIS

------------------------------------------------------------

Example Outputs

- Wildfire risk dashboard  
- Advisory zones and clustering  
- Model performance metrics (model_metrics.json)  

------------------------------------------------------------

Future Improvements

- Add more historical days for stronger lag features  
- Add richer weather features (humidity, temperature)  
- Improve spatial resolution or binning strategy  
- Include terrain and vegetation features  
- Compare multiple baseline models  
- Expand dashboard interactivity and visualization  
- Evaluate class imbalance more deeply  
- Backtest on larger historical windows  

------------------------------------------------------------

Key Takeaways

SafeZone AI demonstrates how to:
- Combine machine learning with geospatial data  
- Move from reactive detection to predictive awareness  
- Build a decision-support visualization system  
- Integrate data pipelines with GIS dashboards  

------------------------------------------------------------

Author

Mario Tafoya  
California State University, Los Angeles  

------------------------------------------------------------

License

This project is licensed under the terms of the LICENSE file.
