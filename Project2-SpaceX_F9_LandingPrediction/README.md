# SpaceX Falcon 9 Landing Prediction Project

## Overview
This project aims to predict the reusability of the first stage of SpaceX Falcon 9 rockets using data analysis, visualization, and machine learning techniques. Reusability is a critical factor in reducing space travel costs and increasing mission efficiency.

The project was developed as part of the IBM Data Science Professional Certificate Capstone and includes comprehensive workflows, from data collection to deploying an interactive dashboard.

## Features
- **Data Collection:** Gathered data using SpaceX REST API and web scraping (Wikipedia).
- **Data Wrangling:** Cleaned, transformed, and prepared the dataset for analysis and modeling.
- **Exploratory Data Analysis (EDA):** Performed using SQL and visualization tools to uncover insights.
- **Interactive Visualizations:** Created with Folium and Plotly Dash.
- **Predictive Modeling:** Implemented machine learning models for classification tasks.
- **Interactive Dashboard:** Built with Plotly Dash and deployed on Azure.

## Technologies Used
- **Languages:** Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- **Data Collection:** SpaceX REST API, BeautifulSoup (web scraping)
- **Visualization Tools:** Plotly, Folium
- **Databases:** SQLite
- **Machine Learning:** Logistic Regression, SVM, Decision Tree, KNN
- **Deployment:** Azure App Service

## Notebooks and Resources
- [Data Collection (Web Scraping)](./notebooks/Part0_Webscraping_Falcon9_Data.ipynb)
- [Data Collection (API)](./notebooks/Part1_SpaceX_DataCollection_API.ipynb)
- [Data Wrangling](./notebooks/Part2_SpaceXDataWrangling.ipynb)
- [EDA with SQL](./notebooks/Part3_EDA_SQL.ipynb)
- [EDA with Visualization](./notebooks/Part4_EDA_DataVizualization.ipynb)
- [Interactive Map (Folium)](./notebooks/Part5_LaunchSiteLocations.ipynb)
- [Predictive Modeling](./notebooks/Part6_SpaceX_ML_Prediction.ipynb)
- [Interactive Dashboard (Dash)](./dash/spacex_dash_app.py)

## Dashboard
Explore the live dashboard on Azure: [SpaceX Falcon 9 Landing Prediction](https://spacex-capstone-ibm.azurewebsites.net/)

## Key Results
- **Launch Success Trends:** Significant improvements in success rates over time.
- **Impact of Payload and Booster Types:** Heavy payloads and specific booster types affect launch success.
- **Machine Learning Performance:** Decision Tree classifier achieved the highest accuracy (90%).

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/himanshusaini11/DataScience.git
2. Navigate to the project folder:
   ```bash
   cd Project2-SpaceX_F9_LandingPrediction/dash
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the dashboard locally:
   ```bash
   python spacex_dash_app.py

## License
This project is licensed under the [MIT License](https://github.com/himanshusaini11/DataScience/Project2-SpaceX_F9_LandingPrediction/LICENSE.md).

## Disclaimer
This project was the part of my learning program from Coursera IBM Professional Data Science Certification.

## Contact
For any queries or collaborations, feel free to reach out:

Name: Himanshu Saini
Email: himanshusaini.rf@gmail.com
LinkedIn: [LinkedIn](https://www.linkedin.com/in/sainihimanshu/)
