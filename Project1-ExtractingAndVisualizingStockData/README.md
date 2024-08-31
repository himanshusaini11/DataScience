# Web Scraping and Data Analysis Project

This project is a hands-on exercise where we perform web scraping using Python's BeautifulSoup library to extract data from a web page and analyze it using Pandas. The data extracted is focused on Tesla's quarterly revenue, and we follow a structured approach to parse, clean, and store the data in a DataFrame.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Data Cleaning](#data-cleaning)
- [Contact Information](#contact-information)

## Project Overview

This project demonstrates the use of web scraping techniques to collect financial data from a specific webpage. The data is then processed and stored in a structured format for further analysis. The main focus is on Tesla's quarterly revenue data, which is extracted, cleaned, and stored in a Pandas DataFrame.

## Features

- **Web Scraping**: Using BeautifulSoup to scrape data from a table on a webpage.
- **Data Cleaning**: Processing and cleaning the scraped data to remove unwanted characters.
- **Data Analysis**: Storing the cleaned data in a Pandas DataFrame for further analysis.

## Installation

To run this project, you need to have Python installed along with the necessary libraries.

### Requirements

- Python 3.x
- BeautifulSoup4
- Pandas
- Requests

### Installation Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/himanshusaini11/Project1-ExtractingAndVisualizingStockData.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Project1-ExtractingAndVisualizingStockData
    ```
3. Install the required Python packages:
    ```bash
    pip install <required-packages>
    ```

## Usage

1. Run the Jupyter Notebook provided to execute the web scraping and data analysis tasks.
2. The notebook walks through the entire process, from scraping the data to analyzing and displaying it.
3. You can modify the code to scrape different data or perform different types of analysis as needed.

## Code Structure

- **Web Scraping**: The code begins by loading the HTML content of the target webpage using the `requests` library. BeautifulSoup is then used to parse the HTML and extract relevant data from the tables.
- **DataFrame Construction**: The extracted data is cleaned and stored in a Pandas DataFrame, which is then used for analysis.

## Data Cleaning

During the data extraction process, the revenue data is cleaned by removing unnecessary characters like dollar signs and commas to ensure accurate numerical analysis.

## Contact Information

For questions or inquiries, please contact:

- **Himanshu Saini**
- Email: himanshusaini.rf@gmail.com
- GitHub: [himanshusaini11](https://github.com/himanshusaini11)
