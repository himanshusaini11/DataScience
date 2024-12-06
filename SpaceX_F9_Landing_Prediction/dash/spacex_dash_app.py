# Import required libraries
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px

import flask
import os

# Initialize Flask server and Dash app
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

# Read the airline data into pandas dataframe
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                dcc.Dropdown(id='site-dropdown',
                                               options=[{'label': 'All Sites', 'value': 'ALL'},
                                                        {'label': 'CCAFS LC-40', 'value': 'CCAFS LC-40'},
                                                        {'label': 'VAFB SLC-4E', 'value': 'VAFB SLC-4E'},
                                                        {'label': 'KSC LC-39A', 'value': 'KSC LC-39A'},
                                                        {'label': 'CCAFS SLC-40', 'value': 'CCAFS SLC-40'},
                                                       ],
                                                value='ALL',
                                                placeholder="CCAFS LC-40",
                                                searchable=True
                                            ),
                                html.Br(),

                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                dcc.RangeSlider(id='payload-slider',
                                                min=0, max=10000, step=1000,
                                                marks={0: '0', 2500: '2500', 5000: '5000', 7500: '7500', 10000: '10000'},
                                                value=[0, 10000]
                                               ),

                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),
                                ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
# Function decorator to specify function input and output
@app.callback(Output(component_id='success-pie-chart', component_property='figure'),
            Input(component_id='site-dropdown', component_property='value'))
def get_pie_chart(entered_site):
    filtered_df = spacex_df.groupby('Launch Site')['class'].mean().reset_index()
    if entered_site == 'ALL':
        fig = px.pie(filtered_df, values='class',
                        names='Launch Site',
                        title='Total Success Launches By ALL Sites'
                    )
        return fig
    else:
        # return the outcomes piechart for a selected site
        filtered_df = spacex_df[spacex_df['Launch Site'] == entered_site][['Launch Site', 'class']].value_counts().reset_index()
        title_site = f"Total Success Launches for {filtered_df['Launch Site'][0]}."
        fig = px.pie(filtered_df, values='count', names='class', title=title_site)
        
        return fig

# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(Output(component_id='success-payload-scatter-chart', component_property='figure'),
              Input(component_id='site-dropdown', component_property='value'),
              Input(component_id='payload-slider', component_property='value')
             )
def get_scatter_plot(entered_site, payload_value):
    low, high = payload_value
    filtered_df = spacex_df[(spacex_df['Payload Mass (kg)'] >= low) & (spacex_df['Payload Mass (kg)'] <= high)]
    if entered_site == "ALL":
        fig = px.scatter(filtered_df, x='Payload Mass (kg)', y='class', color='Booster Version Category', title='Correlation Between Payload and Success for All Sites')
        return fig
    else:
        filtered_df1 = filtered_df[filtered_df['Launch Site'] == entered_site][['Payload Mass (kg)', 'class', 'Booster Version Category']]
        title_site_scatter = f"Total Success Launches for {entered_site}."
        fig = px.scatter(filtered_df1, x='Payload Mass (kg)', y='class', color='Booster Version Category', title=title_site_scatter)
        return fig


# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run_server(debug=False, host='0.0.0.0', port=port)
