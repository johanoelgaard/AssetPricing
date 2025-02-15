# functions created for seminar in Asset Pricing and Financial Markets at the University of Copenhagen, Fall 2024
import requests
import pandas as pd
import os
import re
import gc
import orjson 
import plotly.graph_objects as go
from shapely.geometry import Polygon, Point
import ast
import numpy as np
from sklearn.decomposition import PCA


def dmi_api(base_url, period, parameter_id, api_key, limit=300000):
    """
    Function to fetch data from the DMI API for a given parameter and period.

    Args:
    - base_url: The base URL for the API endpoint
    - period: The period for which to fetch data (e.g., '2021-01-01T00:00:00Z/2021-01-02T00:00:00Z')
    - parameter_id: The ID of the parameter to fetch (e.g., 'temperature')
    - api_key: The API key to use for authentication
    - limit: The maximum number of records to fetch in each request (default: 300000)

    Returns:
    - A list of dictionaries containing the fetched data
    """

    offset = 0  # Start with no offset
    all_data = []  # List to collect all records

    while True:
        # Make the API request with the offset
        params = {
            'parameterId': parameter_id,
            'api-key': api_key,
            'datetime': period,
            'limit': limit,
            'offset': offset,
            'timeResolution': 'hour',
        }
        
        response = requests.get(base_url, params=params)
        response_json = response.json()

        # Extract the features from the response
        features = response_json.get('features', [])

        # If no more features are returned, we have fetched all the data
        if not features:
            break

        # Append the fetched features to the all_data list
        for feature in features:
            feature_data = {
                'cellId': feature['properties']['cellId'],
                'from': feature['properties']['from'],
                'to': feature['properties']['to'],
                parameter_id: feature['properties']['value'],  # Store the value under the parameter name
            }
            all_data.append(feature_data)

        # Increment the offset by the number of records returned in this batch
        offset += response_json.get('numberReturned', 0)

    return all_data


def bulk_load_year(year, file_paths, id_fields):
    """
    Function to bulk load data for a specific year from a list of file paths.

    Args:
    - year: The year for which to load data
    - file_paths: A list of file paths to load data from
    - id_fields: A list of fields to use as IDs for the records

    Returns:
    - A DataFrame containing the aggregated data for the specified
        year and ID fields, or an empty DataFrame if no data was loaded.
        """
    aggregated_data = {}

    total_files = len(file_paths)
    processed_files = 0

    print(f"\nStarted processing {total_files} files for year {year}.")

    # Process files sequentially
    for file_path in file_paths:
        try:
            with open(file_path, 'rb') as file:  # Open in binary mode for orjson
                for line in file:
                    if line.strip():
                        json_obj = orjson.loads(line)
                        properties = json_obj.get('properties', {})
                        geometry = json_obj.get('geometry', {})

                        # Filter out records where timeResolution is not 'hour'
                        if properties.get('timeResolution') != 'hour':
                            continue

                        # Extract ID fields
                        id_values = tuple(properties.get(id_field) for id_field in id_fields)
                        from_time = properties.get('from')
                        to_time = properties.get('to')
                        parameterId = properties.get('parameterId')
                        value = properties.get('value')

                        # Skip records with missing critical data
                        if None in id_values or None in (from_time, to_time, parameterId, value):
                            continue

                        # Build the key
                        key = id_values + (from_time, to_time)

                        # Initialize or update the aggregated record
                        if key not in aggregated_data:
                            # Build the record
                            record = {id_field: id_value for id_field, id_value in zip(id_fields, id_values)}
                            record.update({
                                'from': from_time,
                                'to': to_time,
                                'geometry_type': geometry.get('type'),
                                'coordinates': geometry.get('coordinates')
                            })
                            aggregated_data[key] = record
                        else:
                            record = aggregated_data[key]

                        # Update the record with the parameter value
                        record[parameterId] = value

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

        # Update the processed files counter
        processed_files += 1

        # Print progress every 100 files
        if processed_files % 100 == 0 or processed_files == total_files:
            print(f"Processed {processed_files}/{total_files} files for year {year}.")

    if not aggregated_data:
        print(f"No records were loaded for year {year}. Please check your data and filters.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Convert the aggregated data into a DataFrame
    df = pd.DataFrame.from_dict(aggregated_data, orient='index')

    if df.empty:
        print(f"DataFrame is empty after aggregation for year {year}.")
        return df

    # Optimize data types
    for id_field in id_fields:
        df[id_field] = df[id_field].astype('category')
    df['from'] = pd.to_datetime(df['from'])
    df['to'] = pd.to_datetime(df['to'])

    return df


# function to process all in a folder
def process_all_years(folder_path, output_folder_path, id_fields):
    """
    Function to process all files in a folder and save the aggregated data to CSV files.

    Args:
    - folder_path: The path to the folder containing the input files
    - output_folder_path: The path to the folder where the output CSV files should be saved
    - id_fields: A list of fields to use as IDs for the records

    Returns:
    - None, but saves the aggregated data to CSV files in the specified output folder.

    """

    # Ensure the output folder exists
    os.makedirs(output_folder_path, exist_ok=True)

    # Gather all file paths
    file_paths = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if filename.endswith('.txt')
    ]

    # Create a mapping from year to list of file paths
    year_to_files = {}
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        # Extract the year from the filename using regex
        match = re.match(r'(\d{4})-\d{2}-\d{2}\.txt', filename)
        if match:
            year = match.group(1)
            year_to_files.setdefault(year, []).append(file_path)
        else:
            print(f"Filename {filename} does not match expected pattern. Skipping.")

    # Process files for each year
    for year in sorted(year_to_files.keys()):
        files_for_year = year_to_files[year]
        df_year = bulk_load_year(year, files_for_year, id_fields)

        if not df_year.empty:
            # Construct the output file path
            output_filename = f"{year}.csv"
            output_file_path = os.path.join(output_folder_path, output_filename)

            # Save the DataFrame to a CSV file in the specified output folder
            df_year.to_csv(output_file_path, index=False)
            print(f"Saved data for year {year} to {output_file_path}.")
        else:
            print(f"No data to save for year {year}.")

        # Clear memory
        del df_year
        gc.collect()

# parse the coordinates from string
def parse_poly(coord_str):
    """
    Function to parse a string of coordinates into a list of tuples.
    
    Args:
    - coord_str: A string containing coordinates for a polygon.

    Returns:
    - A list of tuples containing the parsed coordinates.
    """

    return [tuple(map(float, point)) for polygon in ast.literal_eval(coord_str) for point in polygon]

def parse_point(coord_str):
    """
    Function to parse a string of coordinates into a tuple.

    Parameters:
    - coord_str: A string containing coordinates for a point.

    Returns:
    - A tuple containing the parsed coordinates.
    """
    return ast.literal_eval(coord_str)

def plot_map(df, id, color_by=None, save=None, do_print=True):
    """ 
    Function to plot a map of Denmark with the given data.
    
    Args:
    - df: The DataFrame containing the data to plot.
    - id: The name of the column containing the IDs for the data.
    - color_by: The name of the column to use for coloring the data (optional).
    - save: The name of the file to save the plot as (optional).
    - do_print: Whether to print the plot (default: True).
    
    Returns:
    - None, but shows the plot and optionally saves it to a file.
    """
    # parse coordinates depending on the geometry type
    if df['geometry_type'].iloc[0] == 'Polygon':
        df['parsed_coordinates'] = df['coordinates'].apply(parse_poly)
    else:
        df['parsed_coordinates'] = df['coordinates'].apply(parse_point)
    if color_by and color_by in df.columns:
        # If a categorical column is specified, use it for color grouping
        df['color_category'] = df[color_by].astype('category').cat.codes
    else:
        # Default to using the last two digits of the id
        df['color_category'] = df[color_by].apply(lambda x: int(x[-2:]))

    # Normalize the color categories (you can scale them if needed)
    min_color = df['color_category'].min()
    max_color = df['color_category'].max()

    # Create polygons and extract x, y coordinates for each square
    fig = go.Figure()

    for _, row in df.iterrows():
        if row['geometry_type'] == 'Polygon':
            polygon = Polygon(row['parsed_coordinates'])
            x, y = polygon.exterior.xy

            # Convert array.array to list
            x = list(x)
            y = list(y)

            # Normalize color based on the chosen method (either category or id)
            normalized_color = (row['color_category'] - min_color) / (max_color - min_color)

            # Use Plotly's colorscale for colors
            color = f'rgba({int(255 * normalized_color)}, {int(150 * (1 - normalized_color))}, {255-int(255 * normalized_color)}, 0.7)'  # Example using a gradient

            hover_text = f"ID: {row[id]}<br>Area: {row['area']}"

            fig.add_trace(go.Scattermapbox(
                lon=x,
                lat=y,
                fill="toself",
                name=row[id],
                hovertext=hover_text,
                hoverinfo="text",
                mode='lines',
                fillcolor=color,
                line=dict(width=1, color='grey')  # Set line color to a subtle grey
            ))
        elif row['geometry_type'] == 'Point':
            point = Point(row['parsed_coordinates'])
            x, y = point.x, point.y

            # Normalize color based on the chosen method (either category or id)
            normalized_color = (row['color_category'] - min_color) / (max_color - min_color)

            # Use Plotly's colorscale for colors
            color = f'rgba({int(255 * normalized_color)}, {int(150 * (1 - normalized_color))}, {255-int(255 * normalized_color)}, 0.7)'  # Example using a gradient

            hover_text = f"ID: {row[id]}<br>Area: {row['area']}"

            fig.add_trace(go.Scattermapbox(
                lon=[x],
                lat=[y],
                name=row[id],
                hovertext=hover_text,
                hoverinfo="text",
                mode='markers',
                marker=dict(size=10, color=color, opacity=0.7)
            ))

    # Update layout for better visualization and adjusted starting view
    fig.update_layout(
        mapbox_style="carto-positron",  # You can also use "satellite", etc.
        mapbox_zoom=5.8,  # Zoom out slightly to fit all of Denmark
        mapbox_center={"lat": 56.0, "lon": 11.0},  # Adjust center to better fit the map (shift north slightly)
        height=600,
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=False,
    )

    # If a save name is provided, save the figure as a PNG
    if save:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
        save_path = os.path.join(output_dir, save)
        fig.write_image(save_path, width=700, height=600, scale=3)
        print(f"Map saved as: {save_path}")

    # Show the plot
    if do_print:
        fig.show()

def get_energydata(url: str, params: dict):
    """
    Function to fetch data from the Energidata API.
    
    Args:
    - url: The base URL for the API endpoint
    - params: A dictionary of parameters to include in the API request
    
    Returns:
    - A DataFrame containing the fetched data
    """

    res = requests.get(url=url, params=params)
    data = res.json()

    # Extract the records
    records = data.get('records', [])

    # Convert the records into a DataFrame
    df = pd.DataFrame(records)
    
    return df

def degrees_to_cardinal(degrees):
    """
    Function to convert degrees to cardinal directions.

    Args:
    - degrees: The angle in degrees to convert.

    Returns:
    - A string representing the cardinal direction.
    """

    directions = ['N', 'NE',  'E', 'SE', 
                  'S', 'SW', 'W', 'NW']
    idx = round(degrees / 45) % 8
    return directions[idx]

def pca_filtering(df, prefix, n_components=0.99):
    """
    Function to perform PCA on columns with a given prefix in a DataFrame.

    Args:
    - df: The DataFrame containing the data to process.
    - prefix: The prefix to use for filtering columns.
    - n_components: The number of components to keep, or the explained variance ratio to retain.

    Returns:
    - A DataFrame containing the reduced data.
    """
    pca = PCA(n_components=n_components, svd_solver='full')
    
    data_n = pca.fit_transform(df.filter(like=prefix))
    pca_columns = [f"{prefix}pca_{i+1}" for i in range(pca.n_components_)]
    df_n = pd.DataFrame(data_n, columns=pca_columns)
    
    print(f"Reduced from {df.filter(like=prefix).shape[1]} to {pca.n_components_}, explained variance: {pca.explained_variance_ratio_.sum()}")    
    return df_n

# define function for cyclical encoding
def cyclical_encoding(df, column, max_val):
    """
    Function to perform cyclical encoding of a column in a DataFrame.
    
    Args:
    - df: The DataFrame containing the data to process.
    - column: The name of the column to encode.
    - max_val: The maximum value for the cyclical encoding.
    
    Returns:
    - A DataFrame containing the encoded data.
    """

    df[column + '_sin'] = np.sin(2 * np.pi * df[column]/max_val)
    df[column + '_cos'] = np.cos(2 * np.pi * df[column]/max_val)
    df.drop(column, axis=1, inplace=True)
    return df

def cyclical_wind_encoding(df, wind_col):
    """
    Function to perform cyclical encoding of wind directions in a DataFrame.
    
    Parameters:
    - df: The DataFrame containing the data to process.
    - wind_col: The name of the column containing wind directions.
    
    Returns:
    - A DataFrame containing the encoded wind data.
    """
    if wind_col not in df.columns:
        raise ValueError(f"Column '{wind_col}' not found in the DataFrame.")
    
    radians = np.deg2rad(df[wind_col])
    df[wind_col + '_sin'] = np.sin(radians)
    df[wind_col + '_cos'] = np.cos(radians)
    
    return df