# functions created for seminar in Asset Pricing and Financial Markets at the University of Copenhagen, Fall 2024

import requests
import pandas as pd
import os
import re
import gc
import pandas as pd
import orjson 
from shapely.geometry import Polygon
import plotly.graph_objects as go


def dmi_api(base_url, period, parameter_id, api_key, limit=300000):
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


# bulk loader function for every year
def bulk_load_year(year, file_paths):
    all_records = []
    
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

                        cellId = properties.get('cellId')
                        from_time = properties.get('from')
                        to_time = properties.get('to')
                        parameterId = properties.get('parameterId')
                        value = properties.get('value')

                        # Skip records with missing critical data
                        if None in (cellId, from_time, to_time, parameterId, value):
                            continue

                        key = (cellId, from_time, to_time)

                        # Build a record
                        record = {
                            'key': key,
                            'cellId': cellId,
                            'from': from_time,
                            'to': to_time,
                            parameterId: value
                        }

                        # Always add geometry
                        record['geometry_type'] = geometry.get('type')
                        coordinates = geometry.get('coordinates', [[]])
                        flat_coords = coordinates[0] if coordinates else []
                        record['coordinates'] = flat_coords

                        all_records.append(record)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

        # Update the processed files counter
        processed_files += 1

        # Print progress every 10 files
        if processed_files % 100 == 0 or processed_files == total_files:
            print(f"Processed {processed_files}/{total_files} files for year {year}.")

    if not all_records:
        print(f"No records were loaded for year {year}. Please check your data and filters.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Aggregate records by key
    aggregated_data = {}
    for record in all_records:
        key = record.pop('key')
        if key not in aggregated_data:
            aggregated_data[key] = record
        else:
            # Update existing record with new parameterId and value
            aggregated_data[key].update(record)

    # Convert the aggregated data into a DataFrame
    df = pd.DataFrame(aggregated_data.values())

    if df.empty:
        print(f"DataFrame is empty after aggregation for year {year}.")
        return df

    # Optimize data types if 'cellId' exists
    if 'cellId' in df.columns:
        df['cellId'] = df['cellId'].astype('category')
    else:
        print(f"Warning: 'cellId' column is missing in the DataFrame for year {year}.")

    if 'from' in df.columns:
        df['from'] = pd.to_datetime(df['from'])
    if 'to' in df.columns:
        df['to'] = pd.to_datetime(df['to'])

    return df


# function to process all in a folder
def process_all_years(folder_path, output_folder_path):
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
        df_year = bulk_load_year(year, files_for_year)

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

def plot_polygons_from_df(df):
    # Initialize a list to store the polygons and their properties
    polygons = []
    
    # Iterate over the rows in the dataframe to convert coordinates to Polygons
    for index, row in df.iterrows():
        coordinates = row['coordinates']
        
        if coordinates:  # Ensure there are coordinates
            try:
                # Create a Shapely Polygon from the coordinates
                polygon = Polygon(coordinates)
                polygons.append(polygon)
            except ValueError:
                print(f"Invalid polygon at row {index}")
                polygons.append(None)
        else:
            polygons.append(None)

    # Add the polygons as a geometry column to the DataFrame
    df['geometry'] = polygons

    # Drop duplicate polygons to avoid plotting them multiple times
    df_unique = df.drop_duplicates(subset=['geometry'])

    # Extract coordinates in Plotly compatible format (lon, lat for each polygon)
    fig = go.Figure()

    for i, row in df_unique.iterrows():
        polygon_coords = row['coordinates']
        if polygon_coords:
            # Extract the longitude and latitude from the polygon coordinates
            lons, lats = zip(*polygon_coords)

            # Close the polygon by repeating the first point at the end
            lons = list(lons) + [lons[0]]
            lats = list(lats) + [lats[0]]

            # Add the polygon to the Plotly map
            fig.add_trace(go.Scattermapbox(
                fill="toself",
                lon=lons,
                lat=lats,
                mode="lines",
                line=dict(width=1),
                name=row['cellId'],  # Show the parameter ID in the legend
                text=f"Value: {row['value']} Date: {row['from']}",  # Add value as hover info
                hoverinfo="text"
            ))

    # Set up the layout for Plotly Mapbox
    fig.update_layout(
        mapbox_style="carto-positron",  # You can also use "carto-positron", "satellite" etc.
        mapbox_zoom=5,  # Adjust zoom level
        mapbox_center={"lat": 55.0, "lon": 9.5},  # Adjust the center based on your data
        height=600,
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=True
    )

    # Show the figure
    fig.show()


def get_energydata(url: str, params: dict):
    res = requests.get(url=url, params=params)
    data = res.json()

    # Extract the records
    records = data.get('records', [])

    # Convert the records into a DataFrame
    df = pd.DataFrame(records)
    
    return df