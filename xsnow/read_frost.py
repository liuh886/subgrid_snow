import requests, os, glob
import pandas as pd
import pyproj

def read_frost(sources, elements, start, end):
    """ Reads data from Met.no's frost.met.no.
    Based on example from https://frost.met.no/python_example.html.

    Parameters:
    sources: list of station name (list)
    elements: list of element (list)
    start: starttime in formay yyy-mm-dd (str)
    end: endtime in formay yyy-mm-dd (str)

    Returns:
    Dataframe with requested data
    """
    client_id = '379065c1-8d6f-438c-88e2-615938114513'
    # dc7282ea-4dba-43d4-8278-455674c89362

    sep = ','
    time_sep = '/'
    sources = sep.join(sources)
    elements = sep.join(elements)
    referencetime = time_sep.join([start,end])

    endpoint = 'https://frost.met.no/observations/v0.jsonld'
    parameters = {'sources': sources,
                'elements': elements,
                'referencetime': referencetime}

    # Issue an HTTP GET request
    r = requests.get(endpoint, parameters, auth=(client_id,''))
    # Extract JSON data
    json = r.json()

    # Check if the request worked, print out any errors
    if r.status_code == 200:
        data = json['data']
        qt = json['queryTime']
        n = json['totalItemCount']
        print(f"Retrieved {n} items from frost.met.no with query time {qt} s")
    else:
        print('Error! Returned status code %s' % r.status_code)
        print('Message: %s' % json['error']['message'])
        print('Reason: %s' % json['error']['reason'])
        return r.status_code

    print("Converting to df")
    df = pd.json_normalize(json['data'], record_path = 'observations', meta = 'referenceTime')
    df['referenceTime'] = pd.to_datetime(df['referenceTime'])
    df.set_index('referenceTime',drop=True, inplace=True)
    df.sort_index(inplace=True)
    return df

def find_station(country='NO',
                 polygon='POLYGON ((10 60,10 65, 11 65, 10 60))',
                 validtime='2008-01-01/2022-12-31',
                 sources=None,
                 crs='32633'):
    # find stations
    client_id = '379065c1-8d6f-438c-88e2-615938114513'

    endpoint = 'https://frost.met.no/sources/v0.jsonld'
    parameters = {
        'validtime' : validtime
    }
    if country:
        parameters['country'] = country
    if polygon:
        parameters['geometry'] = polygon
    if sources:
        parameters['ids'] = sources

    r = requests.get(endpoint, parameters, auth=(client_id,''))
    # Extract JSON data
    json = r.json()

    # Check if the request worked, print out any errors
    if r.status_code == 200:
        data = json['data']
        print('Data retrieved from frost.met.no!')
    else:
        print('Error! Returned status code %s' % r.status_code)
        print('Message: %s' % json['error']['message'])
        print('Reason: %s' % json['error']['reason'])

    # list all station available
    station_df = pd.DataFrame(data).dropna(subset=['masl', 'county'])

    # Assuming you have the DataFrame stored in a variable called 'df'
    # Extract the latitude and longitude from the nested 'geometry' field
    station_df['longitude'] = station_df['geometry'].apply(lambda x: float(x['coordinates'][0]))
    station_df['latitude'] = station_df['geometry'].apply(lambda x: float(x['coordinates'][1]))

    # Drop the nested 'geometry' field from the DataFrame
    station_df.drop('geometry', axis=1, inplace=True)

    if crs == '32633':
        # Define the projected coordinate system (EPSG:32633)
        crs_target = pyproj.CRS('EPSG:32633')

        # Convert latitude and longitude columns to projected coordinates in EPSG 32633
        transformer = pyproj.Transformer.from_crs('EPSG:4326', crs_target)
        station_coords = station_df.apply(lambda row: transformer.transform(row['latitude'], row['longitude']), axis=1).tolist()
        station_df[['E', 'N']] = station_coords

    return station_df

def read_frost_loop(sources, elements=None, referencetime=None):
    """ Reads data from Met.no's frost.met.no.
    Based on example from https://frost.met.no/python_example.html.

    Parameters:
    sources: list of station name (list)
    elements: list of element (list)
    start: starttime in formay yyy-mm-dd (str)
    end: endtime in formay yyy-mm-dd (str)

    Returns:
    Dataframe with requested data
    """
    client_id = '379065c1-8d6f-438c-88e2-615938114513'
    # dc7282ea-4dba-43d4-8278-455674c89362

    # Define endpoint and parameters

    endpoint = 'https://frost.met.no/observations/v0.jsonld'
    parameters = {}

    if elements:
        parameters['elements'] = ','.join(elements)
    if referencetime:
        parameters['referencetime'] =referencetime

    # List to store the results
    results = []

    # Loop over the station list
    for id in sources:
        # Add the current station to the parameters
        parameters['sources'] = id
        
        # Issue an HTTP GET request
        r = requests.get(endpoint, parameters, auth=(client_id,''))
        
        # Extract JSON data
        json = r.json()
        
        # Check if the request worked
        if r.status_code == 200:
            # ("Converting to df")
            new_df = pd.json_normalize(json['data'], record_path = 'observations', meta = 'referenceTime')
            # Add the station id column to the dataframe
            new_df['station_id'] = id
            results.append(new_df)
            
        else:
            print(f'Error for station {id}: {json["error"]["message"]}')

    # Concatenate the results into a single dataframe
    result_df = pd.concat(results, ignore_index=True)

    result_df['referenceTime'] = pd.to_datetime(result_df['referenceTime'])
    result_df.set_index('referenceTime',drop=True, inplace=True)
    result_df.sort_index(inplace=True)

    return result_df

def read_climate_normals(elements,sources,periods):
    """ Reads data from Met.no's frost.met.no.
    Parameters:
    elements: list of element (list)

    Returns:
    Dataframe with requested data
    """
    client_id = os.getenv('FROST_API_CLIENTID')

    sep = ','
    time_sep = '/'
    sources = sep.join(sources)
    elements = sep.join(elements)
    periods = time_sep.join(periods)

    endpoint = 'https://frost.met.no/climatenormals/v0.jsonld'
    parameters = {
                'period': periods,
                'elements': elements,
                'sources': sources
                }

    # Issue an HTTP GET request
    r = requests.get(endpoint, parameters, auth=(client_id,''))
    # Extract JSON data
    json = r.json()

    # Check if the request worked, print out any errors
    if r.status_code == 200:
        data = json['data']
        qt = json['queryTime']
        n = json['totalItemCount']
        print(f"Retrieved {n} items from frost.met.no with query time {qt} s")
    else:
        print('Error! Returned status code %s' % r.status_code)
        print('Message: %s' % json['error']['message'])
        print('Reason: %s' % json['error']['reason'])
        return r.status_code

    print("Converting to df")
    df = pd.json_normalize(json['data'])
    return df


def frost_info(sources, start, end):
    """
    Script to find out which variable are available for a given station between
    given start and end date

    Parameters:
    sources: list of station number (list)
    elements: list of variables (list)
    start: starttime in formay yyy-mm-dd (str)
    end: endtime in formay yyy-mm-dd (str)
    """
    client_id = os.getenv('379065c1-8d6f-438c-88e2-615938114513')
    time_sep = '/'
    sep = ','

    sources = sep.join(sources)
    referencetime = time_sep.join([start,end])

    endpoint = 'https://frost.met.no/observations/availableTimeSeries/v0.jsonld'
    parameters = {
        'sources': sources,
        'referencetime': referencetime,
        }

    # Issue an HTTP GET request
    r = requests.get(endpoint, parameters, auth=(client_id,''))
    json = r.json()
    # data_list = []
    # [data_list.append([d['elementId'],d['validFrom']]) for d in json]
    # data = pd.DataFrame(data_list, names=['elementID','available_from'])
    df = pd.json_normalize(json['data'])
    outdir = os.path.join('..','data','info')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for id, data in df.groupby('sourceId'):
        source = id.split(':')[0]
        outfile = f"{source}_{start[:4]}_{end[:4]}.csv"
        outfile = os.path.join(outdir,outfile)
        data.drop(['timeSeriesId',
                'exposureCategory','status',
                'uri','codeTable'],axis=1,inplace=True)
        data.to_csv(outfile)
    return

def climate_normals_info(sources):
    endpoint = 'https://frost.met.no/climatenormals/available/v0.jsonld'
    client_id = os.getenv('379065c1-8d6f-438c-88e2-615938114513')
    sep = ','
    sources = sep.join(sources)
    parameters = {
        'sources': sources,
        }
    r = requests.get(endpoint, parameters, auth=(client_id,''))
    json = r.json()
    df = pd.json_normalize(json['data'])
    outdir = os.path.join('..','data','info')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for id, data in df.groupby('sourceId'):
        source = id.split(':')[0]
        outfile = f"{source}_climate_normals.csv"
        outfile = os.path.join(outdir,outfile)
        data.to_csv(outfile)
    return df