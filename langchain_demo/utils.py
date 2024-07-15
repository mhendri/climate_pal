import re

def get_urls(path):
    """ Get urls from a file and return a dictionary of the urls with the following keys:
    activity_id, experiment_id, temporal_resolution, variable
    """
    dicts = {}
    with open(path, 'r') as file:
        for l in file.readlines():
            if '/v20' in l and '.html' in l: # why only html?
                url = l[:-1] #remove newline char
                lsplit = url.split('/')
                url_clean = '/'.join(lsplit[:-1]) # so they go up to and including the version directory (removing .html)
                dicts[url_clean] = {'activity_id':lsplit[3],
                                'experiment_id':lsplit[6],
                                'temporal_resolution':lsplit[8],
                                'variable':lsplit[9]}

    print("Length of dictionary: ",len(dicts))
    return dicts

# Function to read the file and populate the dictionary
def read_dictionary_from_file(file_path):
    dictionary = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.split(': ', 1)  # Split each line into key and value
            dictionary[key.strip()] = value.strip()  # Add the key-value pair to the dictionary
    return dictionary

def queries_to_list(result):
    '''This function takes the output of the llm and returns a list of individuals queries'''
    pattern = re.compile(r'([Qq]uery|[Ss]tart query) (\d+): "([^"]+)"')

    queries = {}

    matches = pattern.findall(result)

    for match in matches:
        query_number = f'Query {match[1]}'
        query_text = match[2]
        queries[query_number] = query_text

    return queries