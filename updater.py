# IMPORTS -------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import requests
import json
import re
from datetime import datetime
import time
from tqdm import tqdm

from bs4 import BeautifulSoup as bs4

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


# CONSTANTS ------------------------------------------------------------------ #

# Set constants for data provider and data API.
PROVIDER = "OpenDataZurich"
PROVIDER_LINK = "https://data.stadt-zuerich.ch/"
BASELINK_DATAPORTAL = "https://data.stadt-zuerich.ch/dataset/"
CKAN_API_LINK = (
    "https://data.stadt-zuerich.ch/api/3/action/current_package_list_with_resources"
)
LANGUAGE = "de"

# Set constants in regard to GitHub account and repo.
GITHUB_ACCOUNT = "opendatazurich"
REPO_NAME = "starter-code"
REPO_BRANCH = "main"
REPO_RMARKDOWN_OUTPUT = "01_r-markdown/"
REPO_PYTHON_OUTPUT = "02_python/"
TEMP_PREFIX = "_work/"

# Set local folders and file names.
TEMPLATE_FOLDER = "_templates/"
# Template for the README.md in the repo.
TEMPLATE_README = "template_md_readme.md"
# Header for list overview that is rendered as a GitHub page.
TEMPLATE_HEADER = "template_md_header.md"
TEMPLATE_PYTHON = "template_python.ipynb"
TEMPLATE_PYTHON_GEO = "template_python_geo.ipynb"
TEMPLATE_RMARKDOWN = "template_rmarkdown.Rmd"
TEMPLATE_RMARKDOWN_GEO = "template_rmarkdown_geo.Rmd"
METADATA_FOLDER = "_metadata_json/"

TODAY_DATE = datetime.today().strftime("%Y-%m-%d")
TODAY_DATETIME = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

# Set max length of dataset title in markdown table.
TITLE_MAX_CHARS = 200

# Sort markdown table by this feature.
SORT_TABLE_BY = f"title"

# Select keys in metadata for dataset and distributions.
KEYS_DATASET = [
    "publisher",
    #f"organization.display_name.{LANGUAGE}",
    #"organization.url",
    "maintainer",
    "maintainer_email",
    "keywords",
    "tags",
    #"issued",
    "metadata_created",
    "metadata_modified",
]
KEYS_DISTRIBUTIONS = ["package_id", "description", "format", "resource_type", "name", "url"]


PREFIX_RESOURCE_COLS = "resources_"
RESOURCE_COLS_TO_KEEP = [
    'name',
    'filename',
    'format',
    'url',
    'id',
    'resource_type',
    'package_id',
]


# FUNCTIONS ------------------------------------------------------------------ #


def get_full_package_list(limit=500, sleep=2):
    """Get full package list from CKAN API"""
    offset = 0
    frames = []
    while True:
        print(f"{offset} packages retrieved.")
        url = CKAN_API_LINK + f"?limit={limit}&offset={offset}"
        res = requests.get(url)
        data = json.loads(res.content)
        if data["result"] == []:
            break
        data = pd.DataFrame(pd.json_normalize(data["result"]))
        frames.append(data)
        offset += limit
        time.sleep(sleep)
    data = pd.concat(frames)
    data.reset_index(drop=True, inplace=True)
    return data

def dataset_to_resource(all_packages, prefix_resource_cols=PREFIX_RESOURCE_COLS, resource_cols_to_keep=RESOURCE_COLS_TO_KEEP):
    """
    Takes pandas df with all datasets (one row for each dataset).
    Column "resources" must contain json info for each resource like:
    [{'cache_last_updated': None, 'cache_url': None},...]
    Json fields in resource get a prefix: prefix_resource_cols
    This function explodes the df, so that each row in the output represents one resource.
    """
    # explode every resource in one row
    all_packages_exploded = all_packages.explode('resources')
    # json to columns and only keep the selected
    resource_cols = pd.json_normalize(all_packages_exploded['resources'])[resource_cols_to_keep]
    # add prefix, to avoid already existing columns
    resource_cols = resource_cols.add_prefix(prefix_resource_cols)
    # merge data from package/dataset
    merged = resource_cols.merge(all_packages, how='left', left_on=PREFIX_RESOURCE_COLS+"package_id",right_on='id')

    # reset index, because later functions will need unique indices
    merged = merged.reset_index(drop=True)

    return merged

def filter_resources(df, desired_formats=['csv','parquet','wfs']):
    """
    Filter df with resources for desired_formats (e.g. csv). Formats should be lower case.
    Be aware that the filtered column has to match the prefix defined in dataset_to_resource.
    """
    return df[df[PREFIX_RESOURCE_COLS+'format'].str.lower().isin(desired_formats)]

def filter_resources(df, desired_formats=['table_data','geo_data']):
    """
    Filter df with resources for desired_formats (e.g. csv). 
    Be aware that the filtered column has to match the prefix defined in dataset_to_resource.
    returns a dict with desired_formats as keys and filtered dataframes as values
    
    """
    out_dict = {}
    if "table_data" in desired_formats:
        # filter desired file formats
        table_formats = ['csv', 'parquet']
        table_data = df[df[PREFIX_RESOURCE_COLS+'format'].str.lower().isin(table_formats)]
        # do not filter geo data csvs
        table_data = table_data[~table_data['tags'].apply(lambda tag: 'geodaten' in tag)]
        out_dict['table_data'] = table_data

    if "geo_data" in desired_formats:
        # filter desired file formats
        geo_data = df[df[PREFIX_RESOURCE_COLS+'url'].str.contains('geojson')]
        # only filter resources from the city of Zürich (not canton)
        geo_data = geo_data[geo_data['tags'].apply(lambda tag: 'stzh' in tag)]
        out_dict['geo_data'] = geo_data

    # set col value for table data
    if "table_data" in desired_formats:
        table_formats = ['csv', 'parquet']
        df.loc[
            # # filter desired file formats
            (df[PREFIX_RESOURCE_COLS+'format'].str.lower().isin(table_formats)) & 
            # do not filter geo data csvs
            (~df['tags'].apply(lambda tag: 'geodaten' in tag)),
            'format_filter'
        ] = 'table_data'
    
    if "geo_data" in desired_formats:
        df.loc[
            # # filter desired file formats
            (df[PREFIX_RESOURCE_COLS+'url'].str.contains('geojson')) & 
            # only filter resources from the city of Zürich (not canton)
            (df['tags'].apply(lambda tag: 'stzh' in tag)),
            'format_filter'
        ] = 'geo_data'


    return df[df['format_filter'].notna()]

def has_csv_distribution(dists):
    """Iterate over package resources and keep only CSV entries in list"""
    csv_dists = [x for x in dists if x.get("format", "") == "CSV"]
    if csv_dists != []:
        return csv_dists
    else:
        return np.nan


def filter_csv(data):
    """Remove all datasets that have no CSV distribution"""
    data.resources = data.resources.apply(has_csv_distribution)
    data.dropna(subset=["resources"], inplace=True)
    return data.reset_index(drop=True)


def extract_keywords(x, sep=','):
    """
    Extract keywords from ckan metadata json. To be used in pandas.apply()
    Example: [{'description': '', 'display_name': 'Mobilität'},]
    """
    out_string = ''
    for elem in x:
        out_string += elem['display_name']+sep
    return out_string.rstrip(sep)


def clean_features(data):
    """Clean various features"""
    # Reduce publisher data to name.
    # In rare cases the publisher is not provided.
    data['publisher'] = data['author'] #.apply(lambda x: json.loads(x)["name"] if "name" in json.loads(x) else "Publisher not provided")

    # Reduce tags to tag names.
    data.tags = data.tags.apply(lambda x: [tag["name"] for tag in x])

    # keywords/groups
    data['keywords'] = data['groups'].apply(extract_keywords)

    return data


def prepare_data_for_codebooks(data):
    """Prepare metadata from catalogue in order to create the code files"""
    # Add new features to save prepared data.
    data["metadata"] = None
    data["contact"] = ""
    data["distributions"] = None
    data["distribution_links"] = None

    # Iterate over datasets and create additional data for markdown and code cells.
    for idx in tqdm(data.index):
        md = [f"- **{k.capitalize()}** `{data.loc[idx, k]}`\n" for k in KEYS_DATASET]
        data.loc[idx, "metadata"] = "".join(md)


    data['description'] = data['notes']
    # Sort values for table.
    data.sort_values(f"{SORT_TABLE_BY}", inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data#[REDUCED_FEATURESET]


def create_python_notebooks(data, notebook_template):
    """Create Jupyter Notebooks with Python starter code"""
    for idx in tqdm(data.index):
        print(data.loc[idx, "name"])
        with open(f"{TEMPLATE_FOLDER}{notebook_template}") as file:
            py_nb = file.read()

        # Populate template with metadata.
        py_nb = py_nb.replace("{{ PROVIDER }}", PROVIDER)

        title = re.sub('"', "'", data.loc[idx, f"title"])
        py_nb = py_nb.replace("{{ DATASET_TITLE }}", title)

        description = data.loc[idx, f"description"]
        description = re.sub('"', "'", description)
        description = re.sub("\\\\", "|", description)
        py_nb = py_nb.replace("{{ DATASET_DESCRIPTION }}", description)

        ssz_comments = str(data.loc[idx, "sszBemerkungen"])
        ssz_comments = re.sub('"', "'", ssz_comments)
        ssz_comments = re.sub("\\\\", "|", ssz_comments)
        py_nb = py_nb.replace("{{ SSZ_COMMENTS }}", ssz_comments)

        py_nb = py_nb.replace("{{ DATASET_IDENTIFIER }}", data.loc[idx, "name"])
        py_nb = py_nb.replace(
            "{{ DATASET_METADATA }}", re.sub('"', "'", data.loc[idx, "metadata"])
        )
        # py_nb = py_nb.replace("{{ DISTRIBUTION_COUNT }}", str(len(data.loc[idx, "distributions"]))        )

        url = f'[Direct link by {PROVIDER} for dataset]({BASELINK_DATAPORTAL}{data.loc[idx, "name"]})\n\n{data.loc[idx,PREFIX_RESOURCE_COLS+"url"]}'
        py_nb = py_nb.replace("{{ DATASHOP_LINK_PROVIDER }}", url)



        py_nb = py_nb.replace("{{ CONTACT }}", data.loc[idx, "maintainer_email"])

        py_nb = json.loads(py_nb, strict=False)

        # Find code cell for dataset imports.
        for id_cell, cell in enumerate(py_nb["cells"]):
            if cell["id"] == "0":
                dist_cell_idx = id_cell
                break

        
        # add metadata from resource
        code_block = ""
        for col in RESOURCE_COLS_TO_KEEP:
            prefix_col = PREFIX_RESOURCE_COLS+col
            # spacer = 30 - len(col)+ 5
            code_block += f"# {col}: {data.loc[idx,prefix_col]}\n"
        # add url to load
        url = data.loc[idx,PREFIX_RESOURCE_COLS+"url"]
        if notebook_template == TEMPLATE_PYTHON_GEO:
            # naming convention for geopandas dataframe is gdf
            df_prefix = 'g'
        else:
            df_prefix = ''
        code_block += f"\n{df_prefix}df = get_dataset('{url}')\n"
        py_nb["cells"][dist_cell_idx]["source"] = code_block

        # Save to disk.
        with open(
            f'{TEMP_PREFIX}{REPO_PYTHON_OUTPUT}{data.loc[idx, "name"]}_{data.loc[idx, PREFIX_RESOURCE_COLS+"id"]}.ipynb',
            "w",
            encoding="utf-8",
        ) as file:
            file.write(json.dumps(py_nb))


def create_rmarkdown(data, notebook_template):
    """Create R Markdown files with R starter code"""
    for idx in tqdm(data.index):
        with open(f"{TEMPLATE_FOLDER}{notebook_template}") as file:
            rmd = file.read()

        # Populate template with metadata.
        title = f"Open Government Data, {PROVIDER}"
        rmd = rmd.replace("{{ DOCUMENT_TITLE }}", title)

        title = re.sub('"', "'", data.loc[idx, f"title"])
        rmd = rmd.replace("{{ DATASET_TITLE }}", title)

        rmd = rmd.replace("{{ TODAY_DATE }}", TODAY_DATE)
        rmd = rmd.replace("{{ DATASET_IDENTIFIER }}", data.loc[idx, "name"])

        description = data.loc[idx, f"description"]
        description = re.sub('"', "'", description)
        description = re.sub("\\\\", "|", description)
        rmd = rmd.replace("{{ DATASET_DESCRIPTION }}", description)

        ssz_comments = str(data.loc[idx, "sszBemerkungen"])
        ssz_comments = re.sub('"', "'", ssz_comments)
        ssz_comments = re.sub("\\\\", "|", ssz_comments)
        rmd = rmd.replace("{{ SSZ_COMMENTS }}", ssz_comments)

        rmd = rmd.replace("{{ DATASET_METADATA }}", data.loc[idx, "metadata"])
        rmd = rmd.replace("{{ CONTACT }}", data.loc[idx, "maintainer_email"])
        # rmd = rmd.replace("{{ DISTRIBUTION_COUNT }}", str(len(data.loc[idx, "distributions"])))

        url = f'[Direct link by **{PROVIDER}** for dataset]({BASELINK_DATAPORTAL}{data.loc[idx, "name"]})'
        rmd = rmd.replace("{{ DATASHOP_LINK_PROVIDER }}", url)

        # add metadata from resource
        code_block = ""
        for col in RESOURCE_COLS_TO_KEEP:
            prefix_col = PREFIX_RESOURCE_COLS+col
            code_block += f"# {col}: \t\t{data.loc[idx,prefix_col]}\n"
        # add url to load
        url = data.loc[idx,PREFIX_RESOURCE_COLS+"url"]
        code_block += f"\ndf <- read_delim('{url}')\n"

        rmd = rmd.replace("{{ DISTRIBUTIONS }}", code_block)

        # Save to disk.
        with open(
            f'{TEMP_PREFIX}{REPO_RMARKDOWN_OUTPUT}{data.loc[idx, "name"]}_{data.loc[idx, PREFIX_RESOURCE_COLS+"id"]}.Rmd',
            "w",
            encoding="utf-8",
        ) as file:
            file.write("".join(rmd))


def get_header(dataset_count):
    """Retrieve header template and populate with date and count of data records"""
    with open(f"{TEMPLATE_FOLDER}{TEMPLATE_HEADER}", encoding="utf-8") as file:
        header = file.read()
    gh_page = f"https://{GITHUB_ACCOUNT}.github.io/{REPO_NAME}/"
    header = re.sub("{{ GITHUB_PAGE }}", gh_page, header)

    gh_link = f"https://www.github.com/{GITHUB_ACCOUNT}/{REPO_NAME}"
    header = re.sub("{{ GITHUB_REPO }}", gh_link, header)

    header = re.sub("{{ PROVIDER }}", PROVIDER, header)
    header = re.sub("{{ DATA_PORTAL }}", PROVIDER_LINK, header)
    header = re.sub("{{ DATASET_COUNT }}", str(int(dataset_count)), header)
    header = re.sub("{{ TODAY_DATE }}", TODAY_DATETIME, header)

    gh_account_repo = f'{GITHUB_ACCOUNT}/{REPO_NAME}'
    header = re.sub("{{ GITHUB_ACCOUNT_REPO }}", gh_account_repo, header)

    return header


def create_readme(dataset_count):
    """Retrieve README template and populate with metadata"""
    with open(f"{TEMPLATE_FOLDER}{TEMPLATE_README}", encoding="utf-8") as file:
        readme = file.read()
    readme = re.sub("{{ PROVIDER }}", PROVIDER, readme)
    readme = re.sub("{{ DATASET_COUNT }}", str(int(dataset_count)), readme)
    readme = re.sub("{{ DATA_PORTAL }}", PROVIDER_LINK, readme)
    gh_page = f"https://{GITHUB_ACCOUNT}.github.io/{REPO_NAME}/"
    readme = re.sub("{{ GITHUB_PAGE }}", gh_page, readme)
    readme = re.sub("{{ TODAY_DATE }}", TODAY_DATETIME, readme)
    with open(f"{TEMP_PREFIX}README.md", "w", encoding="utf-8") as file:
        file.write(readme)


def create_overview(data, header):
    """Create README with link table"""
    baselink_r_gh = f"https://github.com/{GITHUB_ACCOUNT}/{REPO_NAME}/blob/{REPO_BRANCH}/{REPO_RMARKDOWN_OUTPUT}/"
    baselink_py_gh = f"https://github.com/{GITHUB_ACCOUNT}/{REPO_NAME}/blob/{REPO_BRANCH}/{REPO_PYTHON_OUTPUT}/"
    baselink_py_colab = f"https://githubtocolab.com/{GITHUB_ACCOUNT}/{REPO_NAME}/blob/{REPO_BRANCH}/{REPO_PYTHON_OUTPUT}/"
    # baselink_py_kaggle = f"https://kaggle.com/kernels/welcome?src={baselink_py_gh}"

    md_doc = []
    md_doc.append(header)
    md_doc.append(
        f"| Title (abbreviated to {TITLE_MAX_CHARS} chars) | Fileinfo | Python Colab | Python GitHub | R GitHub |\n"
    )
    md_doc.append("| :-- | :-- | :-- | :-- | :-- |\n")

    for idx in tqdm(data.index):
        # Remove square brackets from title, since these break markdown links.
        title_clean = (
            data.loc[idx, f"title"].replace("[", " ").replace("]", " ")
        )
        if len(title_clean) > TITLE_MAX_CHARS:
            title_clean = title_clean[:TITLE_MAX_CHARS] + "…"

        resource_format = f'{data.loc[idx, PREFIX_RESOURCE_COLS+"filename"]} ({data.loc[idx, PREFIX_RESOURCE_COLS+"format"]})'
        
        ds_link = f'{BASELINK_DATAPORTAL}{data.loc[idx, "name"]}'
        filename = f'{data.loc[idx, "name"]}_{data.loc[idx, PREFIX_RESOURCE_COLS+"id"]}'#data.loc[idx, "id"]

        r_gh_link = f"[R GitHub]({baselink_r_gh}{filename}.Rmd)"

        py_gh_link = f"[Python GitHub]({baselink_py_gh}{filename}.ipynb)"
        py_colab_link = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({baselink_py_colab}{filename}.ipynb)"
        # py_kaggle_link = f'[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]({baselink_py_kaggle}{filename}.ipnyb)'

        md_doc.append(
            f"| [{title_clean}]({ds_link}) | {resource_format} | {py_colab_link} | {py_gh_link} | {r_gh_link} |\n"
        )

    md_doc = "".join(md_doc)

    with open(f"{TEMP_PREFIX}index.md", "w", encoding="utf-8") as file:
        file.write(md_doc)


# CREATE CODE FILES ---------------------------------------------------------- #

all_packages = get_full_package_list()

# df = filter_csv(all_packages)

df = dataset_to_resource(all_packages)
df = clean_features(df)
df = filter_resources(df)
df = prepare_data_for_codebooks(df)

# limit output
df = df.head(20)

# table data
print("Make notebooks for table data")
df_tabledata = df[df['format_filter']=='table_data']
create_python_notebooks(df_tabledata, TEMPLATE_PYTHON)
create_rmarkdown(df_tabledata, TEMPLATE_RMARKDOWN)

# geodata
print("Make notebooks for geo data")
df_geodata = df[df['format_filter']=='geo_data']
create_python_notebooks(df_geodata, TEMPLATE_PYTHON_GEO)
create_rmarkdown(df_geodata, TEMPLATE_RMARKDOWN_GEO)


print(df)

header = get_header(dataset_count=len(df))
create_readme(dataset_count=len(df))
create_overview(df, header)
