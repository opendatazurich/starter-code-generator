# IMPORTS -------------------------------------------------------------------- #

import pandas as pd
import numpy as np
import requests
import json
import re
from datetime import datetime
import time


# CONSTANTS ------------------------------------------------------------------ #

# Set constants for data provider and data API.
PROVIDER = "OpenDataZurich"
PROVIDER_LINK = "https://data.stadt-zuerich.ch/"
BASELINK_DATAPORTAL = "https://data.stadt-zuerich.ch/dataset/"
CKAN_API_LINK = (
    "https://data.stadt-zuerich.ch/api/3/action/current_package_list_with_resources"
)

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

# Select keys in metadata for dataset and distributions.
KEYS_DATASET = [
    "publisher",
    "maintainer",
    "maintainer_email",
    "keywords",
    "tags",
    "metadata_created",
    "metadata_modified",
]


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

# renku constants
RENKU_NAMESPACE = "opendatazurich"
RENKU_PROJECT_SLUG = "starter-code"
RENKU_SESSION_ID = "01JZT3TY89P6YRMMJXV9PEDQZW"

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
    print("Number of datasets", data.shape[0])
    return data

def dataset_to_resource(all_packages, prefix_resource_cols=PREFIX_RESOURCE_COLS, resource_cols_to_keep=RESOURCE_COLS_TO_KEEP):
    """
    Takes pandas df with all datasets (one row for each dataset).
    Column "resources" must contain json info for each resource like:
    [{'cache_last_updated': None, 'cache_url': None},...]
    Json fields in resource get a prefix: prefix_resource_cols
    This function explodes the df, so that each row in the output represents one resource.
    """
    print("Explode dataset to resource level")
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
    print("Number of resources", merged.shape[0])
    return merged


def filter_resources(df, desired_formats=['table_data','geo_data']):
    """
    Filter df with resources for desired_formats (e.g. csv). 
    Be aware that the filtered column has to match the prefix defined in dataset_to_resource.
    returns a dict with desired_formats as keys and filtered dataframes as values
    
    """
    print("Filter data by:", desired_formats)
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
    data['publisher'] = data['author']

    # Reduce tags to tag names.
    data.tags = data.tags.apply(lambda x: [tag["name"] for tag in x])

    # keywords/groups
    data['keywords'] = data['groups'].apply(extract_keywords)

    return data


def prepare_data_for_codebooks(data):
    """Prepare metadata from catalogue in order to create the code files"""
    # Add new features to save prepared data.
    print("Preparation for codebook files")

    data["metadata"] = None
    data["contact"] = ""
    data["distributions"] = None
    data["distribution_links"] = None

    # Iterate over datasets and create additional data for markdown and code cells.
    for idx in data.index:
        md = [f"- **{k.capitalize()}** `{data.loc[idx, k]}`\n" for k in KEYS_DATASET]
        data.loc[idx, "metadata"] = "".join(md)


    data['description'] = data['notes']
    # Sort values for table.
    data.sort_values(by=['title', 'name', PREFIX_RESOURCE_COLS+"name"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data


def create_python_notebooks(data, notebook_template):
    """Create Jupyter Notebooks with Python starter code"""
    print("Creating", data.shape[0], "Python Notebook files with template:", notebook_template)
    for idx in data.index:
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
        
        url = f'[Direct link by {PROVIDER} for dataset]({BASELINK_DATAPORTAL}{data.loc[idx, "name"]})\n\n{data.loc[idx,PREFIX_RESOURCE_COLS+"url"]}'
        py_nb = py_nb.replace("{{ DATASHOP_LINK_PROVIDER }}", url)

        py_nb = py_nb.replace("{{ CONTACT }}", data.loc[idx, "maintainer_email"])

        file_url = data.loc[idx,PREFIX_RESOURCE_COLS+"url"]
        py_nb = py_nb.replace("{{ FILE_URL }}", file_url)

        py_nb = json.loads(py_nb, strict=False)
        # Save to disk.
        with open(
            f'{TEMP_PREFIX}{REPO_PYTHON_OUTPUT}{data.loc[idx, "name"]}_{data.loc[idx, PREFIX_RESOURCE_COLS+"id"]}.ipynb',
            "w",
            encoding="utf-8",
        ) as file:
            file.write(json.dumps(py_nb))


def create_rmarkdown(data, notebook_template):
    """Create R Markdown files with R starter code"""
    print("Creating", data.shape[0], "R Markdown files with template:", notebook_template)
    for idx in data.index:
        with open(f"{TEMPLATE_FOLDER}{notebook_template}", "r", encoding="utf-8") as file:
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
        
        url = f'[Direct link by **{PROVIDER}** for dataset]({BASELINK_DATAPORTAL}{data.loc[idx, "name"]})'
        rmd = rmd.replace("{{ DATASHOP_LINK_PROVIDER }}", url)

        # Get file URL and format
        file_url = data.loc[idx, PREFIX_RESOURCE_COLS + "url"]
        rmd = rmd.replace("{{ FILE_URL }}", file_url)

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
    baselink_py_renku = f"https://renkulab.io/p/{RENKU_NAMESPACE}/{RENKU_PROJECT_SLUG}/sessions/{RENKU_SESSION_ID}/"

    md_doc = []
    md_doc.append(header)
    md_doc.append(
        f"| Title (abbreviated to {TITLE_MAX_CHARS} chars) | Python Colab | Python Renku | Python GitHub | R GitHub | SQL Workbench | File |\n"
    )
    md_doc.append("| :-- | :-- | :-- | :-- | :-- | :-- | :-- |\n")

    for idx in data.index:
        # Remove square brackets from title, since these break markdown links.
        title_clean = (
            data.loc[idx, f"title"].replace("[", " ").replace("]", " ")
        )
        if len(title_clean) > TITLE_MAX_CHARS:
            title_clean = title_clean[:TITLE_MAX_CHARS] + "…"
        
        # filename is empty when format is json
        if isinstance(data.loc[idx, PREFIX_RESOURCE_COLS+"name"], str):
            resource_filename = data.loc[idx, PREFIX_RESOURCE_COLS+"name"]
        else:
            resource_filename = 'No filename provided'
        resource_format = f'{resource_filename}' # ({data.loc[idx, PREFIX_RESOURCE_COLS+"format"]})'
        
        ds_link = f'{BASELINK_DATAPORTAL}{data.loc[idx, "name"]}'
        filename = f'{data.loc[idx, "name"]}_{data.loc[idx, PREFIX_RESOURCE_COLS+"id"]}'#data.loc[idx, "id"]
        download_url = data.loc[idx,PREFIX_RESOURCE_COLS+'url'].replace("-","%20") # encoding for duckdb 
        package_name = data.loc[idx, 'name']

        r_gh_link = f"[R GitHub]({baselink_r_gh}{filename}.Rmd)"

        py_gh_link = f"[Python GitHub]({baselink_py_gh}{filename}.ipynb)"
        py_colab_link = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({baselink_py_colab}{filename}.ipynb)"
        py_binder_link = f"[![Jupyter Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{GITHUB_ACCOUNT}/{REPO_NAME}/{REPO_BRANCH}?filepath={REPO_PYTHON_OUTPUT}{filename}.ipynb)"
        py_renku_link = f"[![launch - renku](https://renkulab.io/renku-badge.svg)]({baselink_py_renku}start?PACKAGE_ID={data.loc[idx, 'name']}&RESOURCE_ID={data.loc[idx, PREFIX_RESOURCE_COLS+'id']})"
        
        fileformat = data.loc[idx,PREFIX_RESOURCE_COLS+'format'].lower()
        if fileformat in ('csv', 'parquet'):        
            sql_workbench_link = f"""[![SQL](https://img.shields.io/badge/SQL-grey?style=flat&logo=DuckDB&logoSize=auto&labelColor=grey&color=grey&link=https%3A%2F%2Fsql-workbench.com)](https://sql-workbench.com/#queries=v0,%20%20-This-is-autogenerated-code-to-analyze-(meta)-data-from-OpenDataZurich,%20%20-All-information-refers-to-the-data-set-that-can-be-found-here%3A-https%3A%2F%2Fdata.stadt%20zuerich.ch%2Fdataset%2F{package_name},%20%20-get-metadata,CREATE-TABLE-metadata-AS-SELECT-*-FROM-read_json(%22https%3A%2F%2Fdata.stadt%20zuerich.ch%2Fapi%2Faction%2Fpackage_show%3Fid%3D{package_name}%22)~,%20%20-use-this-to-display-metadata,%20%20SELECT-result.title%2C-result.timeRange%2C-result.notes%2C-result.sszBemerkungen-FROM-metadata~,%20%20-get-data,CREATE-TABLE-ogdszh-AS-SELECT-*-FROM-read_{fileformat}(%22https%3A%2F%2Fcors.sqlqry.run%2F%3Furl%3D{download_url}%22)~,%20%20-show-the-first-5-rows-of-the-data,SELECT-*-FROM-ogdszh-LIMIT-5~)"""
        else:
            sql_workbench_link = ''

        md_doc.append(
            f"| [{title_clean}]({ds_link}) | {py_colab_link} | {py_renku_link} | {py_gh_link} | {r_gh_link} | {sql_workbench_link} | {resource_format} |\n"
        )

    md_doc = "".join(md_doc)

    with open(f"{TEMP_PREFIX}index.md", "w", encoding="utf-8") as file:
        file.write(md_doc)


def update_ckan_metadata(text, ckan_field='description' ,env='int'):
    """
    Use CKAN API to update a field in the resource metadata
    """
    pass


def prepare_for_ckan(df):
    """
    Extract relevant data for metadata update in ckan, like:
    - resource ids
    - links to online tools like colab, binder, renku
    """
    subset_cols = ['name', PREFIX_RESOURCE_COLS+'id',PREFIX_RESOURCE_COLS+'package_id', PREFIX_RESOURCE_COLS+'url']
    df = df[subset_cols].copy()
    
    # badges with url links
    df['colab_url'] = "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/"+GITHUB_ACCOUNT+"/"+REPO_NAME+"/blob/"+REPO_BRANCH+"/"+REPO_PYTHON_OUTPUT+df['name']+"_"+df[PREFIX_RESOURCE_COLS+"id"]+".ipynb)"
    df['binder_py_url'] = "[![Jupyter Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/"+GITHUB_ACCOUNT+"/"+REPO_NAME+"/"+REPO_BRANCH+"?filepath="+REPO_PYTHON_OUTPUT+df['name']+"_"+df[PREFIX_RESOURCE_COLS+"id"]+".ipynb)"

    for index, row in df.iterrows():
        url_string = "Datensatz direkt online analysieren mit " + row['colab_url'] + " " + row['binder_py_url']
        print(url_string)
        #update_ckan_metadata(url_string)


# CREATE CODE FILES ---------------------------------------------------------- #

all_packages = get_full_package_list()

df = dataset_to_resource(all_packages)
df = clean_features(df)
df = filter_resources(df)
print("Number of resources", df.shape[0])
df = prepare_data_for_codebooks(df)



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

print("Create overview an readme files")
header = get_header(dataset_count=len(df))
create_readme(dataset_count=len(df))
create_overview(df, header)
