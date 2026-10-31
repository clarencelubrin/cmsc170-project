from pathlib import Path
import csv
import re

REGIONS = [
    ("NCR", "national capital"),
    ("CAR", "cordillera"),
    ("Region I", "ilocos region"),
    ("Region II", "cagayan valley"),
    ("Region III", "central luzon"),
    ("Region IV-A", "calabarzon"),
    ("MIMAROPA", "mimaropa"),
    ("Region V", "bicol region"),
    ("Region VI", "western visayas"),
    ("NIR", "negros island"),
    ("Region VII", "central visayas"),
    ("Region VIII", "eastern visayas"),
    ("Region IX", "zamboanga peninsula"),
    ("Region X", "northern mindanao"),
    ("Region XI", "davao region"),
    ("Region XII", "soccsksargen"),
    ("Region XIII", "caraga"),
    ("BARMM", "bangsamoro"),
]

KEEP = [
    "Table_2 | Poverty Incidence among Population (%) 2023",
    "Table_2 | Poverty Incidence among Population (%) 2021",
    "Table_2 | Poverty Incidence among Population (%) 2018",
    "Table_12 | Estimate (%) 2023",
    "Table_6a | Estimate ('000) 2023",
    "Table_2 | Annual Per Capita Poverty Threshold (in PhP) 2023",
    "Total Family Size | Average Income",
    "Total Family Size | Average Expenditure",
    "Getting information about goods or services Percentage",
    "Getting information from general government organizations Percentage",
    "Interacting with government organizations Percentage",
    "Seeking health-related information Percentage",
    "Making an appointment with a health practitioner via the internet Percentage",
    "Attending an online class/online course (on any subject) Percentage",
    "online encyclopedias Percentage",
    "websites for a job Percentage",
]


def region_key(name):
    text = " ".join(name.split()).lower()
    for key, pattern in REGIONS:
        if pattern in text:
            return key

    return None #if given is province or city in data

def to_float(text):
    #for those in the thousands
    text = text.strip().replace(",", "")
    #remove missing data
    if text in ("", "..", "-", "..."):
        return None
    try:
        return float(text)
    except ValueError:
        return None

def make_tag(file_name):
    #(?i) = case insensitive, ?! next char not a letter or number, so that table2a is not matched
    m = re.match(r"(?i)table[\s_]*(\d+[a-z]?)(?![a-z0-9])", file_name)
    if m:
        return "Table_" + m.group(1).lower()
    return "Internet"

def read_files(folder_path):
    table  = {} 
    field_names = [] 

    for file_path in sorted(folder_path.glob("*.csv")):
        tag = make_tag(file_path.name)
        
        with open(file_path, "r", encoding="utf-8") as file:
            rows = list(csv.reader(file))

        #row[0] is the table name so skip
        fields = rows[1]

        #check where the first field value starts in the specific data set
        second = rows[2][1].strip()
        #check whether the second column is a label
        checker = to_float(second) is None and second not in ("", "..", "-", "...")
        if checker:
            first_field_index = 2
        else:
            first_field_index = 1

        #used to prevent duplicates as there are data sets with rows for same region
        first_region = None

        for row in rows[2:]:
            region = region_key(row[0])
            
            if region is None:
                continue

            if first_region is None:
                first_region = region

            if region not in table:
                table[region] = []

            #iterate the columns
            for j in range(first_field_index, min(len(fields), len(row))):
                table[region].append(to_float(row[j]))

                #check if still the same region, cuz in some data set, same region multiple rows
                if region == first_region:  
                    #check if second col is label
                    if checker:
                        field_names.append(tag + " | " + row[1].strip() + " | " + fields[j].strip())
                    else:
                        field_names.append(tag + " | " + fields[j].strip())
 
    return table, field_names

def find_column(field_names, name):
    wanted = "".join(name.split()).lower()
    matches_idx = []
    for idx in range(len(field_names)):
        if wanted in "".join(field_names[idx].split()).lower():
            matches_idx.append(idx)

    if len(matches_idx) == 1:
        return matches_idx[0]
    elif len(matches_idx) == 0:
        print("field not found: " + name)
        exit(1)
    else:
        found = []
        for idx in matches_idx:
            found.append(field_names[idx])
        print("multiple fields found")
        exit(1)

def get_values(table, field_names, regions, name):
    col_idx = find_column(field_names, name)
    values = []
    for region in regions:
        values.append(table[region][col_idx])
    return values

#build matrix with the specified fields and regions
def build_matrix(table, field_names):
    regions = []
    for key, pattern in REGIONS:
        if key in table and key not in ("PHILIPPINES", "NIR"):
            regions.append(key)

    fields = []
    #2d list, each row = values per field, inside row values per region, field_values[field_idx][region_idx]
    field_values = []

    for name in KEEP:
        fields.append(name)
        field_values.append(get_values(table, field_names, regions, name))

    inc_2023 = get_values(table, field_names, regions, "Table_2 | Poverty Incidence among Population (%) 2023")
    inc_2021 = get_values(table, field_names, regions, "Table_2 | Poverty Incidence among Population (%) 2021")
    inc_2018 = get_values(table, field_names, regions, "Table_2 | Poverty Incidence among Population (%) 2018")

    #change in poverty in percentage, when negative then poverty went down
    change_2021 = []
    change_2018 = []
    for i in range(len(regions)):
        change_2021.append(round(inc_2023[i] - inc_2021[i], 2))
        change_2018.append(round(inc_2023[i] - inc_2018[i], 2))
    fields.append("DERIVED | Change in poverty incidence 2021 to 2023 (percentage points)")
    field_values.append(change_2021)
    fields.append("DERIVED | Change in poverty incidence 2018 to 2023 (percentage points)")
    field_values.append(change_2018)

    #income inequality = income share of the richest 10% / income share of the poorest 40% from palma ratio
    top_share = get_values(table, field_names, regions, "Table_11 | Tenth Decile | Total Family Income (percent)")
    bottom_share = []
    for i in range(len(regions)):
        bottom_share.append(0)
    for decile in ["First Decile", "Second Decile", "Third Decile", "Fourth Decile"]:
        share = get_values(table, field_names, regions, "Table_11 | " + decile + " | Total Family Income (percent)")
        for i in range(len(regions)):
            bottom_share[i] += share[i]
    inequality = []
    for i in range(len(regions)):
        inequality.append(round(top_share[i] / bottom_share[i], 2))
    fields.append("DERIVED | Income inequality (top 10% share / bottom 40% share)")
    field_values.append(inequality)

    matrix = [] #rows = regions, columns = fields
    for i in range(len(regions)):
        row = []
        for k in range(len(fields)):
            row.append(field_values[k][i])
        matrix.append(row)

    return regions, fields, matrix

def printMatrix(regions, fields, matrix, first_field=0, max_fields=None, max_regions=None):
    last_field = len(fields)
    if max_fields is not None:
        last_field = min(len(fields), first_field + max_fields)
    last_region = len(regions)
    if max_regions is not None:
        last_region = min(len(regions), max_regions)

    print(len(regions), "regions x", len(fields), "fields")
    print("showing fields", first_field, "to", last_field - 1, "\n")

    #legend, which field is in which column
    for j in range(first_field, last_field):
        print("  [" + str(j) + "]", fields[j])
    print()

    #field name row = field numbers, then one row per region
    print("Region".ljust(13), end="")
    for j in range(first_field, last_field):
        print(("[" + str(j) + "]").rjust(12), end="")
    print()
    for i in range(last_region):
        print(regions[i].ljust(13), end="")
        for j in range(first_field, last_field):
            value = matrix[i][j]
            if value is None:
                value = "NA"
            print(str(value).rjust(12), end="")
        print()

def createMatrix(folder_path):
    table, field_names = read_files(folder_path)
    return build_matrix(table, field_names)

# current_dir_parent = Path(__file__).resolve().parent.parent
# data_set_dir = current_dir_parent / "PROJECT_DATASET"

# regions, fields, matrix = createMatrix(data_set_dir)
# printMatrix(regions, fields, matrix, max_regions=18, max_fields=20)