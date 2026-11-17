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

#the labeled dataset has no NIR, so the provinces of NIR use the region they belonged to before NIR
NIR_OLD_REGION = {
    "Negros Occidental": "Region VI",
    "Negros Oriental": "Region VII",
    "Siquijor": "Region VII",
}

#NCR has no provinces, the poverty tables use its 4 districts as the province level rows
#the income tables only have the cities of NCR, so each city is added to its district
NCR_DISTRICTS = {
    "manila city": "NCR 1st District",
    "mandaluyong city": "NCR 2nd District",
    "marikina city": "NCR 2nd District",
    "pasig city": "NCR 2nd District",
    "quezon city": "NCR 2nd District",
    "san juan city": "NCR 2nd District",
    "caloocan city": "NCR 3rd District",
    "malabon city": "NCR 3rd District",
    "navotas city": "NCR 3rd District",
    "valenzuela city": "NCR 3rd District",
    "las pinas city": "NCR 4th District",
    "makati city": "NCR 4th District",
    "muntinlupa city": "NCR 4th District",
    "paranaque city": "NCR 4th District",
    "pasay city": "NCR 4th District",
    "pateros": "NCR 4th District",
    "taguig city": "NCR 4th District",
}

#the table that decides which provinces are used and their order
PROVINCE_TABLE = "Table_2"

KEEP = [
    "Table_2 | Poverty Incidence among Population (%) 2023",
    "Table_2 | Poverty Incidence among Population (%) 2021",
    "Table_2 | Poverty Incidence among Population (%) 2018",
    "Table_12 | Estimate (%) 2023",
    "Table_6 | Estimate ('000) 2023",
    "Table_2 | Annual Per Capita Poverty Threshold (in PhP) 2023",
    "Total Family Size | Average Income",
    "Total Family Size | Average Expenditure",
    #the internet use dataset is by region only, so the share of family spending
    #that goes to information and communication is used for the provinces instead
    "Table_7 | Information and Communication",
]


def region_key(name):
    text = " ".join(name.split()).lower()
    for key, pattern in REGIONS:
        if pattern in text:
            return key

    return None #if given is province or city in data

"""
cleans the name of a province or city so that the same place has the same key in all of the data sets

examples:
"....Benguet 2/, a/"                    -> "benguet"
"BENGUET  (Excluding City of Baguio)"   -> "benguet"
"CITY OF COTABATO" and "Cotabato City"  -> "cotabato city"
"""
def province_key(name):
    #some files have a broken ñ
    text = name.replace("ñ", "n").replace("Ñ", "N").replace("�", "n")
    #remove the notes like (Excluding City of Baguio) or (w/o the City of Baguio)
    text = re.sub(r"\(.*", "", text)
    #remove the footnote marks like 1/ a/ r1 ** and the commas between them
    text = re.sub(r"\d/|\b[a-z]/|\br\d\b|\*|,", " ", text)
    #remove the dots at the start that show the level of the row
    text = " ".join(text.split()).strip(" .").lower()

    if text == "mt. province":
        text = "mountain province"
    #city of cotabato and cotabato city should be the same
    m = re.match(r"city of (.+)", text)
    if m:
        text = m.group(1) + " city"
    return text

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

def read_rows(file_path):
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        return list(csv.reader(file))

"""
gets the list of provinces from the poverty incidence table (Table 2)

the rows of the table have dots at the start of the name to show the level
"..Region" = 2 dots, "....Province" = 4 dots, "......part of a province" = 6 dots
only the rows with 4 dots are used. this means Maguindanao is one row (del Norte and del Sur have no data for 2018 and 2021)

return:
- provinces = list of province names, same order as the table
- keys = dictionary of cleaned name -> province name, used to find the province in the other data sets
- province_regions = dictionary of province name -> region it belongs to (same names as the labeled dataset)
"""
def read_provinces(folder_path):
    provinces = []
    keys = {}
    province_regions = {}

    for file_path in sorted(folder_path.glob("*.csv")):
        if make_tag(file_path.name) != PROVINCE_TABLE:
            continue

        region = None
        for row in read_rows(file_path)[2:]:
            if not row or not row[0].strip():
                continue
            name = row[0]
            dots = len(name) - len(name.lstrip("."))

            if dots == 2:
                region = region_key(name)
            elif dots == 4 and region is not None:
                key = province_key(name)
                #same as the key but with the capital letters of the data set
                province = " ".join(re.sub(r"\d/|\b[a-z]/|\br\d\b|\*|,", " ", name).split()).strip(" .")
                if region == "NCR":
                    province = "NCR " + province

                provinces.append(province)
                keys[key] = province
                if region == "NIR":
                    province_regions[province] = NIR_OLD_REGION[province]
                else:
                    province_regions[province] = region

    if len(provinces) == 0:
        print("province table not found: " + PROVINCE_TABLE)
        exit(1)

    return provinces, keys, province_regions

"""
combines the rows of a province and its highly urbanized cities (HUC) into one row

the income tables (Table 7, 10 and 11) have the HUCs as separate rows, and the row of the province does not include them
e.g. "BENGUET (Excluding City of Baguio)" and "..City of Baguio". the poverty tables used here include the HUCs in the province,
so the rows are combined to have the same meaning of province in all of the fields

args:
- tag = which table the rows are from
- parts = list of rows (list of values) that belong to the same province

return:
- one row of values for the whole province
"""
def combine_parts(tag, parts):
    if len(parts) == 1:
        return parts[0]

    #a part with a missing value cannot be added
    complete = []
    for part in parts:
        if None not in part:
            complete.append(part)
    if len(complete) == 0:
        return [None] * len(parts[0])

    if tag == "Table_10":
        #columns: total income (millions), average income (thousands), total expenditure (millions), average expenditure (thousands)
        #number of families = total / average, so the average of the whole province = sum of totals / sum of families
        total_income, total_expenditure, families = 0, 0, 0
        for part in complete:
            total_income += part[0]
            total_expenditure += part[2]
            families += part[0] / part[1]
        return [round(total_income, 2), round(total_income / families, 2), round(total_expenditure, 2), round(total_expenditure / families, 2)]

    if tag == "Table_11":
        #columns: total income (millions), income (percent), total expenditure (millions), expenditure (percent)
        #the percents are computed again in read_income_table after all of the deciles are added
        total_income, total_expenditure = 0, 0
        for part in complete:
            total_income += part[0]
            total_expenditure += part[2]
        return [round(total_income, 2), None, round(total_expenditure, 2), None]

    if tag == "Table_7":
        #columns: total expenditure (millions), information and communication (percent of total expenditure)
        total_expenditure, communication = 0, 0
        for part in complete:
            total_expenditure += part[0]
            communication += part[0] * part[1] / 100
        return [round(total_expenditure, 2), round(communication / total_expenditure * 100, 2)]

    #do not know how to combine the columns of other tables
    return [None] * len(parts[0])

"""
reads an income table (by Region, Province, and HUC) and stores the values per province

rows of the table are in this order: region, then its provinces, each province is followed by its HUCs.
- region rows are skipped
- HUC rows are added to the province before it, the cities of NCR are added to their district
- Maguindanao del Norte / del Sur rows are skipped because the MAGUINDANAO row already has both
"""
def read_income_table(tag, rows, keys, table, field_names):
    fields = rows[1]

    #check whether the second column is a label (family size or decile)
    second = rows[2][1].strip()
    checker = to_float(second) is None and second not in ("", "..", "-", "...")
    if checker:
        first_field_index = 2
    else:
        first_field_index = 1

    #parts[province][label] = list of rows that belong to the province
    parts = {}
    labels = []
    region = None
    current_province = None

    for row in rows[2:]:
        if not row or not row[0].strip():
            continue
        name = row[0]
        key = province_key(name)

        if key == "philippines":
            continue
        if region_key(name) is not None:
            region = region_key(name)
            current_province = None
            continue
        if key.startswith("maguindanao del"):
            continue

        if key in keys:
            owner = keys[key]
            #Isabela City and Cotabato City are their own rows, HUCs after them do not belong to them
            if not key.endswith(" city"):
                current_province = owner
        elif region == "NCR":
            owner = NCR_DISTRICTS.get(key)
        else:
            owner = current_province

        if owner is None:
            print("no province found for: " + name)
            continue

        label = ""
        if checker:
            label = row[1].strip()
        if label not in labels:
            labels.append(label)

        values = []
        for j in range(first_field_index, len(fields)):
            if j < len(row):
                values.append(to_float(row[j]))
            else:
                values.append(None)

        if owner not in parts:
            parts[owner] = {}
        if label not in parts[owner]:
            parts[owner][label] = []
        parts[owner][label].append(values)

    for owner in parts:
        combined = {}
        for label in parts[owner]:
            combined[label] = combine_parts(tag, parts[owner][label])

        #percent of each decile = total of the decile / total of all income groups
        if tag == "Table_11" and "All Income Groups" in combined:
            all_groups = combined["All Income Groups"]
            for label in combined:
                for j in (0, 2):
                    if combined[label][j + 1] is None and combined[label][j] is not None and all_groups[j]:
                        combined[label][j + 1] = round(combined[label][j] / all_groups[j] * 100, 2)

        for label in labels:
            for j in range(first_field_index, len(fields)):
                if checker:
                    field = tag + " | " + label + " | " + fields[j].strip()
                else:
                    field = tag + " | " + fields[j].strip()
                if field not in field_names:
                    field_names.append(field)
                if label in combined:
                    table[owner][field] = combined[label][j - first_field_index]

"""
reads a poverty table (by Region and Province) and stores the values per province
rows that are not one of the provinces (regions, parts of a province) are skipped
"""
def read_poverty_table(tag, rows, keys, table, field_names):
    fields = rows[1]

    for row in rows[2:]:
        if not row:
            continue
        key = province_key(row[0])
        if key not in keys:
            continue
        province = keys[key]

        for j in range(1, min(len(fields), len(row))):
            field = tag + " | " + fields[j].strip()
            if field not in field_names:
                field_names.append(field)
            table[province][field] = to_float(row[j])

"""
reads all of the data sets in the folder, one row per province

return:
- provinces = list of province names
- table = table[province][field name] = value
- field_names = list of all the field names that were read
- province_regions = dictionary of province name -> region it belongs to
"""
def read_files(folder_path):
    provinces, keys, province_regions = read_provinces(folder_path)

    table = {}
    for province in provinces:
        table[province] = {}
    field_names = []

    for file_path in sorted(folder_path.glob("*.csv")):
        tag = make_tag(file_path.name)

        #not a table (the internet use data set is by region only, and the labeled dataset is read by the models)
        if tag == "Internet":
            continue

        rows = read_rows(file_path)
        #row[0] is the table name so skip
        first_field = " ".join(rows[1][0].split()).lower()

        if first_field.startswith("region, province, and huc"):
            read_income_table(tag, rows, keys, table, field_names)
        elif tag.endswith("a"):
            #the "a" poverty tables (1a, 2a, 5a, 6a, 12a) remove the HUCs from the provinces,
            #the tables without "a" are used so that the provinces include their HUCs
            continue
        else:
            read_poverty_table(tag, rows, keys, table, field_names)

    return provinces, table, field_names, province_regions

def find_column(field_names, name):
    wanted = "".join(name.split()).lower()
    matches = []
    for field in field_names:
        if wanted in "".join(field.split()).lower():
            matches.append(field)

    if len(matches) == 1:
        return matches[0]
    elif len(matches) == 0:
        print("field not found: " + name)
        exit(1)
    else:
        print("multiple fields found: " + name)
        exit(1)

def get_values(table, field_names, provinces, name):
    field = find_column(field_names, name)
    values = []
    for province in provinces:
        #None when the province has no value for the field
        values.append(table[province].get(field))
    return values

#build matrix with the specified fields, one row per province
def build_matrix(provinces, table, field_names):
    fields = []
    #2d list, each row = values per field, inside row values per province, field_values[field_idx][province_idx]
    field_values = []

    for name in KEEP:
        fields.append(name)
        field_values.append(get_values(table, field_names, provinces, name))

    inc_2023 = get_values(table, field_names, provinces, "Table_2 | Poverty Incidence among Population (%) 2023")
    inc_2021 = get_values(table, field_names, provinces, "Table_2 | Poverty Incidence among Population (%) 2021")
    inc_2018 = get_values(table, field_names, provinces, "Table_2 | Poverty Incidence among Population (%) 2018")

    #change in poverty in percentage, when negative then poverty went down
    change_2021 = []
    change_2018 = []
    for i in range(len(provinces)):
        if inc_2023[i] is None or inc_2021[i] is None:
            change_2021.append(None)
        else:
            change_2021.append(round(inc_2023[i] - inc_2021[i], 2))
        if inc_2023[i] is None or inc_2018[i] is None:
            change_2018.append(None)
        else:
            change_2018.append(round(inc_2023[i] - inc_2018[i], 2))
    fields.append("DERIVED | Change in poverty incidence 2021 to 2023 (percentage points)")
    field_values.append(change_2021)
    fields.append("DERIVED | Change in poverty incidence 2018 to 2023 (percentage points)")
    field_values.append(change_2018)

    #income inequality = income share of the richest 10% / income share of the poorest 40% from palma ratio
    top_share = get_values(table, field_names, provinces, "Table_11 | Tenth Decile | Total Family Income (percent)")
    bottom_share = []
    for i in range(len(provinces)):
        bottom_share.append(0)
    for decile in ["First Decile", "Second Decile", "Third Decile", "Fourth Decile"]:
        share = get_values(table, field_names, provinces, "Table_11 | " + decile + " | Total Family Income (percent)")
        for i in range(len(provinces)):
            if share[i] is None or bottom_share[i] is None:
                bottom_share[i] = None
            else:
                bottom_share[i] += share[i]
    inequality = []
    for i in range(len(provinces)):
        if top_share[i] is None or not bottom_share[i]:
            inequality.append(None)
        else:
            inequality.append(round(top_share[i] / bottom_share[i], 2))
    fields.append("DERIVED | Income inequality (top 10% share / bottom 40% share)")
    field_values.append(inequality)

    matrix = [] #rows = provinces, columns = fields
    for i in range(len(provinces)):
        row = []
        for k in range(len(fields)):
            row.append(field_values[k][i])
        matrix.append(row)

    return provinces, fields, matrix

def printMatrix(provinces, fields, matrix, first_field=0, max_fields=None, max_provinces=None):
    last_field = len(fields)
    if max_fields is not None:
        last_field = min(len(fields), first_field + max_fields)
    last_province = len(provinces)
    if max_provinces is not None:
        last_province = min(len(provinces), max_provinces)

    print(len(provinces), "provinces x", len(fields), "fields")
    print("showing fields", first_field, "to", last_field - 1, "\n")

    #legend, which field is in which column
    for j in range(first_field, last_field):
        print("  [" + str(j) + "]", fields[j])
    print()

    #field name row = field numbers, then one row per province
    print("Province".ljust(22), end="")
    for j in range(first_field, last_field):
        print(("[" + str(j) + "]").rjust(12), end="")
    print()
    for i in range(last_province):
        print(provinces[i].ljust(22), end="")
        for j in range(first_field, last_field):
            value = matrix[i][j]
            if value is None:
                value = "NA"
            print(str(value).rjust(12), end="")
        print()

def createMatrix(folder_path):
    provinces, table, field_names, province_regions = read_files(folder_path)
    return build_matrix(provinces, table, field_names)

#region of every province, used to get the class of a province from the labeled dataset (which is by region)
def getProvinceRegions(folder_path):
    provinces, keys, province_regions = read_provinces(folder_path)
    return province_regions

if __name__ == "__main__":
    current_dir_parent = Path(__file__).resolve().parent.parent
    data_set_dir = current_dir_parent / "PROJECT_DATASET"

    provinces, fields, matrix = createMatrix(data_set_dir)
    printMatrix(provinces, fields, matrix)
