"""
creates the labeled dataset of the provinces: PROJECT_DATASET/province_policy_results.csv

the rules are the theory-based policies in the "Policy for Labeling Data" section of the project document.
every policy tags a province as INCREASE, RETAIN or DECREASE, and the Class is the tag that most of the policies gave.
when two or three tags are tied, the class is RETAIN.

- a province is always compared against the PHILIPPINES row of the same column
- the poverty and income features are by province (same values as methods/read_data.py)
- the internet use data set is by region only, so a province uses the internet values of its region

how to run (from the project folder):
    python3 -m methods.label_data
"""
from pathlib import Path
import csv

from methods.read_data import read_files, read_rows, make_tag, region_key, province_key, to_float

LABELS_FILE = "province_policy_results.csv"

#same columns as region_policy_results.csv
POLICIES = [
    "Lewis (1954)",
    "Myrdal (1957)",
    "Schultz (1961) Human Capital",
    "Layne & Lee (2001) E-Government",
    "Roller & Waverman (2001) Critical Mass",
    "van Dijk (2005) Resources & Appropriation",
    "Sen (1999) Capability Approach",
]

#short name -> part of the column name in the internet use data set (the Percentage column is used)
INTERNET_COLUMNS = {
    "health_info": "Seeking health-related information",
    "gov_info": "Getting information from general government organizations",
    "email": "Sending or receiving e-mail",
    "calls": "Making calls via the internet",
    "social": "Participating in social networks",
    "health_appointment": "Making an appointment with a health practitioner",
    "gov_interact": "Interacting with government organizations",
    "online_class": "Attending an online class/online course",
    "wiki": "Consulting websites for formal or informal learning",
    "job": "Looking for a job or sending/submitting a job application",
    "professional": "Participating in professional networks",
    "web_radio": "Listening to web radio",
    "web_tv": "Watching web television",
    "streaming": "Streaming or downloading web images",
    "games": "Playing or downloading games",
    "cloud": "Using storage space on the internet",
    "office": "Using software that runs over the internet",
}

def mean(values):
    return sum(values) / len(values)

"""
reads the internet use data set (by region)

return:
- dictionary of region -> dictionary of short name -> percentage. the national row has the key "PHILIPPINES"
"""
def read_internet(folder_path):
    for file_path in sorted(folder_path.glob("*.csv")):
        rows = read_rows(file_path)
        if len(rows) < 3 or rows[1][0].strip() != "Region":
            continue
        fields = rows[1]

        #find the percentage column of every internet use
        columns = {}
        for short, part in INTERNET_COLUMNS.items():
            for j in range(len(fields)):
                if part in fields[j] and fields[j].strip().endswith("Percentage"):
                    columns[short] = j
            if short not in columns:
                print("internet field not found: " + part)
                exit(1)

        internet = {}
        for row in rows[2:]:
            if not row or not row[0].strip():
                continue
            if province_key(row[0]) == "philippines":
                key = "PHILIPPINES"
            else:
                key = region_key(row[0])
            if key is None:
                continue
            internet[key] = {}
            for short, j in columns.items():
                internet[key][short] = to_float(row[j])
        return internet

    print("internet use data set not found")
    exit(1)

"""
reads the values of one row (e.g. the PHILIPPINES row) from the poverty and income tables

args:
- folder_path = folder of the dataset csv files
- wanted = function that returns True when the name in the first column is the row that is needed

return:
- dictionary of field name -> value, the field names are the same as in read_files of methods/read_data.py
"""
def read_area_row(folder_path, wanted):
    values = {}
    for file_path in sorted(folder_path.glob("*.csv")):
        tag = make_tag(file_path.name)
        if tag == "Internet" or tag.endswith("a"):
            continue
        rows = read_rows(file_path)
        fields = rows[1]

        #check whether the second column is a label (family size or decile)
        second = rows[2][1].strip()
        checker = to_float(second) is None and second not in ("", "..", "-", "...")

        for row in rows[2:]:
            if not row or not wanted(row[0]):
                continue
            if checker:
                for j in range(2, min(len(fields), len(row))):
                    values[tag + " | " + row[1].strip() + " | " + fields[j].strip()] = to_float(row[j])
            else:
                for j in range(1, min(len(fields), len(row))):
                    values[tag + " | " + fields[j].strip()] = to_float(row[j])
    return values

"""
computes the features that the policies use

args:
- values = dictionary of field name -> value of the province (or of the PHILIPPINES row)
- internet = dictionary of short name -> percentage, the internet use of the region of the province

return:
- dictionary of feature name -> value
"""
def get_features(values, internet):
    f = {}
    f["poverty"] = values["Table_2 | Poverty Incidence among Population (%) 2023"]
    poverty_2018 = values["Table_2 | Poverty Incidence among Population (%) 2018"]
    #difference of the poverty from 2018 to 2023, when negative then poverty went down
    f["change"] = round(f["poverty"] - poverty_2018, 2)

    #"-" in the severity of poverty is a value too small to display, so it is 0
    f["severity"] = values["Table_12 | Estimate (%) 2023"]
    if f["severity"] is None:
        f["severity"] = 0

    threshold = values["Table_2 | Annual Per Capita Poverty Threshold (in PhP) 2023"]
    f["income"] = values["Table_10 | Total Family Size | Average Income  (in Thousands)"]
    expenditure = values["Table_10 | Total Family Size | Average Expenditure  (in Thousands)"]
    #subsistence multiple = average income / poverty threshold
    f["subsistence"] = f["income"] * 1000 / threshold
    #savings rate = (income - expenditure) / income
    f["savings"] = (f["income"] - expenditure) / f["income"]

    #inequality ratio = income share of the richest 10% / income share of the poorest 40%
    bottom_share = 0
    for decile in ["First Decile", "Second Decile", "Third Decile", "Fourth Decile"]:
        bottom_share += values["Table_11 | " + decile + " | Total Family Income (percent)"]
    f["inequality"] = round(values["Table_11 | Tenth Decile | Total Family Income (percent)"] / bottom_share, 2)

    #indices from the internet use data set
    i = internet
    f["job"] = i["job"]                                             #online job search
    f["E"] = mean([i["online_class"], i["wiki"]])                   #education investment
    f["R"] = mean([i["job"], i["professional"]])                    #labor-market return
    f["GI"] = i["gov_info"]                                         #govt. information
    f["GX"] = i["gov_interact"]                                     #govt. transaction
    f["C"] = mean([i["calls"], i["social"], i["email"]])            #basic connectivity
    f["H"] = mean([i["health_info"], i["health_appointment"]])      #health access
    #usage-gap ratio = capital-enhancing uses / entertainment uses
    capital = mean([i["online_class"], i["wiki"], i["job"], i["professional"], i["gov_interact"], i["health_appointment"], i["cloud"], i["office"]])
    entertainment = mean([i["streaming"], i["games"], i["web_tv"], i["web_radio"]])
    f["U"] = capital / entertainment
    return f

"""
tags one province under each of the 7 policies

args:
- f = features of the province
- n = features of the PHILIPPINES row (national benchmark)

return:
- list of 7 tags (INCREASE, RETAIN or DECREASE), same order as POLICIES
"""
def get_tags(f, n):
    tags = []

    #Lewis (1954), dual-sector / surplus labor
    #INCREASE = poverty above national and subsistence multiple below national and (savings rate below national or online job search below national)
    #DECREASE = online job search at or above national and subsistence multiple at or above national
    if f["poverty"] > n["poverty"] and f["subsistence"] < n["subsistence"] and (f["savings"] < n["savings"] or f["job"] < n["job"]):
        tags.append("INCREASE")
    elif f["job"] >= n["job"] and f["subsistence"] >= n["subsistence"]:
        tags.append("DECREASE")
    else:
        tags.append("RETAIN")

    #Myrdal (1957), circular and cumulative causation
    #M = signs of a vicious circle
    M = 0
    if f["change"] > n["change"]:           #diverging, poverty fell slower than nationally
        M += 1
    if f["severity"] > n["severity"]:
        M += 1
    if f["inequality"] > n["inequality"]:
        M += 1
    if f["GX"] < n["GX"]:
        M += 1
    #DECREASE = poverty below national and average income above national and poverty fell faster than nationally
    if f["poverty"] < n["poverty"] and f["income"] > n["income"] and f["change"] < n["change"]:
        tags.append("DECREASE")
    elif f["poverty"] >= n["poverty"] and M >= 2:
        tags.append("INCREASE")
    else:
        tags.append("RETAIN")

    #Schultz (1961), human capital
    if f["E"] >= n["E"] and f["R"] >= n["R"]:
        tags.append("DECREASE")
    elif f["E"] < n["E"] and f["poverty"] > n["poverty"]:
        tags.append("INCREASE")
    else:
        tags.append("RETAIN")

    #Layne & Lee (2001), e-government stages
    if f["GI"] >= n["GI"] and f["GX"] >= n["GX"]:
        tags.append("DECREASE")
    elif f["GI"] < n["GI"] and f["GX"] < n["GX"]:
        tags.append("INCREASE")
    else:
        tags.append("RETAIN")

    #Roller & Waverman (2001), telecom critical mass
    if f["C"] >= n["C"] and f["poverty"] < n["poverty"]:
        tags.append("DECREASE")
    elif f["C"] < n["C"]:
        tags.append("INCREASE")
    else:
        tags.append("RETAIN")

    #van Dijk (2005), resources and appropriation
    if f["U"] >= n["U"] and f["C"] >= n["C"]:
        tags.append("DECREASE")
    elif f["U"] < n["U"] and f["C"] < n["C"]:
        tags.append("INCREASE")
    else:
        tags.append("RETAIN")

    #Sen (1999), capability approach. D = number of deprivations
    D = 0
    if f["E"] < n["E"]:
        D += 1
    if f["H"] < n["H"]:
        D += 1
    if f["income"] < n["income"]:
        D += 1
    if f["GX"] < n["GX"]:
        D += 1
    if f["severity"] > n["severity"]:
        D += 1
    if D <= 1:
        tags.append("DECREASE")
    elif D >= 4:
        tags.append("INCREASE")
    else:
        tags.append("RETAIN")

    return tags

#the class is the tag that most of the policies gave. when two or three tags are tied, the class is RETAIN
def get_class(tags):
    counts = {}
    for word in ["INCREASE", "RETAIN", "DECREASE"]:
        counts[word] = tags.count(word)
    highest = max(counts.values())
    winners = []
    for word in counts:
        if counts[word] == highest:
            winners.append(word)
    if len(winners) == 1:
        return winners[0]
    return "RETAIN"

"""
tags every province and its class

return:
- list of rows [province, tag of policy 1, ..., tag of policy 7, class], same order as the provinces in read_files
"""
def label_provinces(folder_path):
    provinces, table, field_names, province_regions = read_files(folder_path)
    internet = read_internet(folder_path)

    national_values = read_area_row(folder_path, lambda name: province_key(name) == "philippines")
    national = get_features(national_values, internet["PHILIPPINES"])

    rows = []
    for province in provinces:
        #the internet use of a province is the internet use of its region
        features = get_features(table[province], internet[province_regions[province]])
        tags = get_tags(features, national)
        rows.append([province] + tags + [get_class(tags)])
    return rows

def write_labels(folder_path):
    rows = label_provinces(folder_path)
    labels_path = folder_path / LABELS_FILE
    with open(labels_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Province"] + POLICIES + ["Class"])
        writer.writerows(rows)
    return labels_path, rows

if __name__ == "__main__":
    data_set_dir = Path(__file__).resolve().parent.parent / "PROJECT_DATASET"
    labels_path, rows = write_labels(data_set_dir)

    print("saved", len(rows), "provinces to", labels_path)
    for word in ["INCREASE", "RETAIN", "DECREASE"]:
        count = 0
        for row in rows:
            if row[-1] == word:
                count += 1
        print(word + ":", count)
