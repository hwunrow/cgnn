import pandas as pd

ZIP_CBSA_PATH = "/burg/apam/users/nhw2114/repos/cgnn/data/raw/ZIP_CBSA_122024.csv"
ZIP_COUNTY_PATH = "/burg/apam/users/nhw2114/repos/cgnn/data/raw/ZIP_COUNTY_122024.csv"
TRACT_ZIP_PATH = "/burg/apam/users/nhw2114/repos/cgnn/data/raw/TRACT_ZIP_122024.csv"


def get_zip_cbsa_map():
    """
    Creates a mapping between ZIP codes and Core-Based Statistical Areas (CBSAs).
    """
    DTYPE = {"ZIP": "str", "CBSA": "str"}
    zip_cbsa_map = pd.read_csv(ZIP_CBSA_PATH, dtype=DTYPE)
    zip_cbsa_map = zip_cbsa_map.sort_values(
        by=["ZIP", "RES_RATIO"], ascending=[True, False]
    )
    zip_cbsa_map = zip_cbsa_map.drop_duplicates(subset="ZIP", keep="first")

    manual_zip_cbsa_map = {
        "08625": "45940",
        "15123": "38300",
        "15130": "38300",
        "20307": "47900",
        "27695": "39580",
        "45221": "17140",
        "49355": "24340",
        "70813": "12940",
        "70837": "12940",
        "71611": "38220",
        "94101": "41860",
        "94199": "41860",
        "99950": "28540",
        "14650": "40380",
        "01601": "49340",
    }
    zip_cbsa_map = pd.concat(
        [
            zip_cbsa_map,
            pd.DataFrame(manual_zip_cbsa_map.items(), columns=["ZIP", "CBSA"]),
        ],
        ignore_index=True,
    )

    assert zip_cbsa_map["ZIP"].nunique() == zip_cbsa_map.shape[0]
    return zip_cbsa_map[["ZIP", "CBSA"]]


def get_tract_zip_map():
    """
    Creates a mapping between census TRACT and ZIP code.

    This function reads TRACT to ZIP xwalk file from HUDS,
    processes the data to ensure unique mappings, and resolves any conflicts
    where a TRACTS is mapped to multiple ZIPs by selecting the ZIP with the
    highest residential ratio (RES_RATIO).

    Returns:
        pandas.DataFrame: A DataFrame with columns 'ZIP' and 'TRACT', representing
        the mapping between ZIP codes and census tracts.
    """
    DTYPE = {"ZIP": "str", "TRACT": "str"}
    tract_zip_map = pd.read_csv(TRACT_ZIP_PATH, dtype=DTYPE)
    tract_zip_map = tract_zip_map.sort_values(
        by=["TRACT", "RES_RATIO"], ascending=[True, False]
    )
    tract_zip_map = tract_zip_map.drop_duplicates(subset="TRACT", keep="first")

    assert tract_zip_map["TRACT"].nunique() == tract_zip_map.shape[0]
    return tract_zip_map[["TRACT", "ZIP"]]


def get_county_cbsa_map():
    """
    Creates a mapping between counties and Core-Based Statistical Areas (CBSAs).

    This function reads ZIP to CBSA and ZIP to COUNTY mapping data from HUDS,
    processes the data to ensure unique mappings, and resolves any conflicts where
    a COUNTY is mapped to multiple CBSAs by selecting the CBSA with the
    highest residential ratio (RES_RATIO).

    Returns:
        pandas.DataFrame: A DataFrame with columns 'COUNTY' and 'CBSA', representing
        the unique mapping between counties and CBSAs.
    """
    zip_cbsa_map = get_zip_cbsa_map()

    DTYPE = {"ZIP": "str", "COUNTY": "str"}
    zip_county_map = pd.read_csv(ZIP_COUNTY_PATH, dtype=DTYPE)
    zip_county_map = zip_county_map.sort_values(
        by=["ZIP", "RES_RATIO"], ascending=[True, False]
    )
    zip_county_map = zip_county_map.drop_duplicates(subset="ZIP", keep="first")

    county_cbsa_map = zip_county_map[["ZIP", "COUNTY"]].merge(
        zip_cbsa_map[["ZIP", "CBSA"]], on="ZIP", how="inner"
    )
    county_cbsa_map = county_cbsa_map.drop_duplicates()

    print("unique ZIPs in zip-cbsa map:", zip_cbsa_map["ZIP"].nunique())
    print("unique ZIPs in zip-county map:", zip_county_map["ZIP"].nunique())
    print(
        "num zips in cbsa but not county:",
        len(set(zip_cbsa_map["ZIP"].unique()) - set(zip_county_map["ZIP"].unique())),
    )
    print(
        "num zips in county but not cbsa:",
        len(set(zip_county_map["ZIP"].unique()) - set(zip_cbsa_map["ZIP"].unique())),
    )

    counts = county_cbsa_map["COUNTY"].value_counts()
    duplicate_counties = counts[counts > 1].index.tolist()

    fix_map = {}
    for county in duplicate_counties:
        cbsa_count = (
            zip_cbsa_map.loc[
                zip_cbsa_map["ZIP"].isin(
                    zip_county_map.loc[zip_county_map["COUNTY"] == county, "ZIP"]
                )
            ]
            .groupby("CBSA")["ZIP"]
            .count()
        )
        correct_cbsa_map = cbsa_count.idxmax()
        fix_map[county] = correct_cbsa_map

    county_cbsa_map["CBSA"] = (
        county_cbsa_map["COUNTY"]
        .map(fix_map)
        .fillna(county_cbsa_map["CBSA"])
        .astype(int)
    )
    county_cbsa_map = county_cbsa_map.drop_duplicates()

    county_cbsa_map.index = county_cbsa_map.index + 1

    county_cbsa_map = county_cbsa_map.sort_index()

    assert county_cbsa_map["COUNTY"].nunique() == county_cbsa_map.shape[0]
    return county_cbsa_map
