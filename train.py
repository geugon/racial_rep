import pandas as pd

def load_data(fname):
    df = pd.read_csv(fname, header=[0,1])

    # Clean up column names
    new_cols = []
    current_top=None
    for top, bot in df.columns:
        if not top.startswith('Unnamed'):
            current_top = top
        if bot.startswith('Unnamed'):
            new_cols.append(top)
        else:
            new_cols.append(': '.join([current_top, bot]))
    df.columns = new_cols
    return df

def parse_data(df):
    useful_cols = ['Race/ Ethnicity',
                   '2014-2018 ACS Citizen Adult Population: White',
                   '2014-2018 ACS Citizen Adult Population: Black',
                   '2014-2018 ACS Citizen Adult Population: Latino',
                   '2014-2018 ACS Citizen Adult Population: Asian and Pacific Islander',
                   '2014-2018 ACS Citizen Adult Population: Native',
                   '2014-2018 ACS Citizen Adult Population: Other',]

    # Remove data for 4 empty seats and the row containing averge values
    subset = df[useful_cols].dropna()

    # Simplify House Rep race to match demographic catagories
    race = subset['Race/ Ethnicity'].str.split().str[0].replace('Pacific', 'Asian')
    """
    The only "Pacific" entry is Tulsi Gabbard, whose own website describes her as being
    "of Asian, Polynesian, and Caucasian descent."  Mapping to "Asian"
    """

    # Get data for fitting
    x = subset.drop(['Race/ Ethnicity', "2014-2018 ACS Citizen Adult Population: Other"], axis=1).values/100
    y = pd.get_dummies(race).values
    # y in Alphabetic order: Asian, Black, Hispanic, Native, White
    # x order matches datatable: White, Black, Latino, Asian, Native

    return x, y