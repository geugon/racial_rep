import pandas as pd
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam


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
    y = y[:,[4,1,2,0,3]]
    # order matches datatable: White, Black, Latino, Asian, Native

    return x, y


def build_model(nfeats, ncats):
    input_ = Input(shape=(x.shape[1]), name='input')
    output_ = Dense(y.shape[1], activation='softmax', name='output')(input_)
    model = Model(input_, output_)
    opt = Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    return model


def extract_coef(model):
    values = np.concatenate([model.layers[1].weights[0].numpy(), model.layers[1].weights[1].numpy().reshape(1,-1)],axis=0)
    coeff_summary = pd.DataFrame(values, columns="White_rep Black_rep Hispanic_rep Asian_rep Native_rep".split())
    coeff_summary.index=("White_pop Black_pop Latino_pop Asian_pop Native_pop Baseline".split())
    return coeff_summary


def train():
    df = load_data(fname)
    x, y = parse_data(df)

    model = build_model(x.shape[0], y.shape[0])
    model.fit(x,y, batch_size=20, epochs=150)
    model.save_weights("weights.h5")


if __name__ == '__main__':
    train()