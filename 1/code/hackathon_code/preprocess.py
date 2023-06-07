import pandas as pd

BOOLEAN_COLUMNS = ['guest_is_not_the_customer',
                   'guest_is_not_the_customer',
                   'is_user_logged_in',
                   'is_first_booking']

DATE_COLUMNS = ['checkin_date', 'checkout_date']
DATETIME_COLUMNS = ['booking_datetime', 'hotel_live_date']

# hotel_country_code:
# 4 empty,
CATEGORY_COLUMNS = ['hotel_country_code', 'accommadation_type_name', 'customer_nationality',
                    'guest_nationality_country_name', 'origin_country_code', 'language', 'original_payment_method',
                    'original_payment_type', 'original_payment_currency', 'cancellation_policy_code', 'hotel_area_code',
                    'hotel_brand_code', 'hotel_chain_code', 'hotel_city_code', 'request_nonesmoke',
                    'request_latecheckin', 'request_highfloor', 'request_largebed', 'request_twinbeds',
                    'request_airport', 'request_earlycheckin']

# TBD what to do with it...
CATEGORY_COLUMNS_WITH_TOO_MANY_UNIQUE_VALUES = ['hotel_id', 'cancellation_policy_code', 'hotel_area_code',
                                                'hotel_city_code']
DROP_BECAUSE_MANY_UNFILLED_ROWS = ['hotel_brand_code', 'hotel_chain_code']

DROPPED_COLUMNS = CATEGORY_COLUMNS_WITH_TOO_MANY_UNIQUE_VALUES + DROP_BECAUSE_MANY_UNFILLED_ROWS

CATEGORY_COLUMNS_TO_EXPAND = [col for col in CATEGORY_COLUMNS if
                              col not in DROPPED_COLUMNS]


def drop_columns(df, columns):
    final_df = df.copy()
    final_df.drop(columns, axis=1, inplace=True)
    return final_df


def categorize_columns(df, column_names):
    final_df = df.copy()
    for column_name in column_names:
        # convert to category using get_dummies
        final_df = pd.get_dummies(final_df, columns=[column_name], prefix='cat', prefix_sep='_')

    return final_df


def replace_empty_with_UNKNOWN(df, column_names):
    final_df = df.copy()
    for column_name in column_names:
        # Replace the empty values with UNKNOWN
        final_df[column_name] = final_df[column_name].fillna("UNKNOWN")

    return final_df


def split_date_to_days_since_2000_and_minutes_since_start_of_day(df, column_names):
    final_df = df.copy()

    for column_name in column_names:
        final_df[column_name] = pd.to_datetime(final_df[column_name])
        # Convert the column to  number of days since beginning of the year
        final_df[f"{column_name}_days_since_2000"] = (final_df[column_name] - pd.to_datetime("2000-01-01")).dt.days

        # Convert the column to minutes since start of day
        final_df[f"{column_name}_minutes_since_start_of_day"] = final_df[column_name].dt.hour * 60 + final_df[
            column_name].dt.minute

        # Drop the original column
        final_df.drop(column_name, axis=1, inplace=True)

    return final_df


def split_date_to_days_since_2000(df, column_names):
    final_df = df.copy()
    for column_name in column_names:
        final_df[column_name] = pd.to_datetime(final_df[column_name])
        # Convert the column to  number of days since beginning of the year
        final_df[f"{column_name}_days_since_2000"] = (final_df[column_name] - pd.to_datetime("2000-01-01")).dt.days

        # Drop the original column
        final_df.drop(column_name, axis=1, inplace=True)

    return final_df


def convert_cancellation_date_to_did_cancel(df):
    final_df = df.copy()

    # Convert the column to 0 and 1
    final_df["did_cancel"] = final_df["cancellation_datetime"].notnull().astype(int)
    final_df.drop("cancellation_datetime", axis=1, inplace=True)
    return final_df


def fix_charge_option(df, column_name):
    final_df = df.copy()

    # Replace the values
    final_df[column_name] = final_df[column_name].replace({"Pay Now": 0, "Pay Later": 1, "Pay at Check-in": 1})

    return final_df


def convert_boolean_columns(df, column_names):
    final_df = df.copy()
    for column_name in column_names:
        # Convert the column to 0 and 1
        final_df[column_name] = final_df[column_name].astype(int)

    return final_df


def convert_date_column_to_days_and_time_of_day(df, column_name):
    final_df = df.copy()

    # Convert the column to  number of days since beginning of the year
    final_df[column_name] = pd.to_datetime(final_df[column_name])


def get_first_last_name(df, col_name):
    final_df = df.copy()

    splited_full_name = final_df[col_name].str.split(",", expand=True)

    final_df["First_Name"] = splited_full_name.get(0)
    final_df["Last_Name"] = splited_full_name.get(1)

    return final_df


def get_application_date_info(df, column_name):
    application_date = df[column_name]

    final_df = df.copy()

    final_df["Day"] = application_date.dt.day
    final_df["Month"] = application_date.dt.month
    final_df["Year"] = application_date.dt.year
    final_df["Day_of_week"] = application_date.dt.day_name()
    final_df["Month_of_year"] = application_date.dt.month_name()

    return final_df


def info_by_row(row):
    # Select columns of interest
    full_name = row.Full_Name.replace(",", " ")
    is_from = row.From
    degree = row.Degree
    from_office = row["From_office (min)"]

    # Generate the description from previous variables
    info = f"""{full_name} from {is_from} holds a {degree} degree 
              and lives {from_office} from the office"""

    return info


# Create the info
def candidate_info(df):
    final_df = df.copy()

    final_df["Info"] = final_df.apply(lambda row: info_by_row(row), axis=1)

    return final_df


if "__main__" == __name__:
    # Create the pipe by using calling all the functions.

    # Read the data from the csv file
    df = pd.read_csv('../datasets/agoda_cancellation_train.csv')

    df = (df

          # Handle boolean columns
          .pipe(fix_charge_option, "charge_option")
          .pipe(convert_boolean_columns, BOOLEAN_COLUMNS)
          )

    # Show boolean columns for inspection...
    # for col in BOOLEAN_COLUMNS:
    #     print(col)
    #     print(df[col].value_counts())

    # Handle date columns
    df = (df
          .pipe(convert_cancellation_date_to_did_cancel)
          .pipe(split_date_to_days_since_2000_and_minutes_since_start_of_day, DATETIME_COLUMNS)
          .pipe(split_date_to_days_since_2000, DATE_COLUMNS)
          )

    df = (df
          .pipe(drop_columns, CATEGORY_COLUMNS_WITH_TOO_MANY_UNIQUE_VALUES + DROP_BECAUSE_MANY_UNFILLED_ROWS))

    # Handle categorical columns
    df = (df
          .pipe(replace_empty_with_UNKNOWN, CATEGORY_COLUMNS_TO_EXPAND)
          .pipe(categorize_columns, CATEGORY_COLUMNS_TO_EXPAND)
          )

    non_numeric_columns = df.select_dtypes(exclude='number').columns

    # Check if there are non-numerical values in the DataFrame
    if len(non_numeric_columns) > 0:
        print("Non-numeric values exist in the DataFrame.")
        print("Non-numeric columns:", non_numeric_columns)
    else:
        print("No non-numeric values in the DataFrame.")

    # Check if there are any empty values in the DataFrame
    if df.isna().any().any():
        print("Empty values exist in the DataFrame.")
    else:
        print("No empty values in the DataFrame.")
