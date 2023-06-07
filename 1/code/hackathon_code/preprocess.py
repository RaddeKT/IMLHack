import pandas as pd

# Read the data from the csv file
bookings_df = pd.read_csv('datasets/agoda_cancellation_train.csv')


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


# Create the pipe by using calling all the functions.
preprocessed_candidates = (bookings_df.
                            pipe(get_first_last_name, "Full_Name").
                            pipe(get_application_date_info, "Application_date").
                            pipe(candidate_info)
                          )

# Show the final result
preprocessed_candidates
