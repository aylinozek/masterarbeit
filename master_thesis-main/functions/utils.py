import sys
import os

from matplotlib.pylab import record
sys.path.append(os.path.abspath(".."))

from functions.config import *

# gender stratified plot colors
color_palette={'female':'#9ef0f0', 'male':'#8a3ffc'}
#single color
color="tab:blue" 

# ==============================================================
# ==============================================================
# METADATA RELATED FUNCTIONS
# ==============================================================
# ==============================================================

def form_age_groups(df, age_col='age', new_col='age_group', bins=None, labels=None, right=False):
    """
    Creates categorical age groups from age column.
    """
    df[new_col] = pd.cut(df[age_col], bins=bins, labels=labels, right=right)
    return df

def analyze_difference(df, column, grouping_col, filter_dict=None):
    """
    Performs a t-test on two groups in 'grouping_col'.
    Also, 'filter_dict' allows for within-group analysis (e.g., {'age_group': 'Young'}).
    """
    temp_df = df.copy()
    if filter_dict:
        for key, value in filter_dict.items():
            temp_df = temp_df[temp_df[key] == value]
            
    groups = sorted(temp_df[grouping_col].unique())
    data1 = temp_df[temp_df[grouping_col] == groups[0]][column].dropna()
    data2 = temp_df[temp_df[grouping_col] == groups[1]][column].dropna()
    
    stat, p = ttest_ind(data1, data2)
    
    return stat, p

def add_full_significance(df, column, ax, age_group_col='age_group', sex_col='sex'):
    '''
    checks for statistically significant difference (with independent t_test between groups 
    for the metadata box plots, and if there is adds a * (star) between ages and bracket between genders
    '''
    age_groups = sorted(df[age_group_col].unique())
    genders = df[sex_col].unique()
    
    y_min, y_max = df[column].min(), df[column].max()
    y_range = y_max - y_min
    y_offset = y_range * 0.05 

    # Gender difference check (within age_groups)
    for i, age in enumerate(age_groups):
        sub_df = df[df[age_group_col] == age]
        g1 = sub_df[sub_df[sex_col] == genders[0]][column].dropna()
        g2 = sub_df[sub_df[sex_col] == genders[1]][column].dropna()
        
        if len(g1) > 1 and len(g2) > 1:
            _, p_gender = ttest_ind(g1, g2)
            if p_gender < 0.05:
                # Places star above the taller of the two boxes in this group
                local_max = sub_df[column].max()
                ax.text(i, local_max + y_offset, '*', 
                        ha='center', fontsize=14, color='blue', fontweight='bold')

    # Age Group significant difference check
    age1_data = df[df[age_group_col] == age_groups[0]][column].dropna()
    age2_data = df[df[age_group_col] == age_groups[1]][column].dropna()
    
    _, p_age = ttest_ind(age1_data, age2_data)
    
    if p_age < 0.05:
        # Drawa bracket at the very top if there is difference
        y_bracket = y_max + (y_offset * 3)
        tick_h = y_offset * 0.5
        
        ax.plot([0, 0, 1, 1], [y_bracket - tick_h, y_bracket, y_bracket, y_bracket - tick_h], 
                lw=1.5, color='black')
        
        # Adds a * if there is difference for age
        star_age = "*" if p_age > 0.01 else "**"
        ax.text(0.5, y_bracket + (y_offset * 0.2), f"Age: {star_age}", 
                ha='center', va='bottom', fontsize=10)
                
def plot_boxplot_meta(df, column, ax=None, save_dir=None, grouping_col='age_group'):
    """
    Plots a boxplot (or many on grid) for any numeric column with age_group on x-axis and sex as hue.
    Checks statistically significan difference
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x='age_group',
        y=column,
        hue='sex',
        palette=color_palette,
        whis=(0, 100),
        width=0.5,
        linewidth=0.8,
        linecolor='black',
        ax=ax
    )
    add_full_significance(df, column, ax)
    ax.set_title(f'{column} by Age Group and Sex', fontsize=15)
    ax.set_xlabel('Age Group', fontsize=13)
    ax.set_ylabel(column, fontsize=13)
    ax.legend(title='Gender', frameon=True, fontsize=9, title_fontsize=9)

    if ax is None and save_dir is not None:  # only save if one plot
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"Boxplot_{column}.png"),
            dpi=400,
            bbox_inches="tight"
        )
        plt.show()

def clean_dataframe(df, keep_case_cols=None):
    '''
    cleans dataframe, numeric columns ensured and str columns cleaned and lower-cased (except for 
    keep_cols e.g. ID)
    '''
    if keep_case_cols is None:
        keep_case_cols = []

    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        elif pd.api.types.is_object_dtype(df[col]):
            s = (
                df[col]
                .astype(str)
                .str.replace(r'\s+', ' ', regex=True)
                .str.replace(r'[\u00A0\u200B]', '', regex=True)
                .str.strip()
            )

            if col not in keep_case_cols:
                s = s.str.lower()

            df[col] = s

    return df

def clean_string_column(df, columns_list):
    """
    Strips whitespace and lowercases a column entries. (for baseline_characteristics for example or annotated meal images)
    """
    for col in columns_list:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
        else:
            raise ValueError(f"Column '{col}' not found in dataframe.")

    return df

#Summary Table for the dataset overview table in the thesis tex
# ==============================================================

def summarize_column(series):
    return {
        "mean": series.mean(),
        "sd": series.std(),
        "min": series.min(),
        "max": series.max()
    }

def format_mean_sd(mean, sd):
    return f"{mean:.1f} $\\pm$ {sd:.1f}"

def create_summary_table(df, group_col=None, rename_dict=None):

    numeric_cols = df.select_dtypes(include=np.number).columns
    if rename_dict:
        numeric_cols = [col for col in numeric_cols if col in rename_dict]

    groups = ["Overall"]
    if group_col:
        groups += sorted(df[group_col].dropna().unique())

    results = {}

    for col in numeric_cols:
        var_name = rename_dict.get(col, col) if rename_dict else col
        results[var_name] = {}

        for group in groups:
            if group == "Overall":
                data = df[col].dropna()
            else:
                data = df[df[group_col] == group][col].dropna()

            stats = summarize_column(data)

            results[var_name][group] = {
                "mean_sd": format_mean_sd(stats["mean"], stats["sd"]),
                "min": f"{stats['min']:.1f}",
                "max": f"{stats['max']:.1f}"
            }

    return results, groups

def to_latex(results, groups):
    header = "Variable & " + " & ".join(groups) + " \\\\\n\\hline\n"

    lines = []
    for var, group_data in results.items():
        # Mean ± SD row
        mean_sd_row = [group_data[g]["mean_sd"] for g in groups]
        lines.append(f"{var} (Mean $\\pm$ SD) & " + " & ".join(mean_sd_row) + " \\\\")
        
        # Min row
        min_row = [group_data[g]["min"] for g in groups]
        lines.append(f"{var} (Min) & " + " & ".join(min_row) + " \\\\")
        
        # Max row
        max_row = [group_data[g]["max"] for g in groups]
        lines.append(f"{var} (Max) & " + " & ".join(max_row) + " \\\\")
        
        lines.append("\\hline")

    table = "\\begin{tabular}{l" + "c" * len(groups) + "}\n\\hline\n"
    table += header
    table += "\n".join(lines)
    table += "\n\\end{tabular}"

    return table

# ==============================================================
# ==============================================================
# CGM DATA LOADING AND PREPARATION
# ==============================================================
# ==============================================================

def load_and_filter_cgm(cgm_path, nrows = 1075, days_to_keep = None, drop_columns = None):
    """
    Loads CGM Excel file with MultiIndex columns and filter by selected days. 
    -------------------------------------------------------------------------
    nrows : number of rows to read (e.g. in this data we have some random values way outside the experiment time range in later rows
    so this number manually checked and fixed)
    days_to_keep : days to keep (e.g., ["Tag 1", ..., "Tag 10"]) (see config)
    """

    df = pd.read_excel(cgm_path, header=[0, 1], nrows=nrows)

    #MultiIndex day column
    day_col = ("Day", "Unnamed: 0_level_1")

    # Clean day column and as all the day names appear once fill all the rows' dayy column to make it easier to filter on later
    df[day_col] = (
        df[day_col]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace("nan", pd.NA)
        .ffill()
    )

    if days_to_keep is not None:
        df = df[df[day_col].isin(days_to_keep)]

    if drop_columns is not None:
        df = df.drop(columns=drop_columns, level=0)

    return df

def load_and_filter_complete_cgm(
    cgm_path,
    metadata_path,
    string_cols=['sex','completed_10days', 'completed_12days', 'complete_cgm_data'],
    days_to_keep=None,
    drop_columns=["Day", "Reference_time"],
    exclude_participants = ['AL28','AL43'] #these 2 had 'yes' for 'complete_cgm' but did not have it complete
):
    """
    Loads CGM data and participant metadata, filters for participants with complete CGM data (related column='yes'),
    also removes excluded participants, and returns filtered CGM and participant IDs.
    
    Returns:
        - cgm_complete: filtered CGM dataframe (only selected (valid) participants)
        - participant_ids: list of participant IDs used
        - metadata 
    """
    # Load CGM
    cgm = load_and_filter_cgm(cgm_path, days_to_keep=days_to_keep, drop_columns=drop_columns)
    
    # Load and clean metadata
    metadata = pd.read_excel(metadata_path)
    metadata = clean_string_column(metadata, string_cols)
    
    # Selects participants with complete CGM
    completed_cgm_metadata = metadata[metadata['complete_cgm_data'] == 'yes']
    participant_ids = completed_cgm_metadata['ID'].tolist()
    
    # Removes excluded participants
    if exclude_participants is not None:
        participant_ids = [pid for pid in participant_ids if pid not in exclude_participants]
    
    # Filter CGM
    cgm_complete = cgm.loc[:, cgm.columns.get_level_values(0).isin(participant_ids)]
    
    return cgm_complete, participant_ids, metadata

def clean_glucose_values(df, value_column='Glucose_Value'):
    """
    Cleans numeric columns stored with comma decimal separators in a MultiIndex column dataframe.
    (for CGM data)
    """
    
    df = df.copy()
    
    for participant in df.columns.get_level_values(0).unique():
        if (participant, value_column) in df.columns:
            df[(participant, value_column)] = (
                df[(participant, value_column)]
                .astype(str)
                .str.replace(',', '.', regex=False)
                .astype(float)
            )
    
    return df

def extract_time(x):
    '''
    extracts time only from datetime columm
    '''
    if isinstance(x, dt.datetime):
        return x.time()
    if isinstance(x, dt.time):
        return x
    return pd.NaT

def reshape_cgm_multicolumn(df):
    """
    Reshapes a MultiIndex CGM dataframe (participant, variable)
    into long format with participant as a column.
    """
    
    reshaped_data = []
    
    for participant in df.columns.get_level_values(0).unique():
        temp_df = df[participant].copy()
        temp_df.columns = ['Date', 'Time_of_the_day', 'Glucose_Value']
        temp_df['Participant_id'] = participant
        
        reshaped_data.append(temp_df[['Participant_id', 'Date', 'Time_of_the_day', 'Glucose_Value']])
    
    result = pd.concat(reshaped_data, ignore_index=True)
    
    # sort by participant
    result = result.sort_values('Participant_id').reset_index(drop=True)
    
    return result

def prepare_cgm_timeseries(df):
    """
    Prepare long-format CGM data for time-series analysis
    without removing glucose NaNs.(if there is no match for the meal event time (and their 2 hour post window
    those will excluded anyway later)
    """

    cgm = df.copy()
    cgm = clean_glucose_values(cgm)
    cgm = reshape_cgm_multicolumn(cgm)

    # Only drop rows where no timestamp can be built
    cgm = cgm.dropna(subset=["Date", "Time_of_the_day"])

    cgm["Date"] = pd.to_datetime(cgm["Date"],dayfirst=True,errors="coerce")
    cgm["time_only"] = cgm["Time_of_the_day"].apply(extract_time)

    # Build full datetime
    cgm["datetime"] = cgm["Date"].dt.normalize() + pd.to_timedelta(cgm["time_only"].astype(str))

    cgm = (
        cgm[["Participant_id", "datetime", "Glucose_Value"]]
        .sort_values(["Participant_id", "datetime"])
        .reset_index(drop=True)
    )
    return cgm

def analysis_ready_cgm(cgm_df):
    '''
    Creates a new df for analysis with columns participant_id, datetime, date and 
    time_hour in decimal hour-of-day (float between 0-24)
    '''
    cgm_timeseries=prepare_cgm_timeseries(cgm_df) 
    analysis_ready_cgm_df=cgm_timeseries.dropna(subset=['Glucose_Value'])
    analysis_ready_cgm_df["datetime"] = pd.to_datetime(analysis_ready_cgm_df["datetime"])
    analysis_ready_cgm_df["date"] = analysis_ready_cgm_df["datetime"].dt.date
    analysis_ready_cgm_df["time_hour"] = (
    analysis_ready_cgm_df["datetime"].dt.hour
    + analysis_ready_cgm_df["datetime"].dt.minute / 60
    + analysis_ready_cgm_df["datetime"].dt.second / 3600
    )
    return analysis_ready_cgm_df
# ==============================================================
# ==============================================================
# LLM (Gemini) output organising functions 
# ==============================================================
# ==============================================================
def clean_numeric_columns(df,numeric_cols_list):
    """
    Converts decimal format (comma to . float.
    """
    for col in numeric_cols_list:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

    return df

def add_time_features(df,datetime_column):
    """
    Extracts hour and date features from the datetime column of the meal image annotated dataset and separates
    """
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df["hour"] = df[datetime_column].dt.hour
    df["date"] = df[datetime_column].dt.date

    return df

def decode_yesno(val):
    '''
    Decodes yes/no answers in the meal annotations (e.g. added_sugar feature)
    '''
    if val is None:
        return np.nan

    val_str = str(val).strip().lower()

    match = re.search(r"\[(yes|no)\]", val_str)
    if match:
        val_str = match.group(1)
    if val_str == "yes":
        return 1
    if val_str == "no":
        return 0

    return np.nan

def decode_fluid(val):
    '''
    Decodes fluid_solid column answers in the meal annotations, 1: only fluid, 0:solid or both
    '''
    if val is None:
        return np.nan

    val_str = str(val).strip().lower()

    match = re.search(r"\[(fluid|solid|both)\]", val_str)
    if match:
        val_str = match.group(1)
        
    if val_str == "both":
        return 0
    if val_str == "fluid":
        return 1
    if val_str == "solid":
        return 0

    return np.nan

def decode_diet(val):
    '''
    Decodes diet column answers in the meal annotations, 0: vegan, 1:vegetarian 2: omnivore
    '''
    if val is None:
        return np.nan

    val_str = str(val).strip().lower()

    match = re.search(r"\[(vegan|vegetarian|omnivore)\]", val_str)
    if match:
        val_str = match.group(1)

    if val_str == "vegan":
        return 0
    if val_str == "vegetarian":
        return 1
    if val_str == "omnivore":
        return 2
    return np.nan

# ==============================================================
# ==============================================================
# CGM FEATURE EXTRACTION FUNCTIONS 
# ==============================================================
# ==============================================================

def compute_time_in_range(df, participant_col = "Participant_id", glucose_col = "Glucose_Value"):
    """
    Computes percentage of time spent in glucose ranges per participant.
    ['<3.9', '3.9-10', '>10']
    """
    df = clean_glucose_values(df)
    cgm = reshape_cgm_multicolumn(df)
    cgm = cgm[[participant_col, glucose_col]].dropna(subset=[glucose_col])

    grouped = cgm.groupby(participant_col)[glucose_col]

    result = pd.DataFrame({
        '<3.9': grouped.apply(lambda x: (x < 3.9).mean() * 100),
        '3.9-10': grouped.apply(lambda x: ((x >= 3.9) & (x < 10)).mean() * 100),
        '>10': grouped.apply(lambda x: (x > 10).mean() * 100),
    }).round(3)

    result = result.reset_index().rename(columns={"index": "Participant_id"})

    return result

# ==============================================================

def cosine_period(t, mesor, amplitude, acrophase, period):
    '''
    Cosinor formula
    '''
    return mesor + amplitude * np.cos(2*np.pi*t/period - acrophase)

def cosinor_curve(time_hours, mesor, amplitude, acrophase_rad, period):
    '''
    for plotting same thing
    '''
    omega = 2 * np.pi / period
    return mesor + amplitude * np.cos(omega * time_hours + acrophase_rad)
    #plotting needs +acrophase to align the peak regarding the onset, 
    #nothing mathematically changes in the curve

def fit_cosinor(cgm_df):
    """
    Fit cosinor model for each participant on the CGM dataframe (input: wide format not anaylsis_ready_cgm_df function's long format output).
    """
    cgm_df = clean_glucose_values(cgm_df)
    cosinor_parameters = []

    for participant_id in cgm_df.columns.get_level_values(0).unique():
        cosinor_participant = cgm_df[participant_id].copy()
        cosinor_participant = cosinor_participant.dropna(subset=['Glucose_Value'])

        # Combine Date and Time into DateTime
        cosinor_participant['DateTime'] = pd.to_datetime(
            cosinor_participant['Date'].astype(str) + ' ' + cosinor_participant['Time_of_the_day'].astype(str),
            errors='coerce',
            utc=True
        )

        # Keep only DateTime and Glucose_Value
        cosinor_participant = cosinor_participant[['DateTime','Glucose_Value']]
        cosinor_participant = cosinor_participant.dropna()

        # Convert time to hours since start
        t = (cosinor_participant['DateTime'] - cosinor_participant['DateTime'].min()).dt.total_seconds() / 3600
        y = cosinor_participant['Glucose_Value']
        # Skip if no data
        if len(y) == 0:
            print(f"Skipping {participant_id}: no glucose data after dropping NaNs")
            continue

        # Fit cosinor
        params, _ = curve_fit(
            cosine_period,
            t,
            y,
            p0=[6,2,0,24],
            bounds=([0,0,-np.pi,15], [np.inf,np.inf,np.pi,30])
        )

        mesor, amplitude, acrophase, period = params

        cosinor_parameters.append({
            'Participant_id': participant_id,
            'mesor': mesor,
            'amplitude': amplitude,
            'acrophase': acrophase,
            'period': period
        })

    return pd.DataFrame(cosinor_parameters)
    
def compute_circadian_baseline(df, mesor, amplitude, acrophase_rad, period):
    '''
    fitted curve for each participant for plotting as the 'estimated circadian baseline'
    '''
    omega = 2 * np.pi / period
    return mesor + amplitude * np.cos(
        omega * df["time_hour"] + acrophase_rad
    )
# ==============================================================

def compute_speed_acceleration(df, participant_col='Participant_id', glucose_col='Glucose_Value', 
                               date_col='Date', time_col='Time_of_the_day', max_gap_minutes=60):
    """
    Compute mean and SD of speed (first derivative) and acceleration (second derivative)
    of CGM glucose values.

    """
    df = clean_glucose_values(df)
    df = reshape_cgm_multicolumn(df)
    results = []

    for participant in df[participant_col].unique():
        df_part = df[df[participant_col] == participant].copy()
        df_part = df_part.dropna(subset=[glucose_col, date_col, time_col])

        datetime = pd.to_datetime(
            df_part[date_col].astype(str) + ' ' + df_part[time_col].astype(str),
            errors='coerce'
        )

        mask = ~datetime.isna()
        glucose = df_part[glucose_col].to_numpy()[mask]
        datetime = datetime[mask]

        # Convert datetime to minutes
        time_minutes = datetime.apply(lambda x: x.timestamp() / 60)

        # Compute speed and acc (first and second derivatives)
        dG = np.diff(glucose)
        dt = np.diff(time_minutes)

        dt[dt <= 0] = np.nan
        dt[dt > max_gap_minutes] = np.nan

        speed = dG / dt
        acc = np.diff(speed) / dt[1:]

        mean_speed = np.nanmean(np.abs(speed))
        sd_speed = np.nanstd(np.abs(speed))
        mean_acc = np.nanmean(np.abs(acc))
        sd_acc = np.nanstd(np.abs(acc))

        results.append({
            'Participant_id': participant,
            'Mean_Speed': mean_speed,
            'SD_Speed': sd_speed,
            'Mean_Acceleration': mean_acc,
            'SD_Acceleration': sd_acc
        })

    return pd.DataFrame(results)

# ==============================================================
# ==============================================================
# PLOTTING FUNCTIONS FOR CGM
# ==============================================================
# ==============================================================
    
def plot_cgm_over_days(df, participant, date_col='Date', glucose_col='Glucose_Value', participant_col='Participant_id'):
    """
    Scatter plot of CGM values over days for one participant.
    """
    df = clean_glucose_values(df)
    df = reshape_cgm_multicolumn(df)
    df_part = df[df[participant_col] == participant].copy()

    df_part = df_part.sort_values(date_col)

    plt.figure(figsize=(15, 6))

    unique_dates = sorted(df_part[date_col].unique())
    date_to_index = {date: idx for idx, date in enumerate(unique_dates)}
    x_values = df_part[date_col].map(date_to_index)

    plt.scatter(
        x_values,
        df_part[glucose_col],
        color=color,
        alpha=0.5,
        s=20
    )

    plt.xticks(
        ticks=range(len(unique_dates)),
        labels=[str(date) for date in unique_dates],
        rotation=45
    )

    plt.title(f'CGM Values Over Time - Participant {participant}')
    plt.xlabel('Date')
    plt.ylabel('Glucose Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_one_day_cgm(cgm_df, participant_id, date):
    """
    Plot one day of CGM glucose values for a participant.
    date : Date to plot (format 'YYYY-MM-DD').
    """
    cgm_df = prepare_cgm_timeseries(cgm_df)
    cgm_p = cgm_df[cgm_df["Participant_id"] == participant_id]

    # Filter for the specific day
    one_day = cgm_p[
        (cgm_p["datetime"] >= pd.to_datetime(date)) &
        (cgm_p["datetime"] <  pd.to_datetime(date) + pd.Timedelta(days=1))
    ]

    if one_day.empty:
        print(f"No data for participant {participant_id} on {date}")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(one_day["datetime"], one_day["Glucose_Value"], marker="o", linestyle='-')
    plt.title(f"{participant_id} – CGM on {date}")
    plt.xlabel("Time")
    plt.ylabel("Glucose (mg/dL)")
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_glucose_profile(cgm_df, participant_id, save_dir=None, glucose_col='Glucose_Value'):
    """
    Plot glucose profile of one participant over all days (full CGM).
    """
    display_id = participant_id.replace('AL', 'P')

    cgm_df = prepare_cgm_timeseries(cgm_df)
    cgm_p = cgm_df[cgm_df["Participant_id"] == participant_id]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cgm_p["datetime"], cgm_p[glucose_col], color=color)
    ax.set_title(f"CGM glucose over time – {display_id}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Glucose")

    ax.xaxis.set_major_locator(mdates.DayLocator())  # every day
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    for tick in ax.get_xticks():
        ax.axvline(x=tick, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"glucose_profile_{display_id}.png"),
            dpi=400,
            bbox_inches="tight"
        )
    plt.show()
    
def plot_cgm_with_circadian_baseline(cgm_df, participant_id, save_dir=None):
    '''
    plots the CGM values overlaid with the cosinor curve representing the estimated circadian rhythm
    '''
    display_id = participant_id.replace('AL', 'P')
    cgm = analysis_ready_cgm(cgm_df)
    cosinor_df = fit_cosinor(cgm_df)
    df_p = cgm[cgm["Participant_id"] == participant_id].copy()

    row = cosinor_df[cosinor_df["Participant_id"] == participant_id].iloc[0]

    # Circadian baseline calcluate
    df_p["circadian_baseline"] = compute_circadian_baseline(
        df_p,
        mesor=row["mesor"],
        amplitude=row["amplitude"],
        acrophase_rad=row["acrophase"],
        period=row['period']
    )

    plt.figure(figsize=(10, 4))

    # CGM: connected continuous line (ALL DAYS)
    plt.plot(
        df_p["datetime"],
        df_p["Glucose_Value"],
        color="tab:blue",
        alpha=0.5,
        linewidth=1,
        label="CGM data"
    )

    # Circadian baseline - plot
    plt.plot(
        df_p["datetime"],
        df_p["circadian_baseline"],
        color="black",
        linewidth=2.5,
        label="Circadian baseline"
    )

    plt.xlabel("Time")
    plt.ylabel("Glucose (mmol/L)")
    plt.title(f"Participant {display_id}")
    plt.legend(frameon=False)

    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"circadian_baseline_{display_id}.png"),
            dpi=400,
            bbox_inches="tight"
        )

    plt.show()
    
def plot_cgm_superposition(cgm_df, participant_id, ax=None, pid_col="Participant_id"):
    '''
    plots the 10 day (day by day) CGM curves overlaid by the mean curve 
    '''
    df = analysis_ready_cgm(cgm_df)
    display_id = participant_id.replace('AL', 'P')
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    df_p = df[df[pid_col] == participant_id].copy()

    # Individual days
    for _, df_day in df_p.groupby("date"):
        ax.plot(
            df_day["time_hour"],
            df_day["Glucose_Value"],
            color="tab:blue",
            alpha=0.25,
            linewidth=1
        )

    # Mean curve
    df_mean = (
        df_p
        .groupby(pd.cut(df_p["time_hour"], bins=np.linspace(0, 24, 97)))
        ["Glucose_Value"]
        .mean()
        .reset_index()
    )
    df_mean["time_center"] = df_mean.iloc[:, 0].apply(lambda x: x.mid)

    ax.plot(
        df_mean["time_center"],
        df_mean["Glucose_Value"],
        color="black",
        linewidth=2.5,
        label="Mean across days"
    )

    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Glucose (mmol/L)")
    ax.set_title(f"Participant {display_id}")

    if ax.figure:
        plt.tight_layout()

    return ax

# ==============================================================
# ==============================================================
# MEAL DATA RELATED FUNCTIONS
# ==============================================================
# ==============================================================

def extract_meal_events(image_paths):
    '''
    Extracts the datetime information from exif time unlesss it is not available or duplicate,
    in such cases the datetime information is extracted from the filename (which is likely the upload time)
    very specific funstion wrt to the filename format and study period
    '''
    records = []

    for path in image_paths:
        filename = os.path.basename(path)
        participant_id = os.path.basename(os.path.dirname(path))

        #Read EXIF
        exif_time = None
        try:
            with open(path, "rb") as img_file:
                img = Image(img_file)
                if hasattr(img, "datetime_original"):
                    exif_time = dt.datetime.strptime(
                        img.datetime_original,
                        "%Y:%m:%d %H:%M:%S"
                    )
        except:
            pass

        #Parse filename time
        file_time = None
        match = re.search(
            r'_?\s*(\d{1,2})-(\d{1,2})_(\d{1,2})-(\d{1,2})-(\d{1,2})_', filename
        )

        if match:
            month, day, hour, minute, second = map(int, match.groups())

            if month in [9,10,11,12]:
                year = 2024
            else:
                year = 2025

            file_time = dt.datetime(year, month, day, hour, minute, second)

        records.append({
            "participant_id": participant_id,
            "file_name": filename,
            "exif_time": exif_time,
            "file_time": file_time
        })

    df = pd.DataFrame(records)

    # First checks exif
    df["date_time_stamp"] = df["exif_time"]

    # Find if there are EXIF duplicates per participant
    duplicate_mask = df.duplicated(
        subset=["participant_id", "exif_time"],
        keep=False
    )

    # For duplicates : use their own filename time
    df.loc[duplicate_mask, "date_time_stamp"] = df.loc[duplicate_mask, "file_time"]

    # Also if EXIF missing : use filename time
    df.loc[df["date_time_stamp"].isna(), "date_time_stamp"] = df["file_time"]

    return df[["participant_id", "file_name", "date_time_stamp"]]

def clean_meal_events(df):
    '''
    from the meal events, cleans the datetime and separates to date and time of the day
    '''
    df = df.copy()

    df["date_time_stamp"] = pd.to_datetime(df["date_time_stamp"])

    df = (
        df
        .sort_values(["participant_id", "date_time_stamp"])
        .reset_index(drop=True)
    )

    df["time_of_the_day"] = df["date_time_stamp"].dt.time
    df["date"] = df["date_time_stamp"].dt.date

    return df

# ==============================================================
# below function was an earlier version, now only used for individual example post meal graphs (almost idle)

def extract_normalized_meal_segments(meal_events_df, cgm_df, post_start_min, post_end_min):
    '''
    matches meals and corresponding cgm windows, below there is an updated version
    '''
    segments = []

    for _, meal in meal_events_df.iterrows():

        pid = meal["participant_id"]
        meal_time = meal["date_time_stamp"]

        # Define CGM window
        start_time = meal_time + pd.Timedelta(minutes=post_start_min)
        end_time   = meal_time + pd.Timedelta(minutes=post_end_min)

        # Selects CGM data
        seg = cgm_df[
            (cgm_df["Participant_id"] == pid) &
            (cgm_df["datetime"] >= start_time) &
            (cgm_df["datetime"] <= end_time)
        ].copy()

        if seg.empty:
            continue

        # Sort and compute time since meal
        seg = seg.sort_values("datetime")
        seg["meal_time"] = meal_time
        seg["time_since_meal_min"] = (
            (seg["datetime"] - meal_time)
            .dt.total_seconds() / 60
        )

        # Normalize glucose values
        baseline = seg["Glucose_Value"].iloc[0]
        seg["glucose_norm"] = seg["Glucose_Value"] - baseline

        seg = seg.drop(columns=["Unnamed: 0"], errors="ignore")
        segments.append(seg)

    return segments
# ==============================================================

def extract_normalized_meal_segments_merged(meal_events_df, cgm_df, post_start_min, post_end_min):
    '''
    MAIN FUNCTION to form a analysis ready dataset of meals and corresponding glucose response windows
    '''
    all_segments = []

    for _, meal in meal_events_df.iterrows():

        pid = meal["participant_id"]
        meal_time = meal["datetime"]

        start_time = meal_time + pd.Timedelta(minutes=post_start_min)
        end_time = meal_time + pd.Timedelta(minutes=post_end_min)

        seg = cgm_df[
            (cgm_df["Participant_id"] == pid) &
            (cgm_df["datetime"] >= start_time) &
            (cgm_df["datetime"] <= end_time)
        ].copy()

        if seg.empty:
            continue

        seg = seg.sort_values("datetime")

        seg["meal_id"] = meal["meal_id"] if "meal_id" in meal else _
        seg["meal_time"] = meal_time
        seg["time_since_meal_min"] = (
            (seg["datetime"] - meal_time).dt.total_seconds() / 60
        )
        #normalizes glucose values (with baseline being the value at the meal event time)
        baseline = seg["Glucose_Value"].iloc[0]
        seg["glucose_norm"] = seg["Glucose_Value"] - baseline

        # attach meal features
        for col in ["total_calories", "carbs_g","protein_g","fat_g","fiber_g", "glycemic_index", "estimated_total_weight_g","diet_type", "ultra_processed", "food_processing_level", "added_sugar", "fluid_or_solid"]:
            if col in meal:
                seg[col] = meal[col]

        all_segments.append(seg)

    return pd.concat(all_segments, ignore_index=True)

# ==============================================================
# ==============================================================
# RECORDING RATE ANALYSIS FUNCTIONS
# ==============================================================
# ==============================================================

def create_sliding_windows(cgm_df, window_size=8, step_size=1):
    '''
    creates 2 hour (8 points in 15 min resolution) windows from the CGM data, glucose values normalised
    '''
    windows = []

    for pid, group in cgm_df.groupby("Participant_id"):

        group = group.sort_values("datetime").reset_index(drop=True)

        for i in range(0, len(group) - window_size + 1, step_size):

            w = group.iloc[i:i+window_size].copy()

            t0 = w.iloc[0]["datetime"]
            baseline = w.iloc[0]["Glucose_Value"]

            w["window_id"] = f"{pid}_{i}"
            w["time_since_start_min"] = (
                (w["datetime"] - t0).dt.total_seconds() / 60
            )
            w["glucose_norm"] = w["Glucose_Value"] - baseline

            windows.append(w)

    return pd.concat(windows, ignore_index=True)

def build_participant_template(meal_segments_df):
    '''
    Participant response template from the 2 hour windows after the recorded meals events
    as the mean curve
    '''
    templates = {}

    for pid, group in meal_segments_df.groupby("Participant_id"):

        curves = []

        for meal_id, seg in group.groupby("meal_id"):

            seg = seg.sort_values("time_since_meal_min")
            y = seg["glucose_norm"].values
            curves.append(y)
        #there might me CGM data missing for that meal
        if len(curves) == 0:
            continue

        min_len = min(len(c) for c in curves)
        curves = [c[:min_len] for c in curves]
        templates[pid] = np.mean(curves, axis=0)

    return templates

def compute_similarity(window_curve, template_curve):
    '''
    computes similarity between each 2-hour window and the template as the Mean Squared Error (MSE)
    '''
    return np.mean((window_curve - template_curve) ** 2)

def compute_meal_distances(meal_segments_df, templates):
    '''
    computes the distance between the template (mean) and true meals, later used for the similarity threshold decision
    '''
    distances = []

    for pid, group in meal_segments_df.groupby("Participant_id"):

        if pid not in templates:
            continue

        template = templates[pid]

        caloric = group[group["total_calories"] > 0]

        for meal_id, seg in caloric.groupby("meal_id"):

            seg = seg.sort_values("time_since_meal_min")

            y = seg["glucose_norm"].values

            if len(y) < len(template):
                continue

            y = y[:len(template)]

            dist = np.linalg.norm(y - template)

            distances.append(dist)

    return np.array(distances)

def detect_candidate_meals(windows_df, templates, threshold):
    '''
    detects candidate meals by the similarity distance, threshold is determined from the data
    (it is the 75th percentile of the templates deviation from each actual meal (so stability of the mean is important))
    '''
    candidates = []

    for pid, group in windows_df.groupby("Participant_id"):

        template = templates[pid]

        for window_id, w in group.groupby("window_id"):

            w = w.sort_values("time_since_start_min")
            if len(w) < 4:
                continue
            y = w["glucose_norm"].values
            y = y[:len(template)]

            # skip if still shorter than template
            if len(y) < len(template):
                continue

            dist = compute_similarity(y, template)
            if dist < threshold:
                candidates.append({
                    "Participant_id": pid,
                    "window_id": window_id,
                    "start_time": w["datetime"].iloc[0],
                    "distance": dist
                })

    return pd.DataFrame(candidates)

def remove_true_meals(candidates_df, meal_events_df):
    '''
    removes the recorded (true) meals from candidates
    '''
    cleaned = []

    for _, row in candidates_df.iterrows():

        pid = row["Participant_id"]
        start = row["start_time"]
        end = start + pd.Timedelta(minutes=120)

        meals_pid = meal_events_df[
            meal_events_df["Participant_id"] == pid
        ]

        overlap = any(
            start <= mt <= end
            for mt in meals_pid["datetime"]
        )

        if not overlap:
            cleaned.append(row)

    return pd.DataFrame(cleaned)

def collapse_overlapping_meals(candidate_df, overlap_minutes=90):
    '''
    since the sliding windows has 7 points common, if from the candidates windows there are 2 with 90 minutes overlap (6 points)
    they are collapsed into one and count as one candidate
    '''

    final_meals = []

    for pid, group in candidate_df.groupby("Participant_id"):

        group = group.sort_values("distance")

        selected = []

        for _, row in group.iterrows():

            start = row["start_time"]

            conflict = False
            for sel in selected:
                if abs((start - sel["start_time"]).total_seconds()) / 60 < overlap_minutes:
                    conflict = True
                    break

            if not conflict:
                selected.append(row)

        final_meals.extend(selected)

    return pd.DataFrame(final_meals)

def compute_recording_rate(candidates_df, meal_events_df):
    '''
    main recording rate calculation from the candidate isolated meals and true recorded meals
    '''
    results = []

    for pid in meal_events_df["Participant_id"].unique():
        # Count distinct meals
        recorded = meal_events_df[
            meal_events_df["Participant_id"] == pid
        ]["meal_id"].nunique()

        detected = len(
            candidates_df[candidates_df["Participant_id"] == pid]
        )

        total_estimated = recorded + detected
        rate = recorded / total_estimated if total_estimated > 0 else np.nan

        results.append({
            "Participant_id": pid,
            "recorded_meals": recorded,
            "detected_unrecorded": detected,
            "recording_rate": rate
        })

    return pd.DataFrame(results)
# ==============================================================
# ==============================================================
#Polar plots, fasting, feeding, meal overlaid cgm plots
# ==============================================================
# ==============================================================
def add_circular_time_features(meal_data, participant_col="participant_id", datetime_col="datetime"):
    """
    Adds circular time features and radii to meal_data for the polar plots
    """
    meal_data['time_of_the_day'] = meal_data[datetime_col].dt.time
    meal_data['hours'] = meal_data['time_of_the_day'].apply(
        lambda t: t.hour + t.minute/60 + t.second/3600
    )
    
    # Convert hours to radians (for circular plots)
    meal_data['radians'] = (meal_data['hours'] / 24) * 2 * np.pi
    
    participants = meal_data[participant_col].unique()
    radii = {pid: 0.5 + i*0.05 for i, pid in enumerate(participants)}
    meal_data['radius'] = meal_data[participant_col].map(radii)
    
    return meal_data
# ==============================================================
def compute_fasting_analysis(meal_data, participant_col="participant_id", datetime_col="datetime", min_photos_per_day=[2,3], max_fasting_hours=[16,24]):
    """
    Computes fasting windows, daily meal counts, and valid participants based on meal data.

    """
    meal_data['date'] = meal_data[datetime_col].dt.date
    meal_data['time_of_the_day'] = meal_data[datetime_col].dt.time
    
    # Compute first and last meal per day
    fasting_times = meal_data.groupby([participant_col, 'date'])['time_of_the_day'].agg(['min','max']).rename(
        columns={'min':'first_meal','max':'last_meal'}).reset_index()
    fasting_times['first_meal'] = pd.to_datetime(fasting_times['date'].astype(str) + ' ' + fasting_times['first_meal'].astype(str), errors='coerce', utc=True)
    fasting_times['last_meal'] = pd.to_datetime(fasting_times['date'].astype(str) + ' ' + fasting_times['last_meal'].astype(str), errors='coerce', utc=True)
    
    # Compute fasting period until next day's first meal
    fasting_times['next_first_meal'] = fasting_times.groupby(participant_col)['first_meal'].shift(-1)
    fasting_times['fasting_period'] = (fasting_times['next_first_meal'] - fasting_times['last_meal']).dt.total_seconds()/3600
    fasting_times['more_than_1day'] = fasting_times['fasting_period'] > 24
    fasting_participant_avg = fasting_times.groupby(participant_col)['fasting_period'].mean().reset_index().rename(
        columns={'fasting_period':'avg_fasting_window'}
    )
    
    # Daily meal event counts
    daily_meal_event_counts = meal_data.groupby([participant_col,'date']).size().reset_index(name='number_of_meal_events')
    avg_number_of_meal_events = daily_meal_event_counts.groupby(participant_col)['number_of_meal_events'].mean().reset_index(name='avg_meal_photos_per_day')
    

    valid_participants_photos = {}
    for n in min_photos_per_day:
        valid_participants_photos[n] = avg_number_of_meal_events[avg_number_of_meal_events['avg_meal_photos_per_day'] >= n][participant_col].to_list()

    valid_participants_fasting = {}
    for h in max_fasting_hours:
        valid_participants_fasting[h] = fasting_participant_avg[fasting_participant_avg['avg_fasting_window'] <= h][participant_col].to_list()
    
    return {
        'fasting_times': fasting_times,
        'fasting_participant_avg': fasting_participant_avg,
        'daily_meal_event_counts': daily_meal_event_counts,
        'avg_number_of_meal_events': avg_number_of_meal_events,
        'valid_participants_photos': valid_participants_photos,
        'valid_participants_fasting': valid_participants_fasting
    }
# ==============================================================
def plot_glucose_with_meals(cgm_df, meal_df,participant_id, date=None,glucose_col="Glucose_Value", 
                            n_days=1, save_dir=None):
    """
    Plot CGM glucose profile for a specific participant and date,
    overlaying meal events as vertical lines with labels.

    for visual exploration only

    date : Date to visualize (format: 'YYYY-MM-DD')
    """

    p_cgm = (cgm_df.loc[cgm_df["Participant_id"] == participant_id].copy())
    p_meals = (meal_df.loc[meal_df["participant_id"] == participant_id].copy())

    if p_cgm.empty:
        print(f"No CGM data found for participant {participant_id}.")
        return
    if n_days == 1:
        font=9
    else:
        font=7

    p_cgm["datetime"] = pd.to_datetime(p_cgm["datetime"])
    p_meals["datetime"] = pd.to_datetime(p_meals["datetime"])

    if n_days == 1:
        dates = [pd.to_datetime(date).date()]
    else:
        unique_days = p_cgm["datetime"].dt.date.unique()
        if len(unique_days) < n_days:
            print("Not enough days available.")
            return
        dates = np.random.choice(unique_days, size=n_days, replace=False)

    if n_days == 1:
        fig, axes = plt.subplots(figsize=(12, 6))
        axes = [axes]
    else:
        n_cols = 2
        n_rows = int(np.ceil(n_days / 2))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        axes = axes.flatten()
    offset_levels = [0.85, 0.20]

    for ax, date in zip(axes, dates):

        day_cgm = p_cgm[p_cgm["datetime"].dt.date == date].sort_values("datetime")
        day_meals = p_meals[p_meals["datetime"].dt.date == date]

        if day_cgm.empty:
            ax.set_title(f"No data ({date})")
            continue


        ax.plot(
            day_cgm["datetime"],
            day_cgm[glucose_col],
            linewidth=2,
            alpha=0.85,
            color="tab:blue"
        )

        for i, (_, row) in enumerate(day_meals.iterrows()):
            meal_time = row["datetime"]
            meal_label = row.get("meal_label", "Meal")

            ax.axvline(
                x=meal_time,
                linestyle="--",
                alpha=0.6,
                linewidth=1,
                color="tab:blue"
            )

            wrapped_label = "\n".join(textwrap.wrap(str(meal_label), width=18))

            ax.text(
                meal_time,
                offset_levels[i % len(offset_levels)],
                wrapped_label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=font,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            )

        ax.set_title(f"Participant {participant_id} — {date}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Glucose")
        ax.set_ylim(p_cgm[glucose_col].min(), p_cgm[glucose_col].max())
        ax.grid(True, linestyle=":", alpha=0.5)

    # hide empty plots
    for ax in axes[len(dates):]:
        ax.axis("off")

    plt.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"meal_overlapped_cgm_{date}_{participant_id}.png"),
            dpi=400,
            bbox_inches="tight"
        )
    plt.show()
# ==============================================================
# ==============================================================
#CLUSTERING RELATED FUNCTIONS
# ==============================================================
# ==============================================================

def extract_eating_habits(df):
    '''
    Extracts features (macronutrient averages and percentages, night eater ratio etc) for clustering tasks
    (input: the meal features dataset with the time_hour (0-24 float format))
    '''
    #night eater ration ( night meals: between 22-4 )
    df['is_late_night'] = ((df['time_hour'] >= 22) | (df['time_hour'] <= 4)).astype(int)

    daily_totals = df.groupby(['Participant_id', 'date']).agg({
        'carbs_g': 'sum',
        'fat_g': 'sum',
        'fiber_g': 'sum',
        'protein_g': 'sum',
        'total_calories': 'sum',  
    }).reset_index()

    # Computes % of daily calories from each macronutrient
    # Carbs & protein = 4 kcal/g, fat = 9 kcal/g
    daily_totals['carb_cal_pct'] = (daily_totals['carbs_g'] * 4) / (daily_totals['total_calories'] + 1e-6)
    daily_totals['fat_cal_pct'] = (daily_totals['fat_g']* 9) / (daily_totals['total_calories'] + 1e-6)
    daily_totals['protein_cal_pct'] = (daily_totals['protein_g'] * 4) / (daily_totals['total_calories'] + 1e-6)
    
    # Fiber density: g of fiber per 1000 kcal - to normalise for total intake
    daily_totals['fiber_density'] = (daily_totals['fiber_g'] / (daily_totals['total_calories'] + 1e-6) * 1000)

    user_daily_avg = daily_totals.groupby('Participant_id').agg(
        avg_daily_carbs=('carbs_g', 'mean'),
        avg_daily_fat=('fat_g', 'mean'),
        avg_daily_fiber=('fiber_g', 'mean'),
        avg_daily_protein=('protein_g', 'mean'),
        sd_daily_carbs=('carbs_g', 'std'),
        avg_daily_calories=('total_calories', 'mean'),
        avg_carb_cal_pct=('carb_cal_pct', 'mean'),       
        avg_fat_cal_pct=('fat_cal_pct', 'mean'),        
        avg_protein_cal_pct=('protein_cal_pct', 'mean'),
        avg_fiber_density=('fiber_density', 'mean')     
    )

    meal_counts = df.groupby(['Participant_id', 'date']).size().reset_index(name='n_meals')
    meal_freq = meal_counts.groupby('Participant_id')['n_meals'].mean().rename('avg_meals_per_day')

    meal_size_std = (df.groupby('Participant_id')['total_calories'].std().rename('std_meal_kcal'))

    user_habits = df.groupby('Participant_id').agg(
        pct_ultra_processed=('ultra_processed', 'mean'),
        pct_added_sugar=('added_sugar', 'mean'),
        pct_liquid_meals=('fluid_or_solid', 'mean'),
        night_eater_ratio=('is_late_night', 'mean'),
        diet_type=('diet_type', lambda x: x.mode()[0] if not x.mode().empty else "unknown"),
        glucose_baseline=('Glucose_Value', 'mean')
    )

    profile_df = pd.concat([user_daily_avg, user_habits, meal_freq, meal_size_std], axis=1).reset_index()

    return profile_df

def transform_features(profile_df, skew_threshold=0.5):
    """
    Computes skewness per feature and applies the appropriate transform:
    Categorical (diet_type) gets one-hot encoded.
    """
    df = profile_df.copy()
    
    # diet_type encode
    if 'diet_type' in df.columns:
        diet_dummies = pd.get_dummies(df['diet_type'], prefix='diet')
        df = df.drop(columns=['diet_type'])
        df = pd.concat([df, diet_dummies], axis=1)
    
    # transform numeric, (non-ID columns)
    skip_cols = {'Participant_id', 'cluster'}
    bool_cols = df.select_dtypes(include='bool').columns.tolist()  # one-hot boolean columns
    numeric_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in skip_cols and c not in bool_cols
    ]
    
    transform_log = []
    for col in numeric_cols:
        s = skew(df[col].dropna())
        abs_s = abs(s)
        if abs_s < skew_threshold:
            action = 'none'
        elif skew_threshold <= abs_s < 1.0:
            action = 'sqrt'
            if s < 0:
                df[col] = -np.sqrt(-df[col] + df[col].max() + 1)
            else:
                df[col] = np.sqrt(df[col].clip(lower=0))
                
        else:  # abs_s >= 1.0
            action = 'log1p'
            if s < 0:
                df[col] = -(np.log1p(-df[col] + df[col].max() + 1))
            else:
                df[col] = np.log1p(df[col].clip(lower=0))
        
        transform_log.append({
            'feature': col, 
            'skewness': round(s, 3), 
            'transform': action
        })
    
    log_df = pd.DataFrame(transform_log).sort_values('skewness', ascending=False)
    print(log_df.to_string(index=False))
    
    return df

def extract_participant_ppgr_phenotypes(merged_df, glucose_col="Glucose_Value", 
                                       time_col="time_since_meal_min",
                                       carb_col="carbs_g"):
    
    '''
    extracts the response behavioral features for clustering
    input: the meal-2-hour window merged dataset 
    '''
    participant_results = []

    for p_id, p_df in merged_df.groupby("Participant_id"):
        meal_metrics_list = []
        
        for m_id, m_df in p_df.groupby("meal_id"):
            
            carbs_g = m_df[carb_col].iloc[0]
            
            # PPGR (iAUC, Peak, Time to Peak)
            iauc, peak_inc, t_peak = compute_ppgr_metrics(m_df, glucose_col, time_col)
            
            # Carb Sensitivity: iAUC per gram of carb
            carb_sens = iauc / (carbs_g + 1e-6) # 1e-6 to prevents division by zero if a meal had 0 carbs
            
            # Recovery Rate : going back to baseline
            peak_val = m_df[glucose_col].max()
            final_val = m_df[glucose_col].iloc[-1]
            t_peak_actual = m_df.loc[m_df[glucose_col].idxmax(), time_col]
            t_final = m_df[time_col].max()
            time_to_recover = t_final - t_peak_actual
            rec_rate = (peak_val - final_val) / time_to_recover if time_to_recover > 0 else 0
            
            meal_metrics_list.append({
                'iauc': iauc,
                'carb_sensitivity': carb_sens,
                'recovery_rate': rec_rate,
                't_peak': t_peak
            })
        
        if meal_metrics_list:
            p_meal_stats = pd.DataFrame(meal_metrics_list)
            participant_results.append({
                'Participant_id': p_id,
                'avg_iauc': p_meal_stats['iauc'].mean(),
                'avg_carb_sensitivity': p_meal_stats['carb_sensitivity'].mean(),
                'avg_recovery_rate': p_meal_stats['recovery_rate'].mean(),
                'avg_time_to_peak': p_meal_stats['t_peak'].mean(),
                'iauc_standard_dev': p_meal_stats['iauc'].std() 
            })

    return pd.DataFrame(participant_results)

#-----------------------------------------------------------------------

def reduce(features, n_components=None, variance_threshold=0.05, target_variance=0.80):

    '''
    Fits PCA: first chooses n_component (if not given), 
    The choice of n_component depends on the target_variance: the number of principal components 
    which explain at least the target variance cumulatively
    Then runs PCA and prints also the top driver of each principal component
    '''
    X = StandardScaler().fit_transform(features)
    
    # Remove features that are nearly constant
    selector = VarianceThreshold(threshold=variance_threshold)
    X_filtered = selector.fit_transform(X)
    kept_cols = features.columns[selector.get_support()]
    
    print(f"Dropped {features.shape[1] - X_filtered.shape[1]} low-variance features")
    print(f"Kept: {list(kept_cols)}")
    
    # Select n_components if not specified, cut from the point (# of components) when cumulative variance reachs target
    if n_components is None:
        pca_full = PCA(random_state=42).fit(X_filtered)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        
        # Variance threshold
        n_components = int(np.argmax(cumvar >= target_variance) + 1)
    
        marginal = pca_full.explained_variance_ratio_
        
        print(f"\n ---Component Selection ---")
        print(f"Selected n_components = {n_components}")

        #Scree Plot for n_component selection
        fig, ax1 = plt.subplots(figsize=(8, 4))

        ax1.bar(
            range(1, len(marginal) + 1), marginal,
            color='#2EC4B6', alpha=0.7, label='Individual variance'
        )
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.tick_params(axis='y')
        ax1.set_ylim(0, max(marginal) * 1.4)
        ax2 = ax1.twinx()
        ax2.plot(
            range(1, len(cumvar) + 1), cumvar,
            color='#9B5DE5', marker='o', markersize=4,
            linewidth=1.8, label='Cumulative variance'
        )
        ax2.axhline(
            target_variance, color='#9B5DE5', linestyle='--',
            linewidth=1, alpha=0.5, label=f'Target: {target_variance:.0%}'
        )
        ax2.axvline(
            n_components, color='#3A86FF', linestyle='--',
            linewidth=1.2, alpha=0.7, label=f'Selected: {n_components}'
        )
        ax2.set_ylabel('Cumulative Variance')
        ax2.tick_params(axis='y')
        ax2.set_ylim(0, 1.05)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)

        sns.despine(fig=fig, right=False)
        plt.title('Scree Plot — PCA Component Selection', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_filtered)
    
    cumvar_final = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nFinal: {n_components} components explain {cumvar_final[-1]:.1%} of variance")
    
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=kept_cols
    )

    print("\n--- Top Driver for each Component ---")
    print(loadings.abs().idxmax())
    
    return X_pca, loadings, pca

def inspect_k(data, k_range=range(2, 10), is_pca=True):
    
    '''
    inspects the best k for clusters, plots dendogram and silhouette coeff plot
    for visual inspection and justification of the number of clusters chosen
    '''
    if is_pca:
        X = data
    else:
        # If no pca performed before drop ID and Scale it
        numeric_data = data.drop(columns=['Participant_id'], errors='ignore')
        X = StandardScaler().fit_transform(numeric_data)

    #Linkage matrix for the Dendrogram
    Z = linkage(X, method='ward')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Dendrogram 
    dendrogram(Z, ax=ax1, truncate_mode='lastp') 
    ax1.set_title('Dendrogram (Ward)')
    ax1.set_xlabel('Participant Clusters')
    ax1.set_ylabel('Distance')

    # Silhouette scores calculation
    scores = {k: silhouette_score(X, AgglomerativeClustering(k, linkage='ward').fit_predict(X))
              for k in k_range}
    
    # Silhouette plot
    ax2.plot(list(scores.keys()), list(scores.values()), marker='o')
    best_k = max(scores, key=scores.get)
    ax2.set_title(f'Silhouette score (Best k={best_k})')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.axvline(x=best_k, color='red', linestyle='--')
    plt.tight_layout()
    plt.show()
    return scores

def assign_clusters(profile_df, X_pca=None, k=3):
    """
    assigns the participants into clusters using AgglomerativeClustering
    
    input: give the best k from the inspect_k function's output

    profile_df: The df containing Participant_id (!)
    X_pca: array from PCA if it is already run (can be none)
    k: Number of clusters.
    """
    if X_pca is not None:
        # If using PCA, it's already an array/signal, so we use it directly
        data_for_clustering = X_pca
    else:
        # If NO PCA, first prepares the df:
        numeric_data = profile_df.drop(columns=['Participant_id'], errors='ignore')
        scaler = StandardScaler()
        data_for_clustering = scaler.fit_transform(numeric_data)
    
    # Clustering
    model = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = model.fit_predict(data_for_clustering)
    
    result_df = profile_df.copy()
    result_df['cluster'] = labels
    
    return result_df

def profile_clusters(clustered_df):
    '''
    gives the profiles/descriptions of the clusters as the means of the numeric columns
    '''
    numeric_cols = clustered_df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'cluster']
    return clustered_df.groupby('cluster')[numeric_cols].mean().round(2)

# ==============================================================
# ==============================================================
# ANALYSIS PREP FUNCTIONS
# ==============================================================
# ==============================================================
def compute_ppgr_metrics(df,glucose_col="Glucose_Value", time_col="time_since_meal_min", response_window=(0, 120)):
    """
    Computes incremental AUC (iAUC) for one meal window.
    So only the area under the baseline glucose counts (otherwise (if not incremental and 
    goes below baseline at some point) might cancel out)
    """  
    df = df.sort_values(time_col).copy()
    
    df = df[
        (df[time_col] >= response_window[0]) &
        (df[time_col] <= response_window[1])
    ].copy()
    
    if df.empty:
        return np.nan
    
    # Baseline glucose defined as the first value after the meal time
    baseline = df.iloc[0][glucose_col]
    
    #Compute incremental glucose
    df["increment"] = df[glucose_col] - baseline
    df["increment"] = df["increment"].clip(lower=0)
    peak = df["increment"].max()
    time_to_peak = df.loc[df["increment"].idxmax(), time_col]
    
    # Integrate using trapezoidal rule, area under the curve
    iauc = np.trapz(df["increment"], df[time_col])
    
    return iauc, peak, time_to_peak
# ==============================================================
def compute_ppgr_metrics_all_meals(cgm_meal_df, meal_id_col="meal_id", glucose_col="Glucose_Value", time_col="time_since_meal_min"):
    '''
    above function, looping over all meals
    '''
    results = []
    
    for meal_id, meal_df in cgm_meal_df.groupby(meal_id_col):
        iauc, peak, time_to_peak = compute_ppgr_metrics(
            meal_df,
            glucose_col=glucose_col,
            time_col=time_col
        )
        results.append({
            "meal_id": meal_id,
            "iauc": iauc,
            "peak_increment": peak,
            "time_to_peak": time_to_peak
        })
    
    return pd.DataFrame(results)
# ==============================================================
def engineer_circadian_feature(df, hour_col='time_hour', acro_col='acrophase', period='period'):
    """
    adds an interaction feature meal_dist_to_circadian: representing how far the meal is from the person's peak
    hour_col: Meal time in decimal hours (0-24)
    acro_col: Participant's Acrophase in Radians (roughly -3.14 to 3.14)
    """
    #Convert Meal Hour to Radians (between 0 and 2pi)
    meal_rad = (df[hour_col] / 24) * (2 * np.pi)
    
    # Calculate the difference
    # This represents how far the meal is from the person's peak
    phase_diff = meal_rad - df[acro_col]
    
    # Apply Cosine transformation
    # Result is 1.0 if meal is at acrophase, -1.0 if at opposite time
    df['meal_dist_to_circadian'] = np.cos(phase_diff)
    df['meal_time_rad']=meal_rad
    df['sin_time'] = np.sin(2 * np.pi * df[hour_col] / df[period])
    df['cos_time'] = np.cos(2 * np.pi * df[hour_col] / df[period])

    return df
# ==============================================================
def _compute_metrics(y_true, y_pred):
    '''
    helper function for the evaluation metrics
    '''
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae
# ==============================================================
def clean_feature_names(tuned_df):
    tuned_df = tuned_df.copy()
    tuned_df.columns = (
        tuned_df.columns
        .astype(str)
        .str.replace("<", "lt_", regex=False)
        .str.replace(">", "gt_", regex=False)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    return tuned_df
# ==============================================================
# ==============================================================
def build_feature_sets_task1(df, feature_sets):
    '''
    prepare datasets from feature subset combinations
    '''
    df = df.copy()
    return {label: df[cols] for label, cols in feature_sets.items()}, df

# get_models_and_params_1 is a smaller hyperparameter space, which is selected after trying
# out both and seeing the broader one does not improve the models significantly
# so the second space is not used
# ==============================================================
def get_models_and_params_1(random_state=42):
    ''''
    hyperparameter search spaces for the algorithms used in the iAUC Prediction
    '''
    models = {
        "LinearRegression": (
            LinearRegression(),
            {}
        ),

        "Ridge": (
            Ridge(random_state=random_state),
            {
                "model__alpha": np.logspace(-3, 3, 20)
            }
        ),

        "RandomForest": (
            RandomForestRegressor(random_state=random_state),
            {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 5]
            }
        ),

        "GradientBoosting": (
            GradientBoostingRegressor(random_state=random_state),
            {
                "model__n_estimators": [100, 300, 500],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [3, 5]
            }
        ),

        "HistGradientBoosting": (
            HistGradientBoostingRegressor(random_state=random_state),
            {
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [None, 5, 10],
                "model__max_iter": [200, 400]
            }
        ),

        "XGBoost": (
            XGBRegressor(
                random_state=random_state,
                objective="reg:squarederror",
                verbosity=0
            ),
            {
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [3, 5, 7],
                "model__subsample": [0.8, 1.0]
            }
        )
    }

    return models

def get_models_and_params_2(random_state=42):
    models = {
        "LinearRegression": (
            LinearRegression(),
            {}
        ),

        "Ridge": (
            Ridge(random_state=random_state),
            {
                "model__alpha": np.logspace(-4, 4, 50) 
            }
        ),

        "RandomForest": (
            RandomForestRegressor(random_state=random_state, n_jobs=-1),
            {
                "model__n_estimators": [100, 300, 500],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None]
            }
        ),

        "GradientBoosting": (
            GradientBoostingRegressor(random_state=random_state),
            {
                "model__n_estimators": [100, 300, 500],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [3, 4, 5],
                "model__subsample": [0.8, 1.0]
            }
        ),

        "HistGradientBoosting": (
            HistGradientBoostingRegressor(random_state=random_state),
            {
                "model__learning_rate": np.logspace(-3, -1, 10), 
                "model__max_iter": [100, 200, 300, 400, 500],
                "model__max_depth": [3, 5, 10],
                "model__l2_regularization": np.logspace(-2, 2, 5),
                "model__max_leaf_nodes": [15, 31, 63]
            }
        ),

        "XGBoost": (
            XGBRegressor(
                random_state=random_state,
                objective="reg:squarederror",
                tree_method="hist",
                verbosity=0
            ),
            {
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_depth": [3, 6, 9],
                "model__subsample": [0.6, 0.8, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
                "model__gamma": [0, 0.1, 0.2]
            }
        )
    }

    return models

# ==============================================================
# ==============================================================
# MAIN ANALYSIS - iAUC PREDICTION - APPROACH 1: GLOBAL MODEL
# ==============================================================
# ==============================================================

def global_model(
    X,
    y,
    subset_name,
    n_iter=20,
    cv_splits=5,   #here with KFold, later also with LOPO_CV
    cv_repeats=2,
    random_state=42,
    return_best_estimator=False,
    algorithm=None
):
    
    '''
    First global model runs with Kfold, over full dataset, no personalisation
    Algorithms and hyperparameter search spaces are specified above
    For the given /current (if looping) subset it runs all models and gives the best performed
    '''
    
    results = []
    #if an algorithm is specified only pulls its parameters else all
    all_models = get_models_and_params_1(random_state=random_state)
    if algorithm is not None:
        models = {algorithm: all_models[algorithm]}
    else:
        models = all_models
    #CV folds
    cv = RepeatedKFold(
        n_splits=cv_splits,
        n_repeats=cv_repeats,
        random_state=random_state
    )
    print(f"Running GLOBAL MODEL")
    print(f"Subset: {subset_name}")
    print(f"Dataset size: {X.shape}")
    print("==============================")

    best_overall_score = -np.inf         
    best_overall_estimator = None         
    best_overall_name = None  

    for model_name, (model, param_dist) in models.items():
        
        print(f"Model: {model_name}")
        
        if model_name in ["LinearRegression", "Ridge"]:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])
        else:
            pipeline = Pipeline([
                ("model", model)
            ])

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=n_iter if param_dist else 1,
            cv=cv,
            scoring={                              
                "r2": "r2",
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error"
            },
            refit="r2",
            n_jobs=-1,
            random_state=random_state
        )

        search.fit(X, y)
        best_index = search.best_index_
        mean_r2 = search.cv_results_["mean_test_r2"][best_index]
        std_r2 = search.cv_results_["std_test_r2"][best_index]
        mean_rmse = -search.cv_results_["mean_test_rmse"][best_index]  
        std_rmse = search.cv_results_["std_test_rmse"][best_index] 
        mean_mae = -search.cv_results_["mean_test_mae"][best_index]
        std_mae = search.cv_results_["std_test_mae"][best_index]

        print("\n--------------------------------")
        print("Model:", model_name)
        print(f"R² : {mean_r2:.3f} ±{std_r2:.3f}")
        print(f"RMSE : {mean_rmse:.3f} ±{std_rmse:.3f}")
        print(f"MAE : {mean_mae:.3f}±{std_mae:.3f}")
        print(f"Params: {search.best_params_}")

        results.append({
            "Subset": subset_name,
            "Model": model_name,
            "Best_R2": mean_r2,
            "Std_R2": std_r2,
            "RMSE": mean_rmse, 
            "Std_RMSE": std_rmse,     
            "MAE": mean_mae,   
            "Std_MAE": std_mae,    
            "Best_Params": search.best_params_
        })

        if mean_r2 > best_overall_score:                         
            best_overall_score = mean_r2                     
            best_overall_estimator = search.best_estimator_        
            best_overall_name = model_name  

    results_df = pd.DataFrame(results).sort_values("Best_R2", ascending=False)

    if return_best_estimator:                                        
        best_overall_estimator.fit(X, y)                            
        print(f"\n  Best overall: {best_overall_name} "             
            f"(R²={best_overall_score:.3f}) - refit on full data") 
        return results_df, best_overall_estimator                   

    return results_df

#-----------------------------------------------------------------------

def global_model_with_selection(
    X,
    y,
    subset_name="Importance - Selected Features",
    n_top_features=20, 
    n_iter=20,
    cv_splits=5,
    cv_repeats=2,
    random_state=42
):  
    '''
    Same as above, only instead of the given feature subsets, it first selects the 
    most important 20 features with Random Forest and uses that feature subset 
    to train the same models in the same way
    '''
    results = []

    if "Participant_id" in X.columns:
        X = X.drop(columns=["Participant_id"])

    models = get_models_and_params_1(random_state=random_state)

    cv = RepeatedKFold(
        n_splits=cv_splits,
        n_repeats=cv_repeats,
        random_state=random_state
    )

    # Feature selector
    selector = SelectFromModel(
        RandomForestRegressor(n_estimators=100, random_state=random_state),
        max_features=n_top_features,
        threshold=-np.inf 
    )

    for model_name, (model, param_dist) in models.items():
        print(f"Model: {model_name}")

        if model_name in ["LinearRegression", "Ridge"]:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("feature_selection", selector),
                ("model", model) 
            ])
        else:
            pipeline = Pipeline([
                ("feature_selection", selector),
                ("model", model) 
            ])

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=n_iter if param_dist else 1,
            cv=cv,
            scoring={
                "r2": "r2",
                "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error"
            },
            refit="r2",
            n_jobs=-1,
            random_state=random_state
        )

        search.fit(X, y)
        
        best_pipeline = search.best_estimator_
        best_selector = best_pipeline.named_steps['feature_selection']
        
        # gets the 'importance' values from the underlying RandomForest 
        importances = best_selector.estimator_.feature_importances_
        #kept features 
        selected_mask = best_selector.get_support()
        
        feature_importance_series = pd.Series(importances, index=X.columns)
        selected_importances = (
            feature_importance_series[selected_mask]
            .sort_values(ascending=False)
        )
        best_index = search.best_index_
        mean_r2 = search.cv_results_["mean_test_r2"][best_index]
        std_r2 = search.cv_results_["std_test_r2"][best_index]
        mean_rmse = -search.cv_results_["mean_test_rmse"][best_index]  
        std_rmse = search.cv_results_["std_test_rmse"][best_index] 
        mean_mae = -search.cv_results_["mean_test_mae"][best_index]
        std_mae = search.cv_results_["std_test_mae"][best_index]


        results.append({
            "Subset": subset_name,
            "Model": model_name,
            "Best_R2": mean_r2,
            "Std_R2": std_r2,
            "RMSE": mean_rmse, 
            "Std_RMSE": std_rmse,        
            "MAE": mean_mae,
            "Std_MAE": std_mae,
            "Best_Params": search.best_params_,
            "Selected_Features": selected_importances.index.tolist(),
            "Feature_Importances": selected_importances.values.tolist()
        })

    return pd.DataFrame(results).sort_values("Best_R2", ascending=False)

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
# MAIN ANALYSIS - iAUC PREDICTION - APPROACH 2: PER PERSON MODELS
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#personalized models, on the pre-defined feature subsets, 
#hyperparameter tuning also per person
#-----------------------------------------------------------------------

def personalized_models(X, y,
                        participant_ids,
                        subset_name,
                        n_iter=20,
                        cv_splits=5,
                        random_state=42
                    ):
    '''
    The global model architecture (same algorithms and same hyperparameter search spaces)
    is applied on personal level. This function trains a model for each person, using only
    that person's data
    '''

    results = []
    models = get_models_and_params_1(random_state=random_state)

    print("Running PERSONALIZED MODELS")
    print(f"Subset: {subset_name}")
    print("==============================")

    for pid in participant_ids:

        X_p = X[X["Participant_id"] == pid].drop(columns=["Participant_id"])
        y_p = y.loc[X_p.index]

        print(f"\nParticipant: {pid} | Samples: {len(X_p)}")

        cv = KFold(
            n_splits=cv_splits,
            shuffle=True,
            random_state=random_state
        )

        for model_name, (model, param_dist) in models.items():

            print(f"Model: {model_name}")

            if model_name in ["LinearRegression", "Ridge"]:
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", model)
                ])
            else:
                pipeline = Pipeline([
                    ("model", model)
                ])

            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_dist,
                n_iter=n_iter if param_dist else 1,
                cv=cv,
                scoring={                          
                    "r2": "r2",
                    "rmse":"neg_root_mean_squared_error",
                    "mae": "neg_mean_absolute_error"
                },
                refit="r2",
                n_jobs=-1,
                random_state=random_state
            )

            search.fit(X_p, y_p)

            best_index = search.best_index_
            mean_r2 = search.cv_results_["mean_test_r2"][best_index]
            std_r2 = search.cv_results_["std_test_r2"][best_index]
            mean_rmse = -search.cv_results_["mean_test_rmse"][best_index]
            std_rmse = search.cv_results_["std_test_rmse"][best_index]
            mean_mae = -search.cv_results_["mean_test_mae"][best_index]
            std_mae = search.cv_results_["std_test_mae"][best_index]

            results.append({
                "Participant": pid,
                "Number of Meals": len(X_p),
                "Subset": subset_name,
                "Model": model_name,
                "Best_R2": mean_r2,
                "Std_R2": std_r2,
                "RMSE": mean_rmse,   
                "Std_RMSE": std_rmse,   
                "MAE": mean_mae,    
                "Std_MAE": std_mae,   
                "Best_Params": search.best_params_
            })

    return pd.DataFrame(results)

#-----------------------------------------------------------------------

def personalized_models_with_selection(X,y,
                        participant_ids,
                        n_top_features=10, # training each participant's model with their most effective 10 features
                        n_iter=20,
                        cv_splits=5,
                        random_state=42
                    ):
    '''
    Personalized models with top 10 most importantfeatures selected for each participant
    by Random forest. 
    '''
    results = []
    models = get_models_and_params_1(random_state=random_state)

    for pid in participant_ids:
        X_p_full = X[X["Participant_id"] == pid].drop(columns=["Participant_id"])
        y_p = y.loc[X_p_full.index]
        
        #feature selection
        selector = RandomForestRegressor(n_estimators=100, random_state=random_state)
        selector.fit(X_p_full, y_p)
        
        importances = pd.Series(selector.feature_importances_, index=X_p_full.columns)
        top_features = importances.sort_values(ascending=False).head(n_top_features).index.tolist()

        X_p = X_p_full[top_features]

        print(f"\nParticipant: {pid} | Samples: {len(X_p)} | Top Features: {top_features[:3]}")

        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

        for model_name, (model, param_dist) in models.items():

            if model_name in ["LinearRegression", "Ridge"]:
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", model)
                ])
            else:
                pipeline = Pipeline([("model", model)])

            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_dist,
                n_iter=n_iter if param_dist else 1,
                cv=cv,
                scoring={         
                    "r2": "r2",
                    "rmse":"neg_root_mean_squared_error",
                    "mae": "neg_mean_absolute_error"
                },
                refit="r2",
                n_jobs=-1,
                random_state=random_state
            )
            search.fit(X_p, y_p)

            best_index = search.best_index_
            mean_r2 = search.cv_results_["mean_test_r2"][best_index]
            std_r2 = search.cv_results_["std_test_r2"][best_index]
            mean_rmse = -search.cv_results_["mean_test_rmse"][best_index]
            std_rmse = search.cv_results_["std_test_rmse"][best_index]
            mean_mae = -search.cv_results_["mean_test_mae"][best_index]
            std_mae = search.cv_results_["std_test_mae"][best_index]

            results.append({
                "Participant": pid,
                "Sample_Size":len(X_p),
                "Model": model_name,
                "Best_R2(Mean)": mean_r2,
                "Std_R2": std_r2,
                "RMSE": mean_rmse,
                "Std_RMSE": std_rmse,          
                "MAE": mean_mae, 
                "Std_MAE": std_mae,          
                "Best_Params": search.best_params_,
                "Selected_Features": top_features
            })

    return pd.DataFrame(results)

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
# MAIN ANALYSIS - iAUC PREDICTION - APPROACH 3: RESIDUAL CORRECTION FUSION
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
def log_fusion_result(results, pid, subset_name, model_name,
                       y_true, y_pred, alpha_blend, n_p,
                       corrector_cv_r2=None, note=""):
    
    '''
    Evaluates the fusion model and logs the results. (Helper to simplfy the fusion function below)
    '''
    r2, rmse, mae = _compute_metrics(y_true, y_pred)  

    print(f"Fusion R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}"  
          + (f"[{note}]" if note else ""))

    results.append({
        "Participant": pid,
        "Subset": subset_name,
        "Global_Model": model_name,
        "Fusion_R2": r2,
        "Fusion_RMSE": rmse,           
        "Fusion_MAE": mae,            
        "Alpha_Blend": alpha_blend,
        "Corrector_CV_R2": corrector_cv_r2,
        "N_Samples": n_p,
        "Note": note
    })

def fusion_model_loo(
    X,
    y,
    participant_ids,
    subset_name,
    n_iter=20,
    cv_splits_global=5,
    cv_repeats_global=2,
    cv_splits_personal=5,
    min_samples_personal=10,
    corrector_alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
    random_state=42,
    algorithm=None
):
    '''
    residual correction fusion approach on the global model:
    1- Global model is trained using LOPO-CV (full feature subset + hyperparameter search each time)
    2- Personal Ridge corrector is trained on that participant's residuals from their own leave-one-out global model

    This means each participant's correction is built on top of a global model that never saw them.
    Fusion falls back to global if there are less than 10 samples, and uses Leave One Out CV if less the 
    cv_splits_personal (here 5)*3. Otherwise 5Fold CV

    Returns 3 outputs:
    results_df : per-participant fusion results 
    correctors : dict {pid: {'corrector': Ridge|None, 'alpha_blend': float,'global_pipeline': fitted pipeline}}
    global_on_personal_df : global model performance per participant (from their LOPO global model)
    '''

    X_global = X.drop(columns=["Participant_id"])

    print("\n" + "=" * 60)
    print(f"FUSION MODEL (LOO-Global)  |  Subset: {subset_name}")
    print(f"Dataset: {X_global.shape}  |  Participants: {len(participant_ids)}")
    print("=" * 60)

    results = []
    correctors = {}
    global_on_personal_rows = []

    for pid in participant_ids:
        mask_pid = (X["Participant_id"] == pid).values
        mask_rest = ~mask_pid

        X_p = X_global[mask_pid]
        y_p = y.values[mask_pid]
        n_p = len(X_p)

        # Everyone except this participant
        X_rest = X_global[mask_rest].reset_index(drop=True)
        y_rest = y[mask_rest].reset_index(drop=True)

        print("\n" + "-" * 60)
        print(f"Participant {pid} | n_personal={n_p} | n_global_train={len(X_rest)}")

        # 1- Global model trained WITHOUT this participant 
        print(f" Global model selection (excluding pid={pid})")
        global_results_df, global_pipeline_pid = global_model(
            X=X_rest,
            y=y_rest,
            subset_name=f"{subset_name}_excl_{pid}",
            n_iter=n_iter,
            cv_splits=cv_splits_global,
            cv_repeats=cv_repeats_global,
            random_state=random_state,
            return_best_estimator=True,
            algorithm=algorithm
        )
        best_global_name = global_results_df.iloc[0]["Model"]
        best_global_r2 = global_results_df.iloc[0]["Best_R2"]

        # Global predictions on this unseen participant
        global_preds_p = global_pipeline_pid.predict(X_p)

        global_r2_p = r2_score(y_p, global_preds_p)
        global_rmse_p = np.sqrt(mean_squared_error(y_p, global_preds_p))
        global_mae_p = mean_absolute_error(y_p, global_preds_p)

        print(f"Global on pid={pid}: "
              f"R²={global_r2_p:.3f}  RMSE={global_rmse_p:.3f}  MAE={global_mae_p:.3f}")

        global_on_personal_rows.append({
            "Participant": pid,
            "Subset": subset_name,
            "Global_Model": best_global_name,
            "Global_R2": global_r2_p,
            "Global_RMSE": global_rmse_p,
            "Global_MAE": global_mae_p,
            "N_Samples": n_p,
        })

        # Residuals on this participant's data
        residuals_p = y_p - global_preds_p   # ε = y_true − ŷ_global

        # 2- Residual correction
        if n_p < min_samples_personal:
            print(f"n < {min_samples_personal}: global prediction only (α=0)")
            correctors[pid] = {
                "corrector": None,
                "alpha_blend": 0.0,
                "global_pipeline": global_pipeline_pid,
            }
            log_fusion_result(results, pid, subset_name, best_global_name,
                              y_p, global_preds_p, alpha_blend=0.0, n_p=n_p,
                              note="global-fallback")
            continue

        if n_p < cv_splits_personal * 3:
            cv_personal = LeaveOneOut()
            print(f"LOO CV for corrector (n={n_p} is small)")
        else:
            cv_personal = KFold(n_splits=cv_splits_personal,
                                shuffle=True, random_state=random_state)

        corrector_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ])
        alpha_grid = {"ridge__alpha": list(corrector_alphas)}

        try:
            corrector_search = RandomizedSearchCV(
                corrector_pipeline,
                param_distributions=alpha_grid,
                n_iter=min(len(corrector_alphas), n_p),
                cv=cv_personal,
                scoring="r2",
                n_jobs=-1,
                random_state=random_state,
                error_score=np.nan,
            )
            corrector_search.fit(X_p, residuals_p)
            best_corrector = corrector_search.best_estimator_
        except Exception as e:
            print(f"Corrector failed ({e}): falling back to global")
            correctors[pid] = {
                "corrector": None,
                "alpha_blend": 0.0,
                "global_pipeline": global_pipeline_pid,
            }
            log_fusion_result(results, pid, subset_name, best_global_name,
                              y_p, global_preds_p, alpha_blend=0.0, n_p=n_p,
                              note="corrector-failed")
            continue

        try:
            oof_corrector_preds_p = cross_val_predict(
                best_corrector, X_p, residuals_p, cv=cv_personal, n_jobs=-1
            )
        except Exception:
            oof_corrector_preds_p = best_corrector.predict(X_p)

        # α-blend: how much to use the corrector
        # α→0: person matches population; α>>: high personal correction
        if len(residuals_p) >= 3:
            alpha_ridge = Ridge(alpha=1.0, fit_intercept=False)
            alpha_ridge.fit(oof_corrector_preds_p.reshape(-1, 1), residuals_p)
            alpha_blend = float(np.clip(alpha_ridge.coef_[0], 0.0, 2.0))
        else:
            alpha_blend = 0.0

        print(f"α={alpha_blend:.3f}  corrector_cv_R²={corrector_search.best_score_:.3f}")

        # Refit corrector on all personal data
        best_corrector.fit(X_p, residuals_p)
        correctors[pid] = {
            "corrector": best_corrector,
            "alpha_blend": alpha_blend,
            "global_pipeline": global_pipeline_pid,
        }

        # Final fusion prediction
        fusion_pred_p = global_preds_p + alpha_blend * oof_corrector_preds_p
        log_fusion_result(results, pid, subset_name, best_global_name,
                          y_p, fusion_pred_p,
                          alpha_blend=alpha_blend, n_p=n_p,
                          corrector_cv_r2=corrector_search.best_score_)

    results_df = pd.DataFrame(results).reset_index(drop=True)
    global_on_personal_df = pd.DataFrame(global_on_personal_rows)

    print("\n" + "=" * 60)
    print("FUSION LOO SUMMARY")
    print(f"Mean R²: {results_df['Fusion_R2'].mean():.3f} ± {results_df['Fusion_R2'].std():.3f}")
    print(f"Mean RMSE: {results_df['Fusion_RMSE'].mean():.3f}")
    print(f"Mean MAE : {results_df['Fusion_MAE'].mean():.3f}")
    print("=" * 60)

    return results_df, correctors, global_on_personal_df

def compare_all_models(
    global_on_personal_df, # from fusion_model_loo()
    personal_results_df,  # from personalized_models()
    fusion_results_df,  # from fusion_model_loo()
    subset_name="Meal+Meta+Glucose+Circadian", 
    best_personal_model='XGBoost'
                              
):
    """
    Compares the per person, global and fusion on global models
    For each person compares the performance of the selected best subset and algortihm
    Gives an overall comparison table across three models
    """

    # 1- Global LOPO-CV
    g = global_on_personal_df[global_on_personal_df["Subset"] == subset_name][["Participant", "Global_R2", "Global_RMSE", "Global_MAE", "N_Samples"]]

    # 2- Personal
    pers = personal_results_df[personal_results_df["Subset"] == subset_name]
    pers = pers[pers["Model"] == best_personal_model]

    pers = pers[["Participant", "Best_R2", "RMSE", "MAE"]].rename(columns={
        "Best_R2": "Personal_R2",
        "RMSE": "Personal_RMSE",
        "MAE":"Personal_MAE"
    })

    # 3- Residual Correction Fusion
    fus = fusion_results_df[fusion_results_df["Subset"] == subset_name][["Participant", "Fusion_R2", "Fusion_RMSE", "Fusion_MAE", "Alpha_Blend"]]

    #Merge
    comparison = (
        g.merge(pers, on="Participant", how="outer")
         .merge(fus, on="Participant", how="outer")
         .sort_values("Fusion_R2", ascending=False)
         .reset_index(drop=True)
    )

    # Deltas: how much each model improves over global
    comparison["ΔR²_personal_vs_global"] = (comparison["Personal_R2"] - comparison["Global_R2"])
    comparison["ΔR²_fusion_vs_global"] = (comparison["Fusion_R2"] - comparison["Global_R2"])
    comparison["ΔR²_fusion_vs_personal"] = (comparison["Fusion_R2"] - comparison["Personal_R2"])

    # Summary
    print(f"\n{'='*65}")
    print(f"COMPARISON SUMMARY  |  Subset: {subset_name}")
    print(f"{'='*65}")
    print(f"Mean Global R² (on person (LOPO-CV)): "
          f"{comparison['Global_R2'].mean():.3f}")
    print(f"Mean Personal R²                   : "
          f"{comparison['Personal_R2'].mean():.3f}")
    print(f"Mean Fusion R²                     : "
          f"{comparison['Fusion_R2'].mean():.3f}")
    print(f"\nMean ΔR² personal vs global : "
          f"{comparison['ΔR²_personal_vs_global'].mean():+.3f}")
    print(f"Mean ΔR² fusion vs global : "
          f"{comparison['ΔR²_fusion_vs_global'].mean():+.3f}")
    print(f"Mean ΔR² fusion vs personal: "
          f"{comparison['ΔR²_fusion_vs_personal'].mean():+.3f}")
    print(f"\nParticipants where fusion > global  : "
          f"{(comparison['ΔR²_fusion_vs_global'] > 0).sum()} / {len(comparison)}")
    print(f"Participants where fusion > personal: "
          f"{(comparison['ΔR²_fusion_vs_personal'] > 0).sum()} / {len(comparison)}")
    print(f"{'='*65}")

    return comparison

#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
# #MAIN ANALYSIS - TIME SERIES PREDICTION
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

# Data prep
#-----------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_window_dataset(
    meal_events_df,
    cgm_df,
    cgm_summary_df,
    metadata_df,
    pre_window_min=60,
    post_window_min=120,
    resample_interval_min=15,
    min_post_points=6,
    min_pre_points=2,
    max_gap_min=25,
):
    '''
    Some additional exclusion is done to obtain isolated meals and thus cleaner repsonse curves
    Some meals were merged, the features are either summed or if categarical the dominant entry is passed:
        - i.e. diet type if one of the meals is monivore other vegan, merged is omnivore etc
    Glycemic index is averaged using the weighted avg formula

    Then the CGM points are checked, if there are enough pre and post poinst
    when there are gaps: if the gaps are reasonable (interpolatable)
    if we have the baseline (there or interpolated)

    overall exclusion criteria:
      1. Meal logged <= 30 min after previous meal -> merge into first meal
      2. Meal within 30-90 min of next meal -> remove both
      3. No CGM data in window
      4. Insufficient raw post-meal readings (< 2)
      5. CGM gap > max_gap_min in post-meal window
      6. NaN after interpolation (pre or post grid)
      7. Baseline at t=0 could not be interpolated
    '''

    #Numeric columns: summed when merging 
    sum_cols = ["total_calories", "carbs_g", "protein_g", "fat_g", "fiber_g",
                "estimated_total_weight_g"]

    # Categorical columns
    # Higher value = more processed;  diet_type: omnivore=2 > vegetarian=1 > vegan=0...
    max_cols = ["ultra_processed", "food_processing_level", "added_sugar", "diet_type"]

    # ── GI: carb-weighted average ─────────────────────────────────────────────
    # handled separately

    meal_feature_cols = [
        "total_calories", "carbs_g", "protein_g", "fat_g", "fiber_g",
        "glycemic_index", "estimated_total_weight_g", "diet_type",
        "ultra_processed", "food_processing_level", "added_sugar", "fluid_or_solid"
    ]
    meal_feature_cols = [c for c in meal_feature_cols if c in meal_events_df.columns]
    sum_cols = [c for c in sum_cols  if c in meal_events_df.columns]
    max_cols = [c for c in max_cols  if c in meal_events_df.columns]

    summary_feature_cols = cgm_summary_df.columns
    metadata_cols= metadata_df.columns

    # Merging and removing
    meal_events_df = meal_events_df.sort_values(["Participant_id", "datetime"]).copy().reset_index(drop=True)

    def merge_meals(group):
        '''
        Pass 1: merge meals <= 30 min apart into one row (earliest time).
        Pass 2: remove merged meals that are 30-90 min apart from the next.
        '''
        rows = group.to_dict("records")
        merged = []

        # Pass 1: merge
        i = 0
        while i < len(rows):
            base = dict(rows[i])
            while i + 1 < len(rows):
                gap = (rows[i + 1]["datetime"] - base["datetime"]).total_seconds() / 60
                if gap <= 30:
                    next_row = rows[i + 1]
                    # sum numeric
                    for c in sum_cols:
                        base[c] = (base.get(c) or 0) + (next_row.get(c) or 0)
                    # carb-weighted GI
                    c1 = base.get("carbs_g", 0) or 0
                    c2 = next_row.get("carbs_g", 0) or 0
                    g1 = base.get("glycemic_index") or 0
                    g2 = next_row.get("glycemic_index") or 0
                    total_c = c1 + c2
                    base["glycemic_index"] = (
                        (g1 * c1 + g2 * c2) / total_c if total_c > 0 else (g1 + g2) / 2
                    )
                    # categorical
                    for c in max_cols:
                        v1 = base.get(c)
                        v2 = next_row.get(c)
                        if v1 is not None and v2 is not None:
                            try:
                                base[c] = max(v1, v2)
                            except TypeError:
                                pass 

                    # fluid_or_solid: solid (0) dominates fluid (1)
                    if "fluid_or_solid" in base and "fluid_or_solid" in next_row:
                        v1 = base.get("fluid_or_solid")
                        v2 = next_row.get("fluid_or_solid")
                        if v1 is not None and v2 is not None:
                            base["fluid_or_solid"] = min(v1, v2)  # 0=solid is dominant
                    i += 1
                else:
                    break
            merged.append(base)
            i += 1

        # Pass 2: remove pairs 30-90 min apart
        keep  = [True] * len(merged)
        for i in range(len(merged) - 1):
            if not keep[i]:
                continue
            gap = (merged[i + 1]["datetime"] - merged[i]["datetime"]).total_seconds() / 60
            if 30 < gap <= 90:
                keep[i] = False
                keep[i + 1] = False

        kept = [m for m, k in zip(merged, keep) if k]
        removed = [m for m, k in zip(merged, keep) if not k]
        return kept, removed

    all_kept = []
    all_removed = []
    n_merged = 0

    original_counts = meal_events_df.groupby("Participant_id").size().to_dict()

    for pid, group in meal_events_df.groupby("Participant_id"):
        kept, removed = merge_meals(group)
        n_original = original_counts[pid]
        n_after_merge = len(kept) + len(removed)  # after pass1, before pass2
        n_merged += (n_original - n_after_merge)   # meals absorbed by merging
        all_kept.extend(kept)
        all_removed.extend(removed)

    n_removed_overlap = len(all_removed)
    clean_meals = pd.DataFrame(all_kept).reset_index(drop=True)

    print(f"\n--- Meal pre-processing ---------------------------------")
    print(f"Original meal rows: {len(meal_events_df)}")
    print(f"Merged: {n_merged}(gap ≤ 30 min -> merged into earlier meal)")
    print(f"Removed (overlap 30-90m): {n_removed_overlap} (overlapping postprandial window)")
    print(f"Meals left: {len(clean_meals)}")

    # CGM window extraction (pre and post meal time points)
    post_grid = np.arange(resample_interval_min, post_window_min + 1, resample_interval_min)
    pre_grid = np.arange(-pre_window_min, 1, resample_interval_min)

    all_records = []
    exclusion_log = []

    for _, meal in clean_meals.iterrows():

        pid = meal["Participant_id"]
        meal_time = meal["datetime"]
        meal_id = meal.get("meal_id", _)

        window_start = meal_time - pd.Timedelta(minutes=pre_window_min)
        window_end = meal_time + pd.Timedelta(minutes=post_window_min)

        seg = cgm_df[
            (cgm_df["Participant_id"] == pid) &
            (cgm_df["datetime"] >= window_start) &
            (cgm_df["datetime"] <= window_end)
        ].copy()

        # If no CGM data
        if seg.empty:
            exclusion_log.append({"meal_id": meal_id, "reason": "no CGM data in window"})
            continue

        seg = seg.sort_values("datetime")
        seg["t_min"] = (seg["datetime"] - meal_time).dt.total_seconds() / 60.0

        # If insufficient post-meal points
        post_raw = seg[seg["t_min"] > 0]
        if len(post_raw) < min_post_points:
            exclusion_log.append({"meal_id": meal_id, "reason": f"insufficient post-meal CGM points (< {min_post_points} points)"})
            continue

        # insufficient pre-meal points
        pre_raw = seg[seg["t_min"] <= 0]
        if len(pre_raw) < min_pre_points:
            exclusion_log.append({"meal_id": meal_id, "reason": f"insufficient pre-meal CGM points (< {min_pre_points} points)"})
            continue

        # lrge CGM gap in post-meal window
        if (post_raw["t_min"].diff().dropna() > max_gap_min).any():
            exclusion_log.append({"meal_id": meal_id, "reason": f"CGM gap > {max_gap_min} min in post-meal window"})
            continue

        # Interpolate
        interp_fn = interp1d(
            seg["t_min"].values,
            seg["Glucose_Value"].values,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan
        )

        post_glucose = interp_fn(post_grid)
        pre_glucose = interp_fn(pre_grid)

        # if NaN after interpolation
        if np.isnan(post_glucose).any() or np.isnan(pre_glucose).any():
            exclusion_log.append({"meal_id": meal_id, "reason": "NaN after interpolation"})
            continue

        # if no point for baseline after interpolation
        baseline_glucose = float(interp_fn(0))
        if np.isnan(baseline_glucose):
            exclusion_log.append({"meal_id": meal_id, "reason": "baseline glucose at t=0 could not be interpolated"})
            continue

        # sequences
        post_incremental = post_glucose - baseline_glucose
        pre_incremental = pre_glucose - baseline_glucose

        # Pre-meal slope
        pre_valid_mask = ~np.isnan(pre_glucose)
        if pre_valid_mask.sum() >= 2:
            pre_slope = np.polyfit(pre_grid[pre_valid_mask], pre_glucose[pre_valid_mask], deg=1)[0]
        else:
            pre_slope = 0.0

        # Participant summary
        participant_summary = cgm_summary_df[cgm_summary_df["Participant_id"] == pid]
        if not participant_summary.empty:
            summary_feats = {col: participant_summary.iloc[0][col] for col in summary_feature_cols}
        else:
            summary_feats = {col: np.nan for col in summary_feature_cols}

        # Participant metadata
        participant_metadata = metadata_df[metadata_df["Participant_id"] == pid]
        if not participant_metadata.empty:
            meta = {col: participant_metadata.iloc[0][col] for col in metadata_cols}
        else:
            meta = {col: np.nan for col in metadata_cols}

        record = {
            "meal_id": meal_id,
            "participant_id": pid,
            "meal_time": meal_time,
            "pre_glucose_abs": pre_glucose.tolist(),
            "pre_glucose_inc": pre_incremental.tolist(),
            "post_glucose_abs": post_glucose.tolist(),
            "post_glucose_inc": post_incremental.tolist(),
            "pre_time_grid": pre_grid.tolist(),
            "post_time_grid":post_grid.tolist(),
            "baseline_glucose":baseline_glucose,
            "pre_slope": pre_slope,
            **{col: meal.get(col, np.nan) for col in meal_feature_cols},
            **summary_feats,
            **meta,
        }

        all_records.append(record)

    print(f"\n--- CGM window extraction -------------------------------")
    print(f"Valid windows extracted: {len(all_records)}")
    print(f"Excluded during extraction: {len(exclusion_log)}")
    if exclusion_log:
        print(pd.DataFrame(exclusion_log)["reason"].value_counts().to_string())
    print(f"\n--- Final dataset -------------------------------------------")
    print(f"Total records: {len(all_records)}")

    return all_records, pd.DataFrame(exclusion_log)

def engineer_features_task2(df):
    '''
    adds the interaction features, i.e. carbs x baseline,meal_dist_to_circadian etc
    encodes sex
    cleans the column names
    transforms the time :to hour  (0,24) and also sin cos time encoding
    '''
    # Column renames
    df = df.rename(columns={
        "<3.9":"lt_3.9",
        ">10":"gt_10",
    })
    encoder = LabelEncoder()
    df['sex'] = encoder.fit_transform(df['sex']) #female 0 male 1

    #Meal hour in decimal
    df["time_hour"] = pd.to_datetime(df["meal_time"]).dt.hour + \
                        pd.to_datetime(df["meal_time"]).dt.minute / 60

    # Circadian features
    meal_rad = (df["time_hour"] / 24) * (2 * np.pi)
    df["meal_dist_to_circadian"] = np.cos(meal_rad - df["acrophase"])
    df["meal_time_rad"] = meal_rad
    df["sin_time"]= np.sin(2 * np.pi * df["time_hour"] / df["period"])
    df["cos_time"]= np.cos(2 * np.pi * df["time_hour"] / df["period"])

    # Interaction features
    df["carbs_x_glucoseSD"]  = df["carbs_g"] * df["std"]
    df["carbs_x_baseline"]= df["carbs_g"] * df["baseline_glucose"]
    df["carbs_x_acrophase"]  = df["carbs_g"] * df["acrophase"]

    return df

#-----------------------------------------------------------------------
#MAIN ANALYSIS - TIME SERIES PREDICTION - LSTM and TRANSFORMER (global models)
#-----------------------------------------------------------------------

def make_ds(pre, scalars, targets):
    '''
    transforms data into tensordataset
    '''
    return TensorDataset(
        torch.tensor(pre, dtype=torch.float32),
        torch.tensor(scalars, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )

def optuna_search(pre_train, sc_train, tgt_train,
                  groups_train,n_trials=10, n_splits=3, nnmodel='LSTM'):
    """
    Bayesian TPE (Tree-structured Parzen Estimator)hyperparameter search using GroupKFold on the training fold.
    Returns the best parameters deictionary
    (worked better than randomsearch)

    Note : in the first task also tried this but there was no significant difference, so went with randomsearch
    to not to overcomplicate things

    group k fold for no leakege

    """
    n_scalars = sc_train.shape[1]
    gkf= GroupKFold(n_splits=n_splits)

    def objective(trial):
        if nnmodel == 'LSTM':  #hyperparameter space for LSTM
            params = {
                "hidden":trial.suggest_categorical("hidden", [64, 128]),
                "layers": trial.suggest_categorical("layers", [1, 2]),
                "dropout":trial.suggest_float("dropout", 0.1, 0.4),
                "lr":trial.suggest_float("lr",5e-4, 1e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
                "epochs":30,  # upper bound; but there is also early stopping 
            }
        else: #hyperparameter space for Transformer
            params = {
                "d_model":trial.suggest_categorical("d_model", [32, 64]),
                "nhead": trial.suggest_categorical("nhead", [2, 4]),   # nhead must divide d_model
                "num_layers": trial.suggest_categorical("num_layers", [1, 2]),
                "dropout":trial.suggest_float("dropout", 0.1, 0.4),
                "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),  # for Transformers lower lr better
                "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
                "epochs": 30,
            }

        fold_rmses = []
        for tr_idx, vl_idx in gkf.split(pre_train, groups=groups_train):

            # (data is already normalised)
            pre_tr = pre_train[tr_idx]
            pre_vl = pre_train[vl_idx]
            sc_tr = sc_train[tr_idx]
            sc_vl = sc_train[vl_idx]

            # reserve last 10 % of train rows for early stopping
            n_es= max(1, int(0.10 * len(tr_idx)))
            split_at = len(tr_idx) - n_es

            ds_tr = make_ds(pre_tr[:split_at], sc_tr[:split_at], tgt_train[tr_idx[:split_at]])
            ds_es = make_ds(pre_tr[split_at:], sc_tr[split_at:], tgt_train[tr_idx[split_at:]])
            ds_vl = make_ds(pre_vl, sc_vl, tgt_train[vl_idx])
            if nnmodel == 'LSTM':
                model = task2_LSTM(
                    n_scalars=n_scalars,
                    hidden=params["hidden"],
                    layers=params["layers"],
                    dropout=params["dropout"]
                )
            else :
                model = task2_Transformer(n_scalars=n_scalars, d_model=params["d_model"],
                          nhead=params["nhead"], num_layers=params["num_layers"],
                          dropout=params["dropout"])
            train_model(
                model,
                DataLoader(ds_tr, batch_size=params["batch_size"], shuffle=True),
                val_loader=DataLoader(ds_es, batch_size=64),
                epochs=params["epochs"],
                lr=params["lr"],
                patience=10
            )

            preds, targets = evaluate_model(model, DataLoader(ds_vl, batch_size=64))
            fold_rmses.append(np.sqrt(mean_squared_error(targets.flatten(), preds.flatten())))

        return np.mean(fold_rmses)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best["epochs"] = 30
    return best

#LSTM model
class task2_LSTM(nn.Module):
    def __init__(self, n_scalars, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1 + n_scalars,  # glucose + all scalars at every timestep
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 8)
        )

    def forward(self, pre_seq, scalars):
        x = pre_seq.unsqueeze(-1)               # (B, 5, 1) (B:batch size)
        sc_expanded = scalars.unsqueeze(1).expand(-1, 5, -1)  # (B, 5, n_scalars)
        x = torch.cat([x, sc_expanded], dim=-1)     # (B, 5, 1+n_scalars)
        out, _ = self.lstm(x)
        h_mean = out.mean(dim=1)
        return self.head(h_mean)

#Transformer Model
class PositionalEncoding(nn.Module):
    """
    For the sequential nature of the data as Transformer does not have it (seeing all at once)
    """
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class task2_Transformer(nn.Module):
    def __init__(self, n_scalars, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.input_proj = nn.Linear(1, d_model)
        self.scalar_proj = nn.Linear(n_scalars, d_model)  # projects scalars
        self.pos_enc = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * nhead,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 8)
        )

    def forward(self, pre_seq, scalars):
        x = pre_seq.unsqueeze(-1)     # (B, 5, 1)
        x = self.input_proj(x)       # (B, 5, d_model)
        x = self.pos_enc(x)
        
        s = self.scalar_proj(scalars)     # (B, d_model)
        s = s.unsqueeze(1)             # (B, 1, d_model)
        x = torch.cat([x, s], dim=1)     # (B, 6, d_model)
        
        x = self.encoder(x)        # (B, 6, d_model)
        h_pooled = x.mean(dim=1)  # (B, d_model)
        return self.head(h_pooled)

def train_model(model, train_loader, val_loader=None,
                epochs=30, lr=1e-3, patience=10):
    """
    Trains the NN model  optimizer and MSE loss with Adam. If val_loader is provided, apply early stopping
    (patience = number of epochs with no/little improvement before early stopping).
    Returns the model with the best validation weights.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
 
    best_val_loss = np.inf
    best_state= None
    no_improve = 0
 
    for epoch in range(epochs):
        #train step
        model.train()
        for pre, sc, tgt in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(pre, sc), tgt)
            loss.backward()
            opt.step()
 
        # early stopping on val set
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for pre, sc, tgt in val_loader:
                    val_loss += loss_fn(model(pre, sc), tgt).item()
            val_loss /= len(val_loader)
 
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break  # stop early
 
    # restore best weights early stopping
    if best_state is not None:
        model.load_state_dict(best_state)
 
    return model

def evaluate_model(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for pre, sc, tgt in loader:
            preds.append(model(pre, sc).numpy())
            targets.append(tgt.numpy())
    return np.vstack(preds), np.vstack(targets)

#Tranining with LOPO CV as in the first task
def lopo_splits(participants):
    for pid in np.unique(participants):
        test_idx = np.where(participants == pid)[0]
        train_idx = np.where(participants != pid)[0]
        yield pid, train_idx, test_idx #yield generates one by one 

def run_lopo(participants, pre_seq, scalars_raw, post_targets,
             subset_name, metadata, best_params, nnmodel='LSTM'):
    """
    Leave-One-Participant-Out cross-validation, scaling inside every fold for no leakage

    Parameters
    ----------
    participants : array of participant IDs, shape (N,)
    pre_seq: raw pre-meal glucose sequences, shape (N, 5)
    scalars_raw: raw scalar feature matrix, shape (N, F)
    post_targets: raw post-meal glucose targets, shape (N, 8)
    subset_name: feature subset (meal, meal+circadian...)
    metadata: DataFrame with Participant_id, age columns
    """
    all_preds, all_targets, all_pids = [], [], []
    results = []
 
    for pid, train_idx, test_idx in lopo_splits(participants):
 
        #Fit scalers on training fold so no leakega
        sc_scaler = StandardScaler()
        pre_scaler = StandardScaler()
 
        training_scalar = sc_scaler.fit_transform(scalars_raw[train_idx]).astype(np.float32)
        test_scalar = sc_scaler.transform(scalars_raw[test_idx]).astype(np.float32)
        training_pre_seq = pre_scaler.fit_transform(pre_seq[train_idx]).astype(np.float32)
        test_pre_seq = pre_scaler.transform(pre_seq[test_idx]).astype(np.float32)
        training_target= post_targets[train_idx].astype(np.float32)
        test_target = post_targets[test_idx].astype(np.float32)
 
        n_scalars  = training_scalar.shape[1]
        print(f"[{subset_name}] Evaluating participant {pid}")
 
        #left out participant
        test_ds = make_ds(test_pre_seq, test_scalar, test_target)
 
        # hold out 10 % of training for early stopping in final retrain too
        n_es = max(1, int(0.10 * len(train_idx)))
        ds_main = make_ds(training_pre_seq[:-n_es], training_scalar[:-n_es], training_target[:-n_es])
        ds_es = make_ds(training_pre_seq[-n_es:], training_scalar[-n_es:], training_target[-n_es:])
        if nnmodel == 'LSTM':
            model = task2_LSTM(
                n_scalars=n_scalars,
                hidden=best_params["hidden"],
                layers=best_params["layers"],
                dropout=best_params["dropout"]
            )
        else:
            model = task2_Transformer(n_scalars=n_scalars, d_model=best_params["d_model"],
                      nhead=best_params["nhead"], num_layers=best_params["num_layers"],
                      dropout=best_params["dropout"])
        train_model(
            model,
            DataLoader(ds_main, batch_size=best_params["batch_size"], shuffle=True),
            val_loader=DataLoader(ds_es, batch_size=64),
            epochs=best_params["epochs"],
            lr=best_params["lr"],
            patience=10
        )
 
        preds, targets = evaluate_model(model, DataLoader(test_ds, batch_size=64))
        all_preds.append(preds)
        all_targets.append(targets)
        all_pids.extend([pid] * len(preds))
 
        flat_p, flat_t = preds.flatten(), targets.flatten()
        results.append({
            "participant_id": pid,
            "subset": subset_name,
            "R2": r2_score(flat_t, flat_p),
            "MAE": mean_absolute_error(flat_t, flat_p),
            "RMSE": np.sqrt(mean_squared_error(flat_t, flat_p)),
        })
        print(f"R²={results[-1]['R2']:.3f}  MAE={results[-1]['MAE']:.3f}  RMSE={results[-1]['RMSE']:.3f}")
 
    results_df = pd.DataFrame(results)
    results_df = results_df.merge(
        metadata[["Participant_id", "age"]].rename(columns={"Participant_id": "participant_id"}),
        on="participant_id", how="left"
    )
    results_df["age_group"] = results_df["age"].apply(
        lambda a: "Age group 19-32" if a <= 32 else "Age group 59-76"
    )
    return results_df, np.vstack(all_preds), np.vstack(all_targets), np.array(all_pids)

def run_all_subsets(records_df, pre_seq, post_targets, participants,
                    metadata, feature_subset_columns, n_trials=10, nnmodel='LSTM'):
    """
    For each feature subset:
      - Runs a global Optuna search on ALL data using GroupKFold (participants are groups, so no participant leaks across folds).
      - Uses the best params found to run LOPO - for evaluation on participant level without leakage (only slightly in hyperparameter training).
 
    This separates hyperparameter selection from evaluation, which is standard
    practice in clinical ML with small N. The limitation (noted in the thesis)
    is that hyperparameters are not re-tuned per LOPO fold; to reduce computation from ~10,000 model trainings to ~320 + 10×3 = 350 total.
    """
    all_results = []
    best_params_by_subset = {}

    for subset_name, cols in feature_subset_columns.items():
        print(f"\n{'='*60}")
        print(f"Feature subset: {subset_name}({len(cols)} features)")
        print('='*60)

        scalars_raw = records_df[cols].values.astype(np.float32)

        pre_sc_global = StandardScaler()
        sc_sc_global = StandardScaler()
        pre_norm_global = pre_sc_global.fit_transform(pre_seq).astype(np.float32)
        sc_norm_global = sc_sc_global.fit_transform(scalars_raw).astype(np.float32)
        tgt_global = post_targets.astype(np.float32)

        print(f"Running global Optuna search on '{subset_name}' features ({n_trials} trials)")
        best_params = optuna_search(
            nnmodel=nnmodel,
            pre_train=pre_norm_global,
            sc_train=sc_norm_global,
            tgt_train=tgt_global,
            groups_train=participants,
            n_trials=n_trials
        )
        print(f"Global best params: {best_params}")
        best_params_by_subset[subset_name] = best_params

        results_df, preds, targets, pids = run_lopo(
            nnmodel=nnmodel,
            participants=participants,
            pre_seq=pre_seq,
            scalars_raw=scalars_raw,
            post_targets=post_targets,
            subset_name=subset_name,
            metadata=metadata,
            best_params=best_params
        )
        all_results.append(results_df)

    all_results_df = pd.concat(all_results, ignore_index=True)
    return all_results_df, preds, targets, pids, best_params_by_subset

def print_subset_summary(all_results_df):
    """
    Prints mean ± std of R², MAE, RMSE per feature subset.
    """
    print("\n" + "="*70)
    print("RESULTS SUMMARY BY FEATURE SUBSET")
    print("="*70)
    summary = (
        all_results_df
        .groupby("subset")[["R2", "MAE", "RMSE"]]
        .agg(["mean", "std"])
        .round(3)
    )
    print(summary.to_string())
    return summary

#-----------------------------------------------------------------------
#MAIN ANALYSIS - TIME SERIES PREDICTION - Residual Correction Fusion on LSTM
#-----------------------------------------------------------------------

def _get_cv(n):
    if n < 15:
        return LeaveOneOut()
    return KFold(n_splits=5, shuffle=True, random_state=42)
 
def _lstm_predict(model, pre_seq, scalars, batch_size=64):
    ds = make_ds(pre_seq, scalars, np.zeros((len(pre_seq), 8)))
    loader = DataLoader(ds, batch_size=batch_size)
    preds, _ = evaluate_model(model, loader)
    return preds  # (N, 8)
 
def _alpha_blend(oof_corr, residuals):

    '''
    fits the ridge regression if the length >3 (if too little samples alpha_blend is automatically =0)
    '''
    if len(residuals) < 3:
        return 0.0
    flat_oof = oof_corr.flatten()
    flat_res = residuals.flatten()
    ar = Ridge(alpha=1.0, fit_intercept=False)
    ar.fit(flat_oof.reshape(-1, 1), flat_res)
    return float(np.clip(ar.coef_[0], 0.0, 2.0))

def _log_flat(results, pid, subset_name, tgt, preds, alpha_blend, note=""):
    flat_p = preds.flatten()
    flat_t = tgt.flatten()
    results.append({
        "Participant": pid,
        "Subset": subset_name,
        "Fusion_R2": r2_score(flat_t, flat_p),
        "Fusion_RMSE": np.sqrt(mean_squared_error(flat_t, flat_p)),
        "Fusion_MAE": mean_absolute_error(flat_t, flat_p),
        "Alpha_Blend": alpha_blend,
        "N_Samples": len(tgt),
        "Note": note,
    })
#-----------------------------------------------------------------------
#flattened multi-output residual correction
# corrector sees X_scalars -> predicts all 8 residuals simultaneously
# MultiOutputRegressor wraps Ridge to handle (N, 8) targets
#-----------------------------------------------------------------------
def fusion_lstm_lopo_flat(
    participants,
    pre_seq,
    scalars_raw,
    post_targets,
    subset_name,
    best_params,
    min_samples_personal=10,
    corrector_alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
    nnmodel='LSTM'
):
    
    '''
    Residual correction on LSTM model (better performing or chosen of the two : LSTM and Transformer)
    Same logic as the scalar iAUC's residual correction
    '''
    results = []
 
    for pid, train_idx, test_idx in lopo_splits(participants):
        n_p = len(test_idx)
        print(f"\n[flat fusion | {subset_name}] pid={pid}  n_personal={n_p}  n_train={len(train_idx)}")
 
        sc_scaler = StandardScaler()
        pre_scaler = StandardScaler()
 
        sc_train = sc_scaler.fit_transform(scalars_raw[train_idx]).astype(np.float32)
        sc_test = sc_scaler.transform(scalars_raw[test_idx]).astype(np.float32)
        pre_train = pre_scaler.fit_transform(pre_seq[train_idx]).astype(np.float32)
        pre_test = pre_scaler.transform(pre_seq[test_idx]).astype(np.float32)
        tgt_train = post_targets[train_idx].astype(np.float32)
        tgt_test = post_targets[test_idx].astype(np.float32)
 
        n_es = max(1, int(0.10 * len(train_idx)))
        ds_main = make_ds(pre_train[:-n_es], sc_train[:-n_es], tgt_train[:-n_es])
        ds_es = make_ds(pre_train[-n_es:], sc_train[-n_es:], tgt_train[-n_es:])
 
        model = task2_LSTM(n_scalars=sc_train.shape[1],
                               hidden=best_params["hidden"],
                               layers=best_params["layers"],
                               dropout=best_params["dropout"])

 
        train_model(model,
                    DataLoader(ds_main, batch_size=best_params["batch_size"], shuffle=True),
                    val_loader=DataLoader(ds_es, batch_size=64),
                    epochs=best_params["epochs"],
                    lr=best_params["lr"],
                    patience=10)
 
        # global (LOO) predictions on this participant
        global_preds = _lstm_predict(model, pre_test, sc_test)  # (n_p, 8), n_p=n_meals of that participant
        residuals = tgt_test - global_preds  # (n_p, 8)
 
        global_flat_p = global_preds.flatten()
        global_flat_t = tgt_test.flatten()
        print(f"global R²={r2_score(global_flat_t, global_flat_p):.3f}"
              f"RMSE={np.sqrt(mean_squared_error(global_flat_t, global_flat_p)):.3f}"
              f"MAE={mean_absolute_error(global_flat_t, global_flat_p):.3f}")
 
        if n_p < min_samples_personal:
            print(f"n < {min_samples_personal}: global only (α=0)")
            _log_flat(results, pid, subset_name, tgt_test, global_preds, alpha_blend=0.0,
                      note="global-fallback")
            continue
 
        # corrector: scalars -> 8 residuals (multi-output)
        # use raw test scalars as features for the personal corrector;
        # they are already scaled by sc_scaler above
        X_corr = sc_test  # (n_p, n_features)
 
        cv = _get_cv(n_p)
        best_alpha= corrector_alphas[0]
        best_mse=np.inf
        for a in corrector_alphas:
            mo = MultiOutputRegressor(Ridge(alpha=a))
            fold_mses = []
            for tr, vl in cv.split(X_corr):
                mo.fit(X_corr[tr], residuals[tr])
                pred = mo.predict(X_corr[vl])
                fold_mses.append(np.mean((residuals[vl] - pred)**2))
            score = np.mean(fold_mses)
            if score < best_mse:
                best_mse = score
                best_alpha = a
 
        # OOF predictions for alpha blending
        oof_corr = np.zeros_like(residuals)
        for tr, vl in cv.split(X_corr):
            c = MultiOutputRegressor(Ridge(alpha=best_alpha))
            c.fit(X_corr[tr], residuals[tr])
            oof_corr[vl] = c.predict(X_corr[vl])
 
        alpha_blend = _alpha_blend(oof_corr, residuals)
        print(f"α={alpha_blend:.3f}  best_corrector_alpha={best_alpha}")
 
        fusion_preds = global_preds + alpha_blend * oof_corr
        flat_fp = fusion_preds.flatten()
        flat_t = tgt_test.flatten()
        print(f"fusion R²={r2_score(flat_t, flat_fp):.3f}"
              f"RMSE={np.sqrt(mean_squared_error(flat_t, flat_fp)):.3f}"
              f"MAE={mean_absolute_error(flat_t, flat_fp):.3f}")
 
        _log_flat(results, pid, subset_name, tgt_test, fusion_preds, alpha_blend=alpha_blend)
 
    results_df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f"FLAT FUSION LOPO | {subset_name}")
    print(f"Mean R²: {results_df['Fusion_R2'].mean():.3f} ± {results_df['Fusion_R2'].std():.3f}")
    print(f"Mean RMSE: {results_df['Fusion_RMSE'].mean():.3f}")
    print(f"Mean MAE: {results_df['Fusion_MAE'].mean():.3f}")
    print('='*60)
    return results_df
 