import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Notes:
# 1. Don't use loops. There is almost always an alternative in Pandas

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
bmi = df['weight'] / ((df['height'] * 0.01) ** 2)
df['overweight'] = bmi > 25

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df[df[['cholesterol', 'gluc']] == 1] = 0
df[df[['cholesterol', 'gluc']] > 1] = 1


# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    counts = df_cat[['cardio', 'variable', 'value']].value_counts()
    new_df = pd.DataFrame()
    new_df['cardio'] = list(map(lambda x: x[0], counts.index.values))
    new_df['variable'] = list(map(lambda x: x[1], counts.index.values))
    new_df['value'] = list(map(lambda x: x[2], counts.index.values))
    new_df['total'] = counts.values
    new_df = new_df.sort_values(by='variable')

    # Get the figure for the output
    fig = sns.catplot(data=new_df, x='variable', y='total', col='cardio', kind='bar', hue='value')

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df.copy()
    df_heat = df_heat[df_heat['ap_lo'] <= df_heat['ap_hi']]
    height_condition = (df_heat['height'] >= df['height'].quantile(0.025)) & (df_heat['height'] <= df['height'].quantile(0.975))
    df_heat = df_heat[height_condition]
    weight_condition = (df_heat['weight'] >= df['weight'].quantile(0.025)) & (df_heat['weight'] <= df['weight'].quantile(0.975))
    df_heat = df_heat[weight_condition]
    
    # Calculate the correlation matrix
    corr = df_heat.corr()
    
    # Generate a mask for the upper triangle
    upper_triangle = pd.DataFrame(np.arange(corr.size).reshape(corr.shape), columns = corr.columns, index=corr.index)
    cond = upper_triangle % upper_triangle.shape[0] >= upper_triangle // upper_triangle.shape[1]
    upper_triangle = upper_triangle.mask(cond)
    mask = upper_triangle.mask(upper_triangle > 0, 0)
    corr_with_mask = corr + mask

    # Set up the matplotlib figure
    fig, ax = plt.subplots()
    # Draw the heatmap with 'sns.heatmap()'
    ax = sns.heatmap(corr_with_mask, annot=True, fmt='.1f')

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
